from __future__ import annotations

from smolagents import Tool
import subprocess
import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from agentflow.tools.session_manager import get_current_session_id
import glob
import json
import openai
from openai import OpenAI
import shlex
import re
import shutil
import hashlib
import time
from smolagents import LiteLLMModel, CodeAgent
from agentflow.models import TokenTrackingModel
from agentflow.models.deepseek_reasoner import create_model

system_case_prompts = "You are a helpful assistant help user generate test examples for coding tasks."
Generator_Agent_prompt = """You are an expert test case designer for competitive programming contests. Your task is to create a C++ generator program using the `testlib.h` library and a corresponding list of commands to produce a comprehensive set of test cases for the problem described below.\n\n### Workflow and Instructions\n\nYour process should be as follows:\n\n**1. Analyze the Problem and Constraints:**\n* Carefully read the provided problem description.\n* Identify and summarize all input variables and their constraints (e.g., $1 \\le n \\le 10^5$).\n\n**2. Brainstorm Adversarial and Corner Cases:**\n* Anticipate potential mistakes contestants might make.\n* Consider edge cases based on the constraints (e.g., minimum/maximum values).\n* Identify different structural patterns for the input data that could challenge common algorithms. For a graph problem, this might include chains, star graphs, complete graphs, disconnected components, etc. For an array problem, this could be sorted, reversed, or all identical elements.\n\n**3. Design and Implement the Generator Program:**\n* Based on your analysis, define the command-line arguments your generator will accept. This should include parameters for data size (e.g., `-n`, `-m`) and a `-type` parameter to specify the kind of test case to generate (e.g., `-type random`, `-type chain`).\n* Write a complete C++ generator program that implements this design.\n\n**Generator Implementation Rules:**\n* You **MUST** use the `testlib.h` library. Start your program with `#include \"testlib.h\"` and `registerGen(argc, argv, 1);`.\n* You **MUST** use `opt<T>(\"param_name\")` to parse command-line arguments.\n* You **MUST** use `rnd.next(...)` from `testlib.h` for all random number generation. Do not use standard C++ random functions like `rand()` or `<random>`.\n* The generator program **MUST NOT** parse or set a random seed. This is handled externally to ensure reproducibility.\n* The program must generate **exactly one** test case to standard output per execution.\n\n**4. Create the Command List:**\n* Provide a list of approximately 20 distinct commands to execute your generator.\n* This list should be diverse and cover all the categories you brainstormed, including:\n    * Minimum and maximum constraint values.\n    * Randomly generated \"average\" cases for various sizes (small, medium, large).\n    * All special `type`s you implemented.\n    * Challenging and adversarial test cases.\n\n### Problem Statement\n\n```\n{problem_to_solve}\n```\n\n### Required Output Format\n\nYour entire response must be a single block of text containing two distinct parts in the following order:\n\n1.  The complete C++ generator source code, enclosed in a C++ markdown block.\n2.  The list of shell commands, enclosed in a plain markdown block.\n\n**Example Structure:**\n\n```cpp\n// C++ generator code starts here.\n#include \"testlib.h\"\n#include <vector>\n// ... rest of the generator code\n```\n\n```commands\n# Command list starts here.\n./gen -n 1 -m 0 -type random\n./gen -n 100000 -m 99999 -type chain\n# ... rest of the commands\n```\n"""

MODEL_NAME = "deepseek-chat"
API_BASE = "https://api.deepseek.com/v1"
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")

class CasesInGeneratorTool(Tool):
    """
    Tool that generates unit test cases and a checker from an algorithm question description.

    Exposes a single public method `forward()` as the entry point for agents or callers.
    All other methods are internal helpers with signatures and documentation only, and
    intentionally omit detailed implementations in this initial scaffold.
    """

    name = "unit_test_generator"
    description = (
"Generate comprehensive test cases for competitive programming problems."
"This tool takes an algorithmic problem description and automatically creates a complete test suite, including: (1) a C++ test-case generator program using testlib.h that can produce various types of test cases (random, boundary, adversarial), (2) a set of commands to run the generator and create multiple test inputs, (3) a brute-force solution used to generate expected outputs for verification, and (4) a pass-rate analysis to ensure test quality. The tool automatically handles compilation, execution, and error handling. Use this tool when you need to build thorough test cases for an algorithm problem, especially for competitive programming. The generated test cases are saved in the TestCases directory, including input files and the corresponding correct outputs for validation."
 )
    inputs = {
        "question": {
            "type": "string",
            "description": "Algorithm problem description used to generate tests and checker.",
        },
        "language": {
            "type": "string",
            "description": "Specify the language of the checker implementation.",
        },
        "conversation_id": {
            "type": "string",
            "description": "The conversation ID for this session. If provided, will use this ID to save test files.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model_id: str = None, 
                 api_base: str = None,
                 api_key: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.model_id = model_id or MODEL_NAME
        self.api_base = api_base or API_BASE
        self.api_key = api_key or API_KEY
        
        model = create_model(
            model_id=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            track_tokens=True,
            token_source="test_generator"
        )
        
        self.agent = CodeAgent(
            tools=[],  
            model=model,
            max_steps=1,  
            verbosity_level=0,  
            add_base_tools=False,
        )
        
        self._current_session_id = None
        self._output_dir = None
        print(f"UnitTestGenerator initialized with model: {MODEL_NAME}")

    def _set_output_dir(self, question: str, conversation_id: str):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        unit_test_data_dir = os.path.join(project_root, "tools", "cpp", "unit_test_data", conversation_id)
        os.makedirs(unit_test_data_dir, exist_ok=True)
        self._output_dir = unit_test_data_dir
        print(f"Set output directory to: {unit_test_data_dir}")
        return unit_test_data_dir

    def _set_session_id(self, session_id: str):
        self._current_session_id = session_id

    def _get_session_id(self) -> Optional[str]:
        return self._current_session_id

    def _generate_conversation_id(self, problem_text: str) -> str:
        timestamp = str(int(time.time() * 1000))
        problem_hash = hashlib.md5(problem_text.encode()).hexdigest()[:8]
        return f"auto_{timestamp}_{problem_hash}"


    def forward(self, question: str, language: str, conversation_id: Optional[str] = None) -> str:
        """
        Generate unit test cases and a checker given an algorithm question description.

        Args:
            question: The full text of the algorithm problem statement.
            language: Specify the language of the checker implementation.

        Returns:
            A string message indicating the result of the test case generation.
        """
        try:
            if not question:
                return "Error: Empty question text provided"
            
            conversation_id = self._get_session_id()
            if not conversation_id:
                try:
                    conversation_id = get_current_session_id()
                except ImportError:
                    pass
            if not conversation_id:
                conversation_id = self._generate_conversation_id(question)
            
            self._set_session_id(conversation_id)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = self._set_output_dir(question, conversation_id)
            cache_unit_test_cases = os.path.join(out_dir, "cache_unitcases.json")

            # Generate test cases
            test_times = 0
            while(test_times < 5):
                print("="*20 + f" Generating test cases... {test_times} times " + "="*20)
                test_cases = self._generate_test_cases_stub(question, cache_unit_test_cases, self._output_dir)
                print(f"Number of test cases: {len(test_cases)}")
                cpp_file, commands_file = self._save_cpp_and_commands_from_test_cases(test_cases, out_dir=self._output_dir)
                # Compile the cpp file using g++ and deal with potential exception
                compile_success, compile_error, test_case_exec_path = self._compile_cpp_file(cpp_file)
                if not compile_success:
                    print(f"Compile error: {compile_error}. Continue to generate test cases...")
                    test_times += 1
                    continue
                # Run the compiled program with the commands file
                run_success, run_error = self._run_program(commands_file, self._output_dir)
                if not run_success:
                    print(f"Run error: {run_error}. Continue to generate test cases...")
                    test_times += 1
                    continue
                print(f"Run success: {run_success}")
                break
            test_times = 0
            # Generate brute-force solution first
            # Support retry for brute-force solution generation and compilation
            while test_times < 5:
                bf_solution_file = self._generate_bf(question)
                if not bf_solution_file:
                    print("Failed to generate brute-force solution, retrying...")
                    test_times += 1
                    continue
                compile_success, compile_error, bf_solution_exec_path = self._compile_cpp_file(bf_solution_file)
                if not compile_success:
                    print(f"Compile error: {compile_error}, retrying brute-force solution generation...")
                    test_times += 1
                    continue
                # If both generation and compilation succeed, break out of the loop
                break
            else:
                # If we exhausted all retries
                raise RuntimeError("Failed to generate and compile brute-force solution after multiple attempts.")

            output_json_path = os.path.join(self._output_dir, 'golden_outputs.json')
            print(f"output_json_path: {output_json_path}")
            # Read the brute-force solution code
            with open(bf_solution_file, 'r') as f:
                bf_solution_code = f.read()
            self._generate_outputs_for_testcases(
                TestCases_dir=self._output_dir,
                bf_solution_exec_path=bf_solution_exec_path,
                output_json_path=output_json_path,
                question=question,
                bf_solution=bf_solution_code
            )   

            pass_rate = self._compute_pass_rate_from_golden_outputs(output_json_path)
            print(f"golden_outputs pass_rate: {pass_rate:.2%}")
            if pass_rate > 0.7:
                return f"Test cases generation succeed. They are available on path '{output_json_path}'. Pass rate: {pass_rate:.2%}"
        except Exception as e:
            print(f"Error: {e}")
            return f"Test cases generation failed: {str(e)}"
    
    def _generate_outputs_for_testcases(self, TestCases_dir: str, bf_solution_exec_path: str, output_json_path: str, question: str, bf_solution: str):
        """
        For each file in TestCases_dir ending with '_stdout.txt', read its content as input,
        run bf_solution_exec_path (a GNU C++ compiled executable) with that input, and save the output in a JSON file.

        Args:
            TestCases_dir: Directory containing test case input files (with '_stdout.txt' suffix).
            bf_solution_exec_path: Path to the compiled brute-force solution executable (GNU C++ output).
            output_json_path: Path to save the resulting JSON file.
            question: The algorithm problem statement.
            bf_solution: The code of the brute-force solution.
        """
        golden_io = []
        input_files = glob.glob(os.path.join(TestCases_dir, 'TestCases', '*_stdout.txt'))
        print(f"Extract input files from {TestCases_dir}: {input_files}")
        print(f"Number of files to read: {len(input_files)}")
        def run_single_case(input_file):
            try:
                with open(input_file, 'r') as f:
                    input_data = f.read()
                
                # Ensure the executable has execute permissions
                if not os.access(bf_solution_exec_path, os.X_OK):
                    os.chmod(bf_solution_exec_path, 0o755)
                
                # Run the GNU C++ compiled executable with the input
                proc = subprocess.run(
                    [bf_solution_exec_path],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                output = proc.stdout
                
                # If error, store error as output, else store output
                if proc.returncode != 0:
                    error_parts = []
                    error_parts.append(f"Exit code: {proc.returncode}")
                    if proc.stderr and proc.stderr.strip():
                        error_parts.append(f"Stderr: {proc.stderr.strip()}")
                    if proc.stdout and proc.stdout.strip():
                        stdout_content = proc.stdout.strip()
                        if any(keyword in stdout_content.lower() for keyword in 
                               ['error', 'exception', 'fault', 'segmentation', 'abort', 'terminate']):
                            error_parts.append(f"Stdout: {stdout_content}")
                    
                    if proc.returncode == -11:
                        error_parts.append("Error type: Segmentation fault (SIGSEGV)")
                    elif proc.returncode == -6:
                        error_parts.append("Error type: Abort signal (SIGABRT)")
                    elif proc.returncode == -9:
                        error_parts.append("Error type: Process killed (SIGKILL)")
                    elif proc.returncode == 1:
                        error_parts.append("Error type: General error (exit code 1)")
                    elif proc.returncode == 2:
                        error_parts.append("Error type: Misuse of shell builtins (exit code 2)")
                    elif proc.returncode == 126:
                        error_parts.append("Error type: Command invoked cannot execute (exit code 126)")
                    elif proc.returncode == 127:
                        error_parts.append("Error type: Command not found (exit code 127)")
                    
                    error_msg = " | ".join(error_parts) if error_parts else f"Process exited with code {proc.returncode}"
                    try:
                        error_msg = str(error_msg)
                    except UnicodeDecodeError:
                        error_msg = f"Unicode decode error in error message (return code: {proc.returncode})"
                    
                    return {"input": input_data, "output": f"ERROR: {error_msg}"}
                else:
                    return {"input": input_data, "output": output}
            except subprocess.TimeoutExpired as e:
                return {"input": input_data, "output": f"TIMEOUT: Process timed out after {e.timeout} seconds"}
            except Exception as e:
                return {"input": input_data, "output": f"EXCEPTION: {str(e)}"}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        cpu_count = os.cpu_count() or 4
        max_workers = min(cpu_count, len(input_files))
        
        results = [None] * len(input_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(run_single_case, f): i for i, f in enumerate(input_files)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        
        golden_io = [r for r in results if r is not None]
        result_json = {
            "question": question,
            "bf_solution": bf_solution,
            "compiled_file": bf_solution_exec_path,
            "golden i/o": golden_io
        }
        with open(output_json_path, 'w') as f:
            json.dump(result_json, f, indent=2)

    def _compute_pass_rate_from_golden_outputs(self, output_json_path: str) -> float:
        """
        Compute pass rate from golden_outputs.json.
        Also filter out items that do not pass, and save the passing items to final_cases.json
        in the same directory as the source file.
        An item is considered failed if its output is an empty string or contains
        words like 'error' or 'exception' (case-insensitive). Otherwise it passes.
        Returns a float in [0,1].
        """
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0.0

        items = []
        if isinstance(data, dict):
            items = data.get("golden i/o", [])
        total = 0
        passed = 0
        passed_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            total += 1
            output = item.get("output", "")
            if output is None:
                output_str = ""
            else:
                output_str = str(output)
            normalized = output_str.strip()
            # Fail if output is empty
            if not normalized:
                continue
            # Fail if output contains error/exception
            if re.search(r"(error|exception)", normalized, flags=re.IGNORECASE):
                continue
            # Otherwise, pass
            passed += 1
            passed_items.append(item)

        # Save the filtered (passed) items to final_cases.json in the same directory as output_json_path
        final_cases_path = os.path.join(os.path.dirname(os.path.abspath(output_json_path)), "final_cases.json")
        filtered_json = {
            "question": data.get("question", "") if isinstance(data, dict) else "",
            "bf_solution": data.get("bf_solution", "") if isinstance(data, dict) else "",
            "compiled_file": data.get("compiled_file", "") if isinstance(data, dict) else "",
            "golden i/o": passed_items
        }
        try:
            with open(final_cases_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_json, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # If saving fails, just print error and continue
            print(f"Warning: Could not save filtered cases to {final_cases_path}: {e}")

        if total == 0:
            return 0.0
        return passed / total

    def _generate_bf(self, question: str) -> str:
        """
        Generate brute-force C++ solution using OpenAI API.
        Merged functionality from generate_bf.py directly into this method.

        Args:
            question: The algorithm problem statement.

        Returns:
            The path to the generated C++ code file if successful, otherwise an empty string.
        """
        # User prompt template for brute-force generation
        user_prompt = """
        You are an expert competitive programming assistant.

        Your task is to write a correct and *brute-force* C++ solution for the following problem. The program must:
        - Read input from standard input and write output to standard output, strictly following the problem's input/output format.
        - Implement a straightforward, brute-force (but correct) algorithm, prioritizing clarity and correctness over efficiency.
        - Avoid any optimizations or advanced algorithms; use only simple, exhaustive approaches.
        - Ensure the code is self-contained and compiles with GNU G++23 14.2 (64bit, msys2).

        ---

        {Question}

        ---

        **Language and Compiler**

        * **Language:** cpp
        * **Compiler:** GNU G++23 14.2 (64bit, msys2)

        **Output:**  
        Provide only the complete C++ source code, enclosed in a single code block using the syntax ```cpp ... ```. Do not include any explanations or comments outside the code.
        """

        def extract_code_from_response(response_text: str) -> str:
            """Extract C++ code from API response, removing markdown fences if present."""
            try:
                if not isinstance(response_text, str):
                    return "We can not extract the code in the output. (raw_response is not a string)"
                
                # Pattern to match ```cpp ... ``` blocks
                pattern = r"```cpp\n(.*?)```"
                matches = re.findall(pattern, response_text, re.DOTALL)
                
                if matches:
                    # Get the last match (in case there are multiple code blocks)
                    code_output = matches[-1].strip()
                else:
                    # Fallback: try to find any code block without language specification
                    pattern_fallback = r"```\n(.*?)```"
                    matches_fallback = re.findall(pattern_fallback, response_text, re.DOTALL)
                    if matches_fallback:
                        code_output = matches_fallback[-1].strip()
                    else:
                        code_output = "We can not extract the code in the output."
                
                return code_output
                
            except Exception as e:
                return f"We can not extract the code in the output. (Exception: {e})"

        final_output = os.path.join(self._output_dir, "bf_solution.cpp")
        
        # Helper function for API call with infinite retry
        def call_api_with_retry():
            try:
                print("Generating brute-force solution using API...")
                
                # Construct prompts
                system_prompt = "You are an elite competitive-programming assistant.\nProduce only valid C++17 code.\nUse standard I/O (cin/cout).\nInclude every helper necessary for a brute-force solution.\nDo NOT write explanations or markdown fences."
                formatted_user_prompt = user_prompt.format(Question=question)
                
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                )
                
                # Send request
                completion = client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': formatted_user_prompt}
                    ],
                    temperature=0.6
                )
                
                # Extract response
                return completion.choices[0].message.content, system_prompt, formatted_user_prompt
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                print(f"[unit_test_generator] API Error: {repr(e)}")
                print("Sleeping for 30 seconds before retry...")
                from time import sleep
                sleep(30)
                return call_api_with_retry()
            except Exception as e:
                print(f"[unit_test_generator] Fatal error (non-retryable): {repr(e)}")
                raise e
        
        try:
            # Call API with infinite retry
            response_content, system_prompt, formatted_user_prompt = call_api_with_retry()
            
            # Log conversation to file
            log_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "request": {
                    "system_prompt": system_prompt,
                    "user_prompt": formatted_user_prompt,
                    "model": self.model_id,
                    "temperature": 0.6
                },
                "response": {
                    "content": response_content
                }
            }
            
            # Write to log file
            log_filename = os.path.join(self._output_dir, "log", "generator_chat_log.json")
            try:
                if os.path.exists(log_filename):
                    with open(log_filename, 'r', encoding='utf-8') as f:
                        log_history = json.load(f)
                else:
                    log_history = []
                
                log_history.append(log_data)
                
                with open(log_filename, 'w', encoding='utf-8') as f:
                    json.dump(log_history, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")
            
            # Extract and write code
            code = extract_code_from_response(response_content)
            
            # Write the extracted C++ code to the output file
            os.makedirs(self._output_dir, exist_ok=True)
            with open(final_output, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print("✓ Brute-force C++ code generated successfully")
            print(f"Code written to {final_output}")
            return final_output
            
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def _compile_cpp_file(self, cpp_file):
        compile_success = False
        compile_error = ""
        exec_path = ""
        print(f"cpp_file: {cpp_file}")
        if cpp_file is not None:
            try:
                # Remove .cpp extension for exec_path
                base_name = os.path.splitext(os.path.basename(cpp_file))[0]
                # Resolve directories
                base_dir = os.path.dirname(os.path.abspath(__file__))
                include_dir = os.path.join(base_dir, "utils", "testlib")
                exec_dir = self._output_dir
                exec_path = os.path.join(exec_dir, base_name)
                # Add absolute -I to the compile command to search for headers in utils/testlib directory
                compile_cmd = [
                    "g++", "-O2", "-std=c++17",
                    f"-I{include_dir}",
                    "-o", exec_path,
                    cpp_file,
                ]
                print(f"compile_cmd: {compile_cmd}")
                result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=exec_dir)
                if result.returncode == 0:
                    compile_success = True
                    # exec_path already absolute
                else:
                    compile_error = result.stderr
                    exec_path = ""
            except Exception as e:
                compile_error = str(e)
                exec_path = ""
        else:
            print("cpp_file is None")
        print(f"compile_success: {compile_success}")
        print(f"compile_error: {compile_error}")
        print(f"exec_path: {exec_path}")
        return compile_success, compile_error, exec_path

    def _run_program(self, commands_file: str, base_dir: str):

        if not commands_file or not os.path.exists(commands_file):
            return False, f"commands file not found: {commands_file}"

        def _sanitize_filename(s):
            # Remove or replace characters not allowed in filenames
            s = re.sub(r'[^\w\-.]', '_', s)
            return s

        def _run_single_command(command_line: str, timeout_seconds: int = 120):

            testcases_dir = os.path.join(base_dir, "TestCases")
            os.makedirs(testcases_dir, exist_ok=True)

            try:
                parts = shlex.split(command_line)
                if not parts:
                    prog = "unknown"
                    args = []
                else:
                    prog = os.path.basename(parts[0])
                    args = parts[1:]
                arg_part = "_".join([_sanitize_filename(a) for a in args])
                if arg_part:
                    base_name = f"{prog}__{arg_part}"
                else:
                    base_name = prog
                if len(base_name) > 128:
                    base_name = base_name[:128]
                stdout_file = os.path.join(testcases_dir, f"{base_name}_stdout.txt")
                stderr_file = os.path.join(testcases_dir, f"{base_name}.stderr.txt")
            except Exception as e:
                # fallback
                stdout_file = os.path.join(testcases_dir, "unknown_stdout.txt")
                stderr_file = os.path.join(testcases_dir, "unknown.stderr.txt")

            try:
                completed = subprocess.run(
                    command_line,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=base_dir,
                )
                # stdout、stderr
                with open(stdout_file, "w", encoding="utf-8") as f:
                    f.write(completed.stdout or "")
                with open(stderr_file, "w", encoding="utf-8") as f:
                    f.write(completed.stderr or "")
                return completed.returncode, completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as e:
                with open(stdout_file, "w", encoding="utf-8") as f:
                    f.write(e.stdout or "")
                with open(stderr_file, "w", encoding="utf-8") as f:
                    f.write(e.stderr or f"Command timed out after {timeout_seconds}s")
                return 124, e.stdout or "", e.stderr or f"Command timed out after {timeout_seconds}s"
            except Exception as e:
                with open(stderr_file, "w", encoding="utf-8") as f:
                    f.write(str(e))
                return 1, "", str(e)

        try:
            with open(commands_file, "r", encoding="utf-8") as f:
                for raw_line in f:
                    cmd = raw_line.strip()
                    if not cmd or cmd.startswith('#'):
                        continue
                    code, out, err = _run_single_command(cmd)
                    if out:
                        print(out, end="" if out.endswith("\n") else "\n")
                    if code != 0:
                        error_msg = f"Command failed (exit {code}): {cmd}\n{err}"
                        return False, error_msg
            return True, ""
        except Exception as e:
            return False, str(e)

    def _save_cpp_and_commands_from_test_cases(self, test_cases, cpp_filename="gen.cpp", commands_filename="commands_to_execute.txt", out_dir: Optional[str] = None):
        """
        Given a test_cases list (from cache), extract the latest cpp_code and commands,
        and write them to files. Returns the absolute path of the cpp file and the commands file.
        """
        if not test_cases:
            return None, None
        # If test_cases is a list of dicts, find the last one with cpp_code and commands
        last = None
        for item in reversed(test_cases):
            if isinstance(item, dict) and "cpp_code" in item and "commands" in item:
                last = item
                break
        if last is None:
            return None, None
        cpp_code = last["cpp_code"]
        commands = last["commands"]
        # Resolve out_dir
        if out_dir is None:
            out_dir = self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        cpp_path = os.path.join(out_dir, cpp_filename)
        commands_path = os.path.join(out_dir, commands_filename)
        # Write cpp_code to file
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(cpp_code)
        # Write commands to file, one per line
        with open(commands_path, "w", encoding="utf-8") as f:
            for cmd in commands:
                f.write(cmd.strip() + "\n")
        abs_cpp_filename = os.path.abspath(cpp_path)
        abs_commands_path = os.path.abspath(commands_path)
        return abs_cpp_filename, abs_commands_path
        

    def _load_last_test_cases(self, cache_unit_test_cases):
        import json
        if not os.path.exists(cache_unit_test_cases):
            return []
        with open(cache_unit_test_cases, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                last_item = data[-1]
                if "test_cases" in last_item:
                    return last_item["test_cases"]
                elif isinstance(last_item, list):
                    return last_item
                else:
                    return [last_item]
            else:
                return []

    def _generate_test_cases_stub(self, question: str, cache_unit_test_cases: str, out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate test cases using the agent instead of direct API calls.

        Args:
            question: The algorithm problem statement.
            cache_unit_test_cases: Path to cache file for storing test case results.
            out_dir: Output directory for generated files.

        Returns:
            A list of test case dictionaries containing cpp_code and commands.
        """

        def _parse_cpp_code(language: str, raw_response: str) -> str:
            """
            Parse model output into a checker specification capable of evaluating predictions.

            Args:
                language: The programming language (not used in current implementation).
                raw_response: The raw text output that includes the checker definition.

            Returns:
                A checker specification (e.g., code string, rules, or a normalized schema).
            """
            # Exception-safe extraction of C++ code block
            try:
                if not isinstance(raw_response, str):
                    return "We can not extract the code in the output. (raw_response is not a string)"
                pattern = r"```cpp\n(.*?)```"
                matches = re.findall(pattern, raw_response, re.DOTALL)
                if matches:
                    code_output = matches[-1].strip()
                else:
                    code_output = "We can not extract the code in the output. "
                return code_output
            except Exception as e:
                return f"We can not extract the code in the output. (Exception: {e})"

        def _parse_commands(response_string: str) -> List[str]:
            """
            Parses a string to find a commands block and extract the commands.

            The function looks for a markdown block starting with ```commands
            and ending with ```. It cleans the output by removing comments
            (lines starting with '#') and empty lines.

            Args:
                response_string: The string containing the commands block.

            Returns:
                A list of command strings, or an empty list if no block is found.
            """
            # Exception-safe extraction of commands block
            try:
                if not isinstance(response_string, str):
                    return []
                # Pattern to find the content within ```commands ... ```
                match = re.search(r"```commands\n(.*?)\n```", response_string, re.DOTALL)
                if not match:
                    return []

                # Extract the raw content of the commands block
                raw_commands = match.group(1)

                # Process the commands
                command_list = []
                for line in raw_commands.splitlines():
                    # Remove leading/trailing whitespace
                    stripped_line = line.strip()
                    # Add the line if it's not empty and not a comment
                    if stripped_line and not stripped_line.startswith('#'):
                        command_list.append(stripped_line)

                return command_list
            except Exception as e:
                # If any error occurs, return an empty list
                return []

        system_prompt = system_case_prompts
        user_prompt = Generator_Agent_prompt.format(problem_to_solve=question)
        
        # Combine system and user prompts for agent call
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Use agent instead of direct API call
        api_response = self.agent.run(full_prompt)
        
        print("="*20+"Response Content"+"="*20)
        print(api_response)
        print("="*20+"Agent call completed"+"="*20)
        
        # Store the data into the json file in an appending way, including model name
        data_to_store = {
            "prompt": full_prompt,
            "response": api_response,
            "model": MODEL_NAME
        }
        try:
            if out_dir is None:
                out_dir = self._output_dir
            os.makedirs(out_dir, exist_ok=True)
            temp_chat_log = os.path.join(out_dir, "_temp_chat_log.json")
            with open(temp_chat_log, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        temp_chat_log = os.path.join(out_dir, "_temp_chat_log.json")
        existing_data.append(data_to_store)
        with open(temp_chat_log, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        # Save the extracted cpp code and commands into a JSON log file for record-keeping, including model name
        try:
            cpp_code = _parse_cpp_code("cpp", api_response)
            commands = _parse_commands(api_response)
        except Exception as e:
            print(f"Error occurred while parsing cpp code or commands: {e}")
            cpp_code = None
            commands = None
        gen_data_to_store = {
            "cpp_code": cpp_code,
            "commands": commands,
            "timestamp": datetime.datetime.now().isoformat(),
            "model": MODEL_NAME
        }
        try:
            with open(cache_unit_test_cases, "r", encoding="utf-8") as f:
                gen_existing_data = json.load(f)
                if not isinstance(gen_existing_data, list):
                    gen_existing_data = [gen_existing_data]
        except (FileNotFoundError, json.JSONDecodeError):
            gen_existing_data = []
        gen_existing_data.append(gen_data_to_store)
        with open(cache_unit_test_cases, "w", encoding="utf-8") as f:
            json.dump(gen_existing_data, f, ensure_ascii=False, indent=2)
        print(f"Generator tool log saved to {cache_unit_test_cases}")
        print(f"Generator cpp code and commands log saved to {cache_unit_test_cases}")

        # Write the cpp code into file named gen.cpp and commands list into commands_to_execute.txt in out_dir
        if cpp_code:
            with open(os.path.join(out_dir, "gen.cpp"), "w", encoding="utf-8") as f:
                f.write(cpp_code)
        if commands:
            commands_path = os.path.join(out_dir, "commands_to_execute.txt")
            with open(commands_path, "w", encoding="utf-8") as f:
                if isinstance(commands, list):
                    for cmd in commands:
                        f.write(f"{cmd}\n")
                elif isinstance(commands, str):
                    f.write(commands)
        return [{"cpp_code": cpp_code, "commands": commands}]

# Example usage
if __name__ == "__main__":
    try:
        from session_manager import set_current_session_id, generate_session_id
        
        UnitTestGenerator = UnitTestGenerator()

        # Sample algorithm question
        example_question = """
# E. Garden

time limit per test: 2 seconds
memory limit per test: 256 megabytes

Vasya has a very beautiful country garden that can be represented as an $n \\times m$ rectangular field divided into $n \\times m$ squares. One beautiful day Vasya remembered that he needs to pave roads between $k$ important squares that contain buildings. To pave a road, he can cover some squares of his garden with concrete.

For each garden square we know number $a_{ij}$ that represents the number of flowers that grow in the square with coordinates $(i, j)$. When a square is covered with concrete, all flowers that grow in the square die.

Vasya wants to cover some squares with concrete so that the following conditions were fulfilled:

* all $k$ important squares should necessarily be covered with concrete
* from each important square there should be a way to any other important square. The way should go be paved with concrete-covered squares considering that neighboring squares are squares that have a common side
* the total number of dead plants should be minimum

As Vasya has a rather large garden, he asks you to help him.

## Input

The first input line contains three integers $n$, $m$ and $k$ ($1 \\le n, m \\le 100$, $n \\times m \\le 200$, $1 \\le k \\le min(n \\times m, 7)$) -- the garden's sizes and the number of the important squares. Each of the next $n$ lines contains $m$ numbers $a_{ij}$ ($1 \\le a_{ij} \\le 1000$) -- the numbers of flowers in the squares. Next $k$ lines contain coordinates of important squares written as "$x$ $y$" (without quotes) ($1 \\le x \\le n$, $1 \\le y \\le m$). The numbers written on one line are separated by spaces. It is guaranteed that all $k$ important squares have different coordinates.

## Output

In the first line print the single integer -- the minimum number of plants that die during the road construction. Then print $n$ lines each containing $m$ characters -- the garden's plan. In this plan use character "X" (uppercase Latin letter X) to represent a concrete-covered square and use character "." (dot) for a square that isn't covered with concrete. If there are multiple solutions, print any of them.

### Examples

**input**

3 3 2
3 3 2
1 2 3
1 2 3
1 2 3
1 2
3 3

**output**
9
.X.
.X.
.X.

**input**
4 5 4
4 5 4 2 2
2 4 1 2 2
2 4 1 4 5
4 1 2 7 1
1 5
1 1
3 2
4 4

**output**
26
X..XX
.X.X.
..X.X
X..XX
"""

        # 生成并设置会话ID
        session_id = generate_session_id(example_question)
        set_current_session_id(session_id)
        UnitTestGenerator._set_session_id(session_id)
        
        
        # 提取题目信息
        result = UnitTestGenerator.forward(question=example_question, language="cpp", conversation_id=session_id)
        print(result)

    except Exception as e:
        print(f"Error: {e}")