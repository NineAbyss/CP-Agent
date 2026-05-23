from __future__ import annotations

from smolagents import Tool
import subprocess
import datetime
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from agentflow.tools.session_manager import get_current_session_id
import json
import shlex
import time
import openai
from openai import OpenAI
import concurrent.futures
import hashlib

# Model configuration for generator synthesis
MODEL_NAME = "deepseek-chat"
API_BASE = "https://api.deepseek.com/v1"
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")

system_case_prompts = "You are a helpful assistant help user generate test examples for coding tasks."
Generator_Agent_prompt = """You are an expert test case designer for competitive programming contests. Your task is to create a C++ generator program using the `testlib.h` library and a corresponding list of commands to produce a comprehensive set of test cases for the problem described below.\n\n### Workflow and Instructions\n\nYour process should be as follows:\n\n**1. Analyze the Problem and Constraints:**\n* Carefully read the provided problem description.\n* Identify and summarize all input variables and their constraints (e.g., $1 \\le n \\le 10^5$).\n\n**2. Brainstorm Adversarial and Corner Cases:**\n* Anticipate potential mistakes contestants might make.\n* Consider edge cases based on the constraints (e.g., minimum/maximum values).\n* Identify different structural patterns for the input data that could challenge common algorithms. For a graph problem, this might include chains, star graphs, complete graphs, disconnected components, etc. For an array problem, this could be sorted, reversed, or all identical elements.\n\n**3. Design and Implement the Generator Program:**\n* Based on your analysis, define the command-line arguments your generator will accept. This should include parameters for data size (e.g., `-n`, `-m`) and a `-type` parameter to specify the kind of test case to generate (e.g., `-type random`, `-type chain`).\n* Write a complete C++ generator program that implements this design.\n\n**Generator Implementation Rules:**\n* You **MUST** use the `testlib.h` library. Start your program with `#include \"testlib.h\"` and `registerGen(argc, argv, 1);`.\n* You **MUST** use `opt<T>(\"param_name\")` to parse command-line arguments.\n* You **MUST** use `rnd.next(...)` from `testlib.h` for all random number generation. Do not use standard C++ random functions like `rand()` or `<random>`.\n* The generator program **MUST NOT** parse or set a random seed. This is handled externally to ensure reproducibility.\n* The program must generate **exactly one** test case to standard output per execution.\n\n**4. Create the Command List:**\n* Provide a list of approximately 20 distinct commands to execute your generator.\n* This list should be diverse and cover all the categories you brainstormed, including:\n    * Minimum and maximum constraint values.\n    * Randomly generated \"average\" cases for various sizes (small, medium, large).\n    * All special `type`s you implemented.\n    * Challenging and adversarial test cases.\n\n### Problem Statement\n\n```\n{problem_to_solve}\n```\n\n### Required Output Format\n\nYour entire response must be a single block of text containing two distinct parts in the following order:\n\n1.  The complete C++ generator source code, enclosed in a C++ markdown block.\n2.  The list of shell commands, enclosed in a plain markdown block.\n\n**Example Structure:**\n\n```cpp\n// C++ generator code starts here.\n#include \"testlib.h\"\n#include <vector>\n// ... rest of the generator code\n```\n\n```commands\n# Command list starts here.\n./gen -n 1 -m 0 -type random\n./gen -n 100000 -m 99999 -type chain\n# ... rest of the commands\n```\n"""


class Cases_In_Generator(Tool):
    """
    Generates a C++ test generator (using testlib.h) and a command list for an
    algorithmic problem, then compiles and executes those commands to produce
    input files under a cases_in directory.
    """

    name = "cases_in_generator"
    description = (
    "Generate a testlib-based C++ generator and a diverse list of commands for the problem, then compile and run them to produce input test cases in the cases_in/ directory."
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "Algorithm problem description used to generate test generator and commands.",
        },
        "language": {
            "type": "string",
            "description": "Language hint (unused).",
        },
        "conversation_id": {
            "type": "string",
            "description": "Optional conversation ID to scope output directory.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model_id: str = "deepseek-chat",
                 api_base: str = "https://api.deepseek.com/v1",
                 api_key: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('QWEN_API_KEY')
        self._current_session_id = None
        self._output_dir: Optional[str] = None
        self._conversation: List[Dict[str, str]] = []
        print(f"Cases_IN_Generator initialized with model: {self.model_id}")
        print(f"Cases_IN_Generator api_base: {self.api_base}")
        print(f"Cases_IN_Generator api_key: {self.api_key[:20] if self.api_key else 'None'}...")

    def _set_output_dir(self, question: str, conversation_id: str):
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
        
        if global_dir:
            unit_test_data_dir = os.path.join(global_dir, "unit_test_data")
        else:
            from agentflow import config
            work_dir_base = config.WORK_DIR
            if not os.path.isabs(work_dir_base):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                work_dir_base = os.path.join(project_root, work_dir_base)
            unit_test_data_dir = os.path.join(work_dir_base, conversation_id, "unit_test_data")
        
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
        problem_hash = re.sub(r"[^0-9a-f]", "", str(hash(problem_text)))[:8]
        return f"auto_{timestamp}_{problem_hash}"

    # ----------------------------- Conversation helpers ----------------------------- #

    def _append_message(self, role: str, content: str):

        self._conversation.append({"role": role, "content": content})
        work_dir = self._output_dir or "."
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
            if global_dir:
                unit_test_data_dir = os.path.join(global_dir, "unit_test_data")
        except Exception as e:
            print(f"Warning: get_global_work_dir failed: {e}")
        logs_dir = os.path.join(global_dir, "logs")
        log_path = os.path.join(logs_dir, "generator_chat_log.json")

        try:
            os.makedirs(logs_dir, exist_ok=True)
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []
            existing.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "role": role,
                "content": content,
                "model": MODEL_NAME,
            })
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to persist conversation log: {e}")

    # ----------------------------- API integration ----------------------------- #

    def _api_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 60) -> str:
        """
        Call the OpenAI-style chat completions API and return the content string.
        Implements infinite retry with 30-second delay on API errors.
        """

        # Create OpenAI client with instance configuration
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

        # Remove LiteLLM provider prefix (e.g., "openai/") for OpenAI-compatible APIs
        model_id_clean = self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id

        try:
            response = client.chat.completions.create(
                model=model_id_clean,  # Use cleaned model ID without provider prefix
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                stream=False,
            )
            return response.choices[0].message.content
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            print(f"[CasesInGenerator API Error] {repr(e)}")
            print("Sleeping for 30 seconds before retry...")
            print("Consider reducing the number of parallel processes.")
            from time import sleep
            sleep(30)
            return self._api_chat(messages, temperature, timeout)
        except Exception as e:
            print(f"[CasesInGenerator] Fatal error (non-retryable): {repr(e)}")
            raise e

    def _call_llm_and_extract_response(self) -> str:
        """
        Call the LLM API with the full conversation and return the response.
        """
        response_text = self._api_chat(self._conversation)
        print(response_text)
        self._append_message("assistant", response_text)
        return response_text

    def _compile_cpp_file(self, cpp_file: str):
        compile_success = False
        compile_error = ""
        exec_path = ""
        print(f"cpp_file: {cpp_file}")
        if cpp_file is not None:
            try:
                base_name = os.path.splitext(os.path.basename(cpp_file))[0]
                base_dir = os.path.dirname(os.path.abspath(__file__))
                include_dir = os.path.join(base_dir, "utils", "testlib")
                exec_dir = self._output_dir
                exec_path = os.path.join(exec_dir, base_name)
                compile_cmd = [
                    "g++", "-O2", "-std=c++17",
                    f"-I{include_dir}",
                    "-o", exec_path,
                    cpp_file,
                ]
                print(f"compile_cmd: {compile_cmd}")
                result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=exec_dir, timeout=15)
                if result.returncode == 0:
                    compile_success = True
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
            s = re.sub(r'[^\w\-.]', '_', s)
            return s

        testcases_dir = os.path.join(base_dir, "cases_in")
        os.makedirs(testcases_dir, exist_ok=True)

        def _run_single_command(command_line: str, timeout_seconds: int = 10, idx: int = 0):
            try:
                parts = shlex.split(command_line)
                if not parts:
                    prog = "unknown"
                    args = []
                else:
                    prog = os.path.basename(parts[0])
                    args = parts[1:]
                arg_part = "_".join([_sanitize_filename(a) for a in args])
                cmd_hash = hashlib.md5(command_line.encode()).hexdigest()[:8]
                if arg_part:
                    base_name = f"{prog}__{arg_part}__{cmd_hash}__{idx}"
                else:
                    base_name = f"{prog}__{cmd_hash}__{idx}"
                if len(base_name) > 128:
                    base_name = base_name[:128]
                stdout_file = os.path.join(testcases_dir, f"{base_name}_stdout.txt")
                stderr_file = os.path.join(testcases_dir, f"{base_name}.stderr.txt")
            except Exception:
                stdout_file = os.path.join(testcases_dir, f"unknown_stdout_{idx}.txt")
                stderr_file = os.path.join(testcases_dir, f"unknown.stderr_{idx}.txt")

            try:
                print("--- Starting Subprocess ---")
                print(f"Command: {command_line}")
                print(f"Timeout: {timeout_seconds} seconds")
                print("--------------------------")
                
                import signal
                import resource
                def limit_resources():
                    mem_limit = 8 * 1024 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                    resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds + 5, timeout_seconds + 5))
                
                proc = subprocess.Popen(
                    command_line,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=base_dir,
                    start_new_session=True,  
                    preexec_fn=limit_resources,  
                )
                try:
                    stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
                    with open(stdout_file, "w", encoding="utf-8") as f:
                        f.write(stdout_data or "")
                    with open(stderr_file, "w", encoding="utf-8") as f:
                        f.write(stderr_data or "")
                    return command_line, proc.returncode, stdout_data, stderr_data, stdout_file, stderr_file
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass 
                    proc.wait()  
                    timeout_msg = f"Command timed out after {timeout_seconds}s"
                    with open(stdout_file, "w", encoding="utf-8") as f:
                        f.write("")
                    with open(stderr_file, "w", encoding="utf-8") as f:
                        f.write(timeout_msg)
                    return command_line, 124, "", timeout_msg, stdout_file, stderr_file
            except Exception as e:
                with open(stderr_file, "w", encoding="utf-8") as f:
                    f.write(str(e))
                return command_line, 1, "", str(e), stdout_file, stderr_file

        try:
            with open(commands_file, "r", encoding="utf-8") as f:
                raw_lines = [ln.strip() for ln in f.readlines()]
            commands = [ln for ln in raw_lines if ln and not ln.startswith('#')]
            if not commands:
                return True, ""

            max_workers = min(8, max(1, (os.cpu_count() or 4) * 2), len(commands))
            failures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_cmd = {executor.submit(_run_single_command, cmd, 15, idx): (cmd, idx)
                                 for idx, cmd in enumerate(commands)}
                for fut in concurrent.futures.as_completed(future_to_cmd):
                    cmd, idx = future_to_cmd[fut]
                    try:
                        command_line, code, out, err, stdout_file, stderr_file = fut.result()
                        if out:
                            print(f"Command '{command_line}' succeed")
                        if code != 0:
                            failures.append((command_line, code, err, stdout_file, stderr_file))
                    except Exception as e:
                        failures.append((cmd, 1, str(e), None, None))

            if failures:
                msgs = []
                for cmd, code, err, stdout_file, stderr_file in failures:
                    msgs.append(f"Command failed (exit {code}): {cmd}\n{err}\nstdout_file: {stdout_file}\nstderr_file: {stderr_file}")
                return False, "\n".join(msgs)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _run_with_input(self, exec_path: str, input_path: str, timeout_seconds: int = 60) -> Tuple[int, str, str]:

        try:
            with open(input_path, "rb") as fin:
                completed = subprocess.run(
                    [exec_path],
                    stdin=fin,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=self._output_dir,
                )
            return completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or "", e.stderr or f"Command timed out after {timeout_seconds}s"
        except Exception as e:
            return 1, "", str(e)

    def _save_cpp_and_commands_from_test_cases(self, test_cases, cpp_filename="gen.cpp", commands_filename="commands_to_execute.txt", out_dir: Optional[str] = None):
        if not test_cases:
            return None, None
        last = None
        for item in reversed(test_cases):
            if isinstance(item, dict) and "cpp_code" in item and "commands" in item:
                last = item
                break
        if last is None:
            return None, None
        cpp_code = last["cpp_code"]
        commands = last["commands"]
        if out_dir is None:
            out_dir = self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        cpp_path = os.path.join(out_dir, cpp_filename)
        commands_path = os.path.join(out_dir, commands_filename)
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(cpp_code)
        with open(commands_path, "w", encoding="utf-8") as f:
            for cmd in commands:
                f.write(cmd.strip() + "\n")
        abs_cpp_filename = os.path.abspath(cpp_path)
        abs_commands_path = os.path.abspath(commands_path)
        return abs_cpp_filename, abs_commands_path

    def _generate_test_cases_stub(self, question: str, cache_unit_test_cases: str) -> Dict[str, Any]:
        def _parse_cpp_code(language: str, raw_response: str) -> str:
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
            try:
                if not isinstance(response_string, str):
                    return []
                match = re.search(r"```commands\n(.*?)\n```", response_string, re.DOTALL)
                if not match:
                    return []
                raw_commands = match.group(1)
                command_list = []
                for line in raw_commands.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        command_list.append(stripped_line)
                return command_list
            except Exception:
                return []

        # Call LLM API with the existing conversation and parse
        api_response = self._call_llm_and_extract_response()
        
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
            "model": self.model_id,
        }
        try:
            cache_file = cache_unit_test_cases
            with open(cache_file, "r", encoding="utf-8") as f:
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

        return {"cpp_code": cpp_code, "commands": commands}

    def _write_cpp_and_commands(self, cpp_code: str, commands: List[str], cpp_filename: str = "gen.cpp", commands_filename: str = "commands_to_execute.txt", out_dir: Optional[str] = None):
        if out_dir is None:
            out_dir = self._output_dir
        os.makedirs(out_dir, exist_ok=True)
        cpp_path = os.path.join(out_dir, cpp_filename)
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(cpp_code or "")
        commands_path = os.path.join(out_dir, commands_filename)
        with open(commands_path, "w", encoding="utf-8") as f:
            if isinstance(commands, list):
                for cmd in commands:
                    f.write(f"{cmd}\n")
            elif isinstance(commands, str):
                f.write(commands)
        return os.path.abspath(cpp_path), os.path.abspath(commands_path)

    def forward(self, question: str, language: str, conversation_id: Optional[str] = None) -> str:
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
        except Exception as e:
            print(f"Error occurred while setting up conversation_id: {e}")
            return f"Error: {e}"

        out_dir = self._set_output_dir(question, conversation_id)
        cache_unit_test_cases = os.path.join(out_dir, "generator_and_commands.json")
        # Save the original question to disk as original_question.txt
        original_question_path = os.path.join(out_dir, "original_question.txt")
        with open(original_question_path, "w", encoding="utf-8") as f:
            f.write(question)
        print(f"Original question saved to {original_question_path}")

        # Initialize conversation for this session
        self._conversation = []
        self._append_message("system", system_case_prompts)
        self._append_message("user", Generator_Agent_prompt.format(problem_to_solve=question))

        # Run validator tool
        try:
            from agentflow.tools.validator import ValidatorTool
        except Exception as e:
            print(f"Error importing ValidatorTool: {e}")
            print("Make sure 'agentflow/tools/validator.py' exists and is importable.")

        tool = ValidatorTool()
        result = tool.forward(question=question, language="cpp", conversation_id=conversation_id)

        # If successful, extract and display key paths
        validate_exec = None
        if isinstance(result, str) and result.startswith("OK: generator and commands created."):
            m = re.search(r"dir=([^,]+),\s*validate_cpp=([^,]+),\s*executable=(\S+)", result)
            if m:
                out_dir, cpp_path, exe_path = m.groups()
                print("\nArtifacts:")
                print(f"- dir: {out_dir}")
                print(f"- validate_cpp: {cpp_path}")
                print(f"- executable: {exe_path}")
                validate_exec = exe_path
            else:
                print("Could not parse artifact paths from result string.")

        attempt = 0
        max_attempts = 5
        cpp_file = None
        commands_file = None

        while attempt < max_attempts:
            print(f"==================== Generating test cases input... attempt {attempt + 1} ====================")

            # Call LLM and parse response
            gen_result = self._generate_test_cases_stub(question, cache_unit_test_cases)
            cpp_code = gen_result.get("cpp_code")
            commands = gen_result.get("commands")

            # Validate parsing result
            if (not isinstance(cpp_code, str)) or ("#include \"testlib.h\"" not in (cpp_code or "")) or (not commands):
                error_msg = (
                    "The previous response did not include a valid C++ generator inside a single ```cpp``` block "
                    "and a ```commands``` block with one command per line.\n"
                    "Requirements reminder: the generator MUST use testlib.h (registerGen, opt<T>(), rnd.next()), "
                    "and generate exactly one test case per run."
                )
                self._append_message("user", error_msg)
                attempt += 1
                continue

            # Persist code and commands
            cpp_file, commands_file = self._write_cpp_and_commands(cpp_code, commands, out_dir=self._output_dir)

            # Try to compile
            compile_success, compile_error, _ = self._compile_cpp_file(cpp_file)
            if not compile_success:
                print(f"Compile error: {compile_error}. Retrying with error feedback...")
                compile_feedback = (
                    "The previous C++ generator program failed to compile with g++ -O2 -std=c++17.\n"
                    f"Compiler errors:\n{compile_error}\n\n"
                    "Please provide a corrected COMPLETE generator in a single ```cpp``` block (must use testlib.h) "
                    "and a corresponding ```commands``` list."
                )
                self._append_message("user", compile_feedback)
                attempt += 1
                continue

            # Try to run
            run_success, run_error = self._run_program(commands_file, self._output_dir)
            if not run_success:
                print(f"Run error: {run_error}. Retrying with error feedback...")
                run_feedback = (
                    "The compiled generator failed during execution when running the provided commands.\n"
                    f"Details:\n{run_error}\n\n"
                    "Please fix the generator program and/or the command list and reply with a COMPLETE ```cpp``` block "
                    "and a ```commands``` block."
                )
                self._append_message("user", run_feedback)
                attempt += 1
                continue

            print("Input generation run success.")

            # Validate the test cases in the cases_in directory
            if validate_exec:
                testcases_dir = os.path.join(self._output_dir, "cases_in")
                failures: List[str] = []
                validated_count = 0
                try:
                    for file_name in os.listdir(testcases_dir):
                        if file_name.endswith("_stdout.txt"):
                            input_path = os.path.join(testcases_dir, file_name)
                            code, out, err = self._run_with_input(validate_exec, input_path)
                            if code != 0:
                                failures.append(f"- {file_name}: exit={code}, stderr=\n{err}")
                            else:
                                validated_count += 1
                                print(f"Test case {file_name} validated successfully.")
                except FileNotFoundError:
                    failures.append("No cases_in directory or stdout files were found.")

                if failures or validated_count == 0:
                    error_report = (
                        "The generated stdin did not pass the validator. Please correct the generator and/or commands.\n"
                        f"Validation details (failed {len(failures)} cases):\n" + "\n".join(failures)
                    )
                    self._append_message("user", error_report)
                    attempt += 1
                    continue

            # If no validator provided or all validations passed
            break
        else:
            return "Failed to generate and run generator after multiple attempts."

        gen_cpp_path = os.path.join(self._output_dir, "gen.cpp")
        cmds_path = os.path.join(self._output_dir, "commands_to_execute.txt")
        return f"OK: generator and commands created. dir={self._output_dir}, gen={gen_cpp_path}, cmds={cmds_path}"

