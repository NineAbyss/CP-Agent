from __future__ import annotations

import os
import re
import json
import datetime
import subprocess
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed  # NEW

from smolagents import Tool
import openai
from openai import OpenAI
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from agentflow.tools.cpp_validation import create_cpp_validation_tool
except Exception:
    # Fallback for relative import in local execution environments
    from cpp_validation import create_cpp_validation_tool


class Bf_Solution_Generator(Tool):
    """
    Generate a brute-force C++ solution via LLM, validate it with local cases, and compile it into an executable.

    Behavior:
    - Reads the problem statement from original_question.txt under the session's unit_test_data directory
    - Calls an LLM to produce C++17 code for a brute-force solution
    - Validates the code via cpp_validation tool in 'detailed' mode using provided test cases
      - If validation fails, feeds the validation report and previous code back to the LLM to regenerate
    - Writes bf_solution.cpp in the same directory
    - Compiles it with g++ into an executable named 'bf_solution' in that directory

    Output (string): On success, returns the absolute path to the compiled executable.
                      On failure, returns a descriptive error message.
    """

    name = "bf_solution_generator"
    description = "Use LLM to generate a C++ brute-force solution, validate it, and compile it into an executable in the current session output directory."
    inputs = {
        "sample_N": {
            "type": "integer",
            "description": "Number of brute-force solution samples to generate in parallel (default 1).",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, model_id: str = "deepseek-chat", 
                 api_base: str = "https://api.deepseek.com/v1",
                 api_key: str = None,
                 temperature: float = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._current_session_id: Optional[str] = None
        self._output_dir: Optional[str] = None
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv('QWEN_API_KEY')
        
        if temperature is not None:
            self.temperature = temperature
        else:
            try:
                from agentflow import config
                self.temperature = getattr(config, 'BF_SOLUTION_TEMPERATURE', 0.6)
            except Exception:
                self.temperature = 0.6
        
        print(f"[bf_solution_generator] Initialized with model: {self.model_id}, temperature: {self.temperature}")

    # --- Session/paths utilities ---
    def _set_output_dir(self, conversation_id: str):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
            print("global_dir is None")
        if global_dir:
            unit_test_data_dir = os.path.join(global_dir, "unit_test_data")
        else:
            from agentflow import config
            work_dir_base = config.WORK_DIR
            if not os.path.isabs(work_dir_base):
                work_dir_base = os.path.join(project_root, work_dir_base)
            unit_test_data_dir = os.path.join(work_dir_base, conversation_id, "unit_test_data")
        os.makedirs(unit_test_data_dir, exist_ok=True)
        self._output_dir = unit_test_data_dir
        print(f"[bf_solution_generator] Output directory: {unit_test_data_dir}")
        return unit_test_data_dir

    def _set_session_id(self, session_id: str):
        self._current_session_id = session_id

    def _get_session_id(self) -> Optional[str]:
        return self._current_session_id

    # --- Validation via cpp_validation tool (detailed mode) ---
    def _validate_cpp_code(self, code: str) -> tuple[bool, str]:
        """
        Validate C++ code using CppValidationTool in 'basic' mode.

        Returns: (passed, report)
        - passed: True if all tests passed, False otherwise
        - report: validation summary/details
        """
        try:
            from agentflow.config import LONG_TERM_EXP_ENABLED
            # Handle None case (default) as False, or respect explicit False
            enable_analysis = LONG_TERM_EXP_ENABLED if LONG_TERM_EXP_ENABLED is not None else False
            tool = create_cpp_validation_tool(
                enable_error_analysis=enable_analysis,
                model_id=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key
            )
            report = tool.forward(code=code)
            # Heuristic: consider pass if summary starts with or contains ALL TESTS PASSED
            normalized = (report or "").strip()
            passed = "Test PASSED" in normalized or normalized.startswith("✅ ALL TESTS PASSED")
            return passed, report
        except Exception as e:
            err_msg = f"Validation tool error: {e}"
            print(f"[bf_solution_generator] {err_msg}")
            return False, err_msg

    # --- Compile ---
    def _compile_cpp_file(self, cpp_file: str) -> Tuple[bool, str, str]:
        compile_success = False
        compile_error = ""
        exec_path = ""
        print(f"[bf_solution_generator] Compiling cpp_file: {cpp_file}")
        if cpp_file:
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
                print(f"[bf_solution_generator] compile_cmd: {compile_cmd}")
                result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=exec_dir)
                if result.returncode == 0:
                    compile_success = True
                else:
                    compile_error = result.stderr
                    exec_path = ""
            except Exception as e:
                compile_error = str(e)
                exec_path = ""
        else:
            compile_error = "cpp_file is empty"

        print(f"[bf_solution_generator] compile_success: {compile_success}")
        if not compile_success:
            print(f"[bf_solution_generator] compile_error: {compile_error}")
        return compile_success, compile_error, exec_path

    # --- NEW: Single sample generation (LLM -> validate -> compile) ---
    def single_bf_generation(self, index: int, question: str) -> str:
        """
        Generate, validate and compile one brute-force C++ solution.
        Returns: exec_path on success, or an error string on failure.
        """
        if not self._output_dir:
            return "Error: _output_dir is not set."
        if not self.api_key:
            return "Error: API key is not set."

        # Helpers
        def extract_code_from_response(response_text: str) -> str:
            try:
                if not isinstance(response_text, str):
                    return "We can not extract the code in the output. (raw_response is not a string)"
                pattern = r"```cpp\n(.*?)```"
                matches = re.findall(pattern, response_text, re.DOTALL)
                if matches:
                    return matches[-1].strip()
                pattern_fallback = r"```\n(.*?)```"
                matches_fallback = re.findall(pattern_fallback, response_text, re.DOTALL)
                if matches_fallback:
                    return matches_fallback[-1].strip()
                return "We can not extract the code in the output."
            except Exception as e:
                return f"We can not extract the code in the output. (Exception: {e})"

        system_prompt = (
            "You are an elite competitive-programming assistant.\n"
            "Produce only valid C++17 code.\n"
            "Use standard I/O (cin/cout).\n"
            "Include every helper necessary for a brute-force solution.\n"
            "Return only the C++ source code, wrapped in a single fenced code block: ```cpp ... ```."
        )
        user_prompt_template = (
            "You are an expert competitive programming assistant.\n\n"
            "Your task is to write a correct and brute-force C++ solution for the following problem. The program must:\n"
            "- Read input from standard input and write output to standard output, strictly following the problem's input/output format.\n"
            "- Implement a straightforward, brute-force (but correct) algorithm, prioritizing clarity and correctness over efficiency.\n"
            "- Avoid any optimizations or advanced algorithms; use only simple, exhaustive approaches.\n"
            "- Ensure the code is self-contained and compiles with GNU G++23 14.2 (64bit, msys2).\n\n"
            "---\n\n{Question}\n\n---\n\n"
            "Language and Compiler\n"
            "- Language: cpp\n"
            "- Compiler: GNU G++23 14.2 (64bit, msys2)\n\n"
            "Output:\n"
            "Provide only the complete C++ source code, enclosed in a single code block using the syntax ```cpp ... ```."
        )

        cpp_path = os.path.join(self._output_dir, f"bf_solution_{index}.cpp")
        exec_expected_path = os.path.join(self._output_dir, f"bf_solution_{index}")  # expected executable path
        logs_path = os.path.join(self._output_dir, "logs", f"bf_chat_log_{index}.json")
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)

        formatted_user_prompt = user_prompt_template.format(Question=question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            trace_available = True
        except ImportError:
            tracer = None
            trace_available = False

        # Helper function for API call with infinite retry
        def call_api_with_retry(messages_arg):

            if tracer:
                with tracer.start_as_current_span("bf_solution_generator.llm_call") as span:
                    span.set_attribute("tool.name", "bf_solution_generator")
                    span.set_attribute("llm.model", self.model_id)
                    span.set_attribute("llm.temperature", self.temperature)
                    span.set_attribute("llm.sample_index", index)
                    span.set_attribute("llm.input_messages_count", len(messages_arg))
                    try:
                        import json
                        messages_json = json.dumps(messages_arg, ensure_ascii=False)
                        span.set_attribute("llm.input.messages", messages_json)
                    except Exception as e:
                        span.set_attribute("llm.input.messages", str(messages_arg))
                    
                    try:
                        print(f"[bf_solution_generator] (#{index}) Generating brute-force solution using model: {self.model_id}, temperature: {self.temperature}...")
                        completion = client.chat.completions.create(
                            model=self.model_id,
                            messages=messages_arg,
                            temperature=self.temperature,
                        )
                        content = completion.choices[0].message.content
                        
                        if content is None:
                            error_msg = "API returned None content. This may indicate the model refused to respond or encountered an error."
                            span.set_attribute("llm.output.error", error_msg)
                            from opentelemetry import trace
                            span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                            span.record_exception(ValueError(error_msg))
                            print(f"[bf_solution_generator] API Error: {error_msg}")
                            print("Sleeping for 30 seconds before retry...")
                            from time import sleep
                            sleep(30)
                            return call_api_with_retry(messages_arg)
                        
                        span.set_attribute("llm.output.length", len(content) if content else 0)
                        span.set_attribute("llm.output.content", content if content else "")
                        from opentelemetry import trace
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return content
                    except (
                        openai.APIError,
                        openai.RateLimitError,
                        openai.InternalServerError,
                        openai.OpenAIError,
                        openai.APIStatusError,
                        openai.APITimeoutError,
                        openai.APIConnectionError,
                    ) as e:
                        from opentelemetry import trace
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        print(f"[bf_solution_generator] API Error: {repr(e)}")
                        print("Sleeping for 30 seconds before retry...")
                        from time import sleep
                        sleep(30)
                        return call_api_with_retry(messages_arg)
                    except Exception as e:
                        from opentelemetry import trace
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        print(f"[bf_solution_generator] Fatal error (non-retryable): {repr(e)}")
                        raise e
            else:
                try:
                    print(f"[bf_solution_generator] (#{index}) Generating brute-force solution using model: {self.model_id}, temperature: {self.temperature}...")
                    completion = client.chat.completions.create(
                        model=self.model_id,
                        messages=messages_arg,
                        temperature=self.temperature,
                    )
                    content = completion.choices[0].message.content
                    if content is None:
                        error_msg = "API returned None content. This may indicate the model refused to respond or encountered an error."
                        print(f"[bf_solution_generator] API Error: {error_msg}")
                        print("Sleeping for 30 seconds before retry...")
                        from time import sleep
                        sleep(30)
                        return call_api_with_retry(messages_arg)
                    return content
                except (
                    openai.APIError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                    openai.OpenAIError,
                    openai.APIStatusError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                ) as e:
                    print(f"[bf_solution_generator] API Error: {repr(e)}")
                    print("Sleeping for 30 seconds before retry...")
                    from time import sleep
                    sleep(30)
                    return call_api_with_retry(messages_arg)
                except Exception as e:
                    print(f"[bf_solution_generator] Fatal error (non-retryable): {repr(e)}")
                    raise e

        attempts = 0
        last_error = ""
        while attempts < 10:
            print("=" * 40 + f"index:{index} Attempt {attempts + 1} (index={index}) " + "=" * 40)
            # Call LLM with infinite retry
            try:
                response_content = call_api_with_retry(messages)
            except Exception as e:
                attempts += 1
                last_error = f"LLM API fatal error: {e}"
                continue

            # Log request/response (best-effort)
            try:
                log_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "request": {"messages": messages, "model": self.model_id, "temperature": self.temperature},
                    "response": {"content": response_content},
                }
                if os.path.exists(logs_path):
                    with open(logs_path, "r", encoding="utf-8") as f:
                        history = json.load(f)
                else:
                    history = []
                history.append(log_data)
                with open(logs_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[bf_solution_generator] Warning: Could not write to log file (#{index}): {e}")

            if response_content is None:
                attempts += 1
                last_error = "LLM API returned None content. Retrying..."
                print(f"[bf_solution_generator] (#{index}) Warning: API returned None content, skipping this attempt.")
                continue
            
            # Extract code
            code = extract_code_from_response(response_content)
            try:
                messages.append({"role": "assistant", "content": response_content})
            except Exception:
                messages.append({"role": "assistant", "content": "<unserializable assistant content>"})

            if not code or code.startswith("We can not extract the code"):
                attempts += 1
                last_error = "Failed to extract C++ code from LLM response."
                messages.append({
                    "role": "user",
                    "content": (
                        "Please return ONLY valid C++17 source code inside a single fenced code block like ```cpp ... ```; "
                        "no prose or explanations."
                    ),
                })
                continue

            # Write file
            try:
                with open(cpp_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print(f"[bf_solution_generator] (#{index}) Code written to {cpp_path}")
            except Exception as e:
                attempts += 1
                last_error = f"Failed to write C++ file: {e}"
                continue

            # Validate
            print(f"[bf_solution_generator] (#{index}) Validating generated code...")
            passed, report = self._validate_cpp_code(code)
            if not passed:
                print(f"[bf_solution_generator] (#{index}) Validation failed. Feeding back to LLM for correction.")
                fb_note = (
                    "The previous attempt failed on the following tests. "
                    "Please fix the code to satisfy ALL tests and return ONLY the corrected C++17 source code in one code block.\n\n"
                )
                messages.append({"role": "user", "content": fb_note + (report or "")})
                attempts += 1
                last_error = f"Validation failed before compile. Report: {(report or '')[:3000]}"
                continue
            print("report:", (report or "")[:3000])
            print(f"[bf_solution_generator] (#{index}) Validation passed the basic test.")

            # Compile
            ok, err, exec_path = self._compile_cpp_file(cpp_path)
            if ok:
                return exec_path

            attempts += 1
            last_error = f"Compile failed: {err}"
            compile_feedback = (
                "The code failed to compile with g++17. Here is the compiler error output:\n\n{err}\n\n"
                "Please correct and return ONLY the fixed C++17 code in one fenced code block."
            ).format(err=(err or ""))
            messages.append({"role": "user", "content": compile_feedback})

        # Cleanup after 10 failed attempts: remove generated cpp and exec files
        try:
            if os.path.exists(cpp_path):
                os.remove(cpp_path)
            if os.path.exists(exec_expected_path):
                os.remove(exec_expected_path)
            print(f"[bf_solution_generator] (#{index}) Cleanup done for failed attempts.")
        except Exception as e:
            print(f"[bf_solution_generator] (#{index}) Cleanup warning: {e}")

        return (
            f"(index={index}) Failed to generate and compile brute-force solution after multiple attempts. "
            f"Last error: {last_error}"
        )

    def forward(self, sample_N: int = 1) -> str:
        # Acquire session and output dir
        try:
            from agentflow.tools.session_manager import get_current_session_id
            conversation_id = get_current_session_id()
            if not conversation_id:
                return "Error: No active conversation session. Please use run_coding_agent_with_session()."
            self._set_session_id(conversation_id)
            self._set_output_dir(conversation_id)
        except Exception as e:
            return f"Error: Cannot access session manager: {e}"

        # Read original question
        try:
            original_question_path = os.path.join(self._output_dir, "original_question.txt")
            if not os.path.exists(original_question_path):
                return f"Error: original_question.txt not found at {original_question_path}"
            with open(original_question_path, "r", encoding="utf-8") as f:
                question = f.read()
        except Exception as e:
            return f"Error: Cannot read original question: {e}"

        # Validate inputs
        if not isinstance(sample_N, int) or sample_N <= 0:
            sample_N = 1
        if not self.api_key:
            return "Error: API key is not set."

        # Parallel generation
        results = []
        errors = []
        max_workers = min(sample_N, max(1, (os.cpu_count() or 4)))
        print(f"[bf_solution_generator] Spawning {sample_N} parallel generations with {max_workers} workers...")

        try:
            from opentelemetry import context
            current_context = context.get_current()
        except ImportError:
            current_context = None


        from agentflow.tools.session_manager import set_current_session_id, set_global_work_dir, get_global_work_dir
        parent_session_id = conversation_id  # Already obtained at the beginning of forward()
        parent_work_dir = get_global_work_dir()

        def run_with_context(idx, question):
            set_current_session_id(parent_session_id)
            if parent_work_dir:
                set_global_work_dir(parent_work_dir)
            
            if current_context:
                token = context.attach(current_context)
                try:
                    return self.single_bf_generation(idx, question)
                finally:
                    context.detach(token)
            else:
                return self.single_bf_generation(idx, question)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_with_context, idx, question): idx for idx in range(sample_N)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                    # Identify success vs error by path existence heuristic
                    if isinstance(res, str) and os.path.isabs(res) and os.path.exists(res):
                        results.append({"index": idx, "exec_path": res,
                                        "cpp_path": os.path.join(self._output_dir, f"bf_solution_{idx}.cpp")})
                    else:
                        errors.append({"index": idx, "error": res})
                except Exception as e:
                    errors.append({"index": idx, "error": f"Unexpected exception: {e}"})

        summary = {
            "success": len(results) > 0 and len(errors) == 0,
            "generated": results,
            "failed": errors,
        }
        try:
            return json.dumps(summary, ensure_ascii=False, indent=2)
        except Exception:
            # Fallback plain text if JSON fails
            lines = []
            for r in results:
                lines.append(f"OK index={r['index']} exec={r['exec_path']} cpp={r['cpp_path']}")
            for e in errors:
                lines.append(f"ERR index={e['index']} {e['error']}")
            return "\n".join(lines)


if __name__ == "__main__":
    # Optional quick manual run
    from agentflow.tools.session_manager import set_current_session_id

    # Create a synthetic session id if none
    conversation_id = "session_1760099054725_29b98407"
    set_current_session_id(conversation_id)

    from agentflow.tools.session_manager import set_global_work_dir
    from agentflow import config
    work_dir_base = config.WORK_DIR
    if not os.path.isabs(work_dir_base):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        work_dir_base = os.path.join(project_root, work_dir_base)
    global_work_dir = os.path.join(work_dir_base, conversation_id)
    os.makedirs(global_work_dir, exist_ok=True)
    set_global_work_dir(global_work_dir)
    print(f"Global work dir set to: {global_work_dir}")

    gen = Bf_Solution_Generator()
    # Example: generate 3 parallel brute-force solutions
    msg = gen.forward(sample_N=8)
    print(msg)
