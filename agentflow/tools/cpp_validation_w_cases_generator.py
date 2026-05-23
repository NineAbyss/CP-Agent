from smolagents import Tool
import requests
import base64
import os
import re
from typing import Optional, List, Dict, Tuple
import concurrent.futures
import json
# import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agentflow.tools.cases_out_calculator import Cases_Out_Calculator
from agentflow.tools.session_manager import get_spj_code


class CppValidationTool(Tool):
    """
    A tool for validating C++ code against local test cases.
    Compiles and runs C++ code with test inputs, checks if outputs match expected results.
    Returns detailed I/O information for failed test cases.
    """
    
    name = "cpp_validation"
    description = """
This tool validates your C++ code by compiling and running it against local test cases.
It automatically injects test inputs and compares the actual output with the expected output.
For failed tests, it returns the input data, the expected output, and the actual output.
The output is limited to 500 words to prevent excessive verbosity.
This tool is used to verify whether the code you intend to submit is correct, and it does not require you to provide input.
Before submitting the final code, you must use this tool at least once. 
    """
    
    inputs = {
        "code": {
            "type": "string",
            "description": "The C++ code to validate against test cases."
        }
    }
    output_type = "string"

    def __init__(
        self,
        sandbox_url,
        debug: bool = False,
        default_run_timeout: int = 30,
        enable_error_analysis: bool = True,
        model_id: str = None,
        api_base: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize the C++ validation tool
        
        Args:
            sandbox_url: The URL address of the sandbox service
            debug: Enable debug output for troubleshooting
            default_run_timeout: Default timeout for code execution in seconds
            enable_error_analysis: Whether to enable automatic error analysis
            model_id: LLM model ID for error analysis
            api_base: API endpoint for error analysis
            api_key: API key for error analysis
        """
        super().__init__(**kwargs)
        self.sandbox_url = sandbox_url
        self.debug = debug
        self.default_run_timeout = default_run_timeout
        self.enable_error_analysis = enable_error_analysis
        print(f"[DEBUG] CppValidationTool initialized with enable_error_analysis={self.enable_error_analysis}")
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        self._error_summarizer = None
        self.last_run_stats = {}
        self._generated_cases_triggered_sessions = set()

    def _get_error_summarizer(self):
        if self._error_summarizer is None and self.enable_error_analysis:
            try:
                from agentflow.tools.error_summarizer import create_error_summarizer_tool
                self._error_summarizer = create_error_summarizer_tool(
                    model_id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key
                )
            except Exception as e:
                self.enable_error_analysis = False
        return self._error_summarizer

    def _post_to_sandbox_with_retry(self, payload: dict, timeout: int = None):

        # HTTP timeout should be greater than run_timeout, leaving enough margin
        if timeout is None:
            run_timeout = payload.get("run_timeout", self.default_run_timeout)
            timeout = run_timeout + 30

        try:
            response = requests.post(self.sandbox_url, json=payload, timeout=timeout)
            return response
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            print(f"[CppValidation Sandbox Error] {repr(e)}")
            print("Sleeping for 30 seconds before retry...")
            from time import sleep
            sleep(30)
            return self._post_to_sandbox_with_retry(payload, timeout)
        except Exception as e:
            print(f"[CppValidation Sandbox] Fatal error (non-retryable): {repr(e)}")
            raise e
    
    def _analyze_error(self, error_message: str, error_type: str, code: str):
        if not self.enable_error_analysis:
            return ""
        
        try:
            summarizer = self._get_error_summarizer()
            if summarizer:
                print("\n" + "=" * 60)
                
                code_to_analyze = code if error_type == 'validation' else code[:500]
                
                summarizer.forward(
                    error_message=error_message,
                    error_type=error_type,
                    code_snippet=code_to_analyze
                )
                print("=" * 60 + "\n")
        except Exception as e:
            print(f"{e}")
        
        return ""

    def _inject_freopen(self, code: str) -> str:
        """Inject freopen statement at the beginning of main function to read input file"""
        # Pattern to match int main() function start
        main_pattern = r'(int\s+main\s*\([^)]*\)\s*\{)'
        
        # Check if freopen already exists
        if 'freopen' in code:
            return code
            
        # Injection code
        freopen_code = '\n    freopen("input", "r", stdin);'
        
        # Replace the first matching main function
        modified_code = re.sub(main_pattern, r'\1' + freopen_code, code, count=1)
        
        return modified_code
    
    def _status_file_path(self, conversation_id: str) -> str:
        """Decide where to read/write best_status.json"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
        if global_dir:
            return os.path.join(global_dir, "best_status.json")
        # fallback: keep session separated
        return os.path.join(project_root, "tools", "cpp", "best_status", conversation_id, "best_status.json")

    # NEW: locate final_cases.json and the target base dir for generated test files
    def _final_cases_locations(self, conversation_id: str) -> Tuple[Optional[str], str]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None

        if global_dir:
            final_cases_path = os.path.join(global_dir, "unit_test_data", "final_cases.json")
            final_test_root = os.path.join(global_dir, "final_test_cases")
        else:
            # fallback to project tools/cpp tree
            final_cases_path = os.path.join(
                project_root, "tools", "cpp", "unit_test_data", conversation_id, "final_cases.json"
            )
            final_test_root = os.path.join(
                project_root, "tools", "cpp", "final_test_cases", conversation_id
            )
        return final_cases_path, final_test_root

    # NEW: read basic_cases & generated_cases contents from final_cases.json
    def _read_final_cases(self, conversation_id: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        final_cases_path, _ = self._final_cases_locations(conversation_id)
        basic_cases: List[Dict[str, str]] = []
        gen_cases: List[Dict[str, str]] = []

        if not final_cases_path or not os.path.exists(final_cases_path):
            if self.debug:
                print(f"Debug: final_cases.json not found at {final_cases_path}")
            return basic_cases, gen_cases

        try:
            with open(final_cases_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if self.debug:
                print(f"Debug: Failed to load final_cases.json: {e}")
            return basic_cases, gen_cases

        def normalize_items(items: List[Dict]) -> List[Dict[str, str]]:
            norm: List[Dict[str, str]] = []
            for item in items or []:
                inp = item.get("input")
                out = item.get("output")
                # allow input_path / output_path fallback
                if not inp and item.get("input_path") and os.path.exists(item["input_path"]):
                    try:
                        with open(item["input_path"], "r", encoding="utf-8") as fin:
                            inp = fin.read()
                    except Exception:
                        inp = ""  # keep going
                if not out and item.get("output_path") and os.path.exists(item["output_path"]):
                    try:
                        with open(item["output_path"], "r", encoding="utf-8") as fout:
                            out = fout.read()
                    except Exception:
                        out = ""
                # Only accept when at least strings exist
                if isinstance(inp, str) and isinstance(out, str):
                    norm.append({"input": inp, "output": out})
            return norm

        # Preferred shape: final_cases.json provides basic_cases & generated_cases arrays
        if "basic_cases" in data or "generated_cases" in data:
            basic_cases = normalize_items(data.get("basic_cases", []))
            gen_cases = normalize_items(data.get("generated_cases", []))
        # Fallback: some older generators store everything in "fina_lcases"
        elif "final_cases" in data and isinstance(data["final_cases"], list):
            # Treat them as basic_cases by default
            items = []
            for it in data["final_cases"]:
                inp = it.get("input")
                if inp is None and it.get("input_path") and os.path.exists(it["input_path"]):
                    try:
                        with open(it["input_path"], "r", encoding="utf-8") as fin:
                            inp = fin.read()
                    except Exception:
                        inp = ""
                out = it.get("output")
                # accept only when output available as string
                if isinstance(inp, str) and isinstance(out, str):
                    items.append({"input": inp, "output": out})
            basic_cases = items
            gen_cases = []
        else:
            if self.debug:
                print("Debug: final_cases.json missing expected keys: basic_cases/generated_cases/final_cases")

        return basic_cases, gen_cases

    # NEW: write cases into target folder; return concrete file paths
    def _write_cases_to_dir(self, cases: List[Dict[str, str]], target_dir: str) -> List[Tuple[str, str]]:
        os.makedirs(target_dir, exist_ok=True)
        paths: List[Tuple[str, str]] = []
        for i, item in enumerate(cases, start=1):
            in_path = os.path.join(target_dir, f"input_{i}")
            out_path = os.path.join(target_dir, f"output_{i}")
            
            input_content = item.get("input", "")
            output_content = item.get("output", "")
            
            if not input_content and item.get("input_path"):
                input_path = item.get("input_path")
                if os.path.exists(input_path):
                    try:
                        with open(input_path, "r", encoding="utf-8") as f:
                            input_content = f.read()
                        if self.debug:
                            print(f"Debug: Case {i} input read from input_path: {input_path}, length={len(input_content)}")
                    except Exception as e:
                        print(f"Warning: Failed to read input from {input_path}: {e}")
                else:
                    print(f"Warning: input_path does not exist: {input_path}")
            
            if not input_content:
                print(f"Warning: Case {i} has empty input content!")
            
            try:
                with open(in_path, "w", encoding="utf-8") as fi:
                    fi.write(input_content)
                with open(out_path, "w", encoding="utf-8") as fo:
                    fo.write(output_content)
            except Exception as e:
                if self.debug:
                    print(f"Debug: writing case {i} failed in {target_dir}: {e}")
                continue
            paths.append((in_path, out_path))
        return paths

    def _load_or_init_best_status(self, conversation_id: str, code: str) -> Tuple[Dict, str]:
        """
        Load best_status.json; if absent or not in new schema, generate:
        1) Read basic_cases & generated_cases from final_cases.json
        2) Materialize them into final_test_cases/basic_cases and final_test_cases/generated_cases
        3) Save best_status with separate fields
        """
        status_path = self._status_file_path(conversation_id)
        # Try load existing
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If already in new schema, return as-is
                    if "basic_cases" in data or "generated_cases" in data:
                        return data, status_path
                    # If old schema with unit_test_cases, we will rebuild from final_cases.json below
            except Exception:
                print("Debug: Failed to load data from best_status.json")
        print("Debug: Initializing best_status.json from final_cases.json")
        # Try run generator (non-fatal if fails)
        try:
            calculator = Cases_Out_Calculator()
            _ = calculator.forward()
        except Exception as e:
            if self.debug:
                print(f"Cases_Out_Calculator error: {e}")

        basic_items, gen_items = self._read_final_cases(conversation_id)
        if self.debug:
            print(f"Debug: Read {len(basic_items)} basic_cases and {len(gen_items)} generated_cases from final_cases.json")
        # Fallback: if basic_items is empty (e.g., generator failed), try building from test_data
        if not basic_items:
            try:
                # Locate test_data directory
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                try:
                    from agentflow.tools.session_manager import get_global_work_dir
                    global_dir = get_global_work_dir()
                except Exception:
                    global_dir = None

                if global_dir:
                    test_data_dir = os.path.join(global_dir, "test_data")
                else:
                    # fallback to project tools/cpp tree with conversation_id segregation
                    test_data_dir = os.path.join(project_root, "tools", "cpp", "test_data", conversation_id)

                # Collect test cases from test_data directory
                test_cases = self._get_all_test_cases(test_data_dir)
                if self.debug:
                    print(f"Debug: basic_items empty; attempting fallback from test_data at {test_data_dir}, found {len(test_cases)} case(s)")

                # Build basic_items in the same shape as final_cases.json: [{"input": str, "output": str}, ...]
                items: List[Dict[str, str]] = []
                for (inp_path, out_path, _idx) in test_cases:
                    try:
                        with open(inp_path, "r", encoding="utf-8") as fi:
                            inp = fi.read()
                    except Exception:
                        inp = ""
                    try:
                        with open(out_path, "r", encoding="utf-8") as fo:
                            out = fo.read()
                    except Exception:
                        out = ""
                    if isinstance(inp, str) and isinstance(out, str):
                        items.append({"input": inp, "output": out})

                if items:
                    basic_items = items
            except Exception as e:
                if self.debug:
                    print(f"Debug: fallback building basic_items from test_data failed: {e}")

        # Prepare final_test_cases folders and write files
        _, final_test_root = self._final_cases_locations(conversation_id)
        basic_dir = os.path.join(final_test_root, "basic_cases")
        generated_dir = os.path.join(final_test_root, "generated_cases")

        basic_paths = self._write_cases_to_dir(basic_items, basic_dir)
        gen_paths = self._write_cases_to_dir(gen_items, generated_dir)

        # Build best_status with separated fields
        best_status = {
            "current_turn": 0,
            "best_solution": f"{code}",
            "pass_rate": 0.0,
            "basic_cases": [
                {"input_path": inp, "output_path": out, "passed": False}
                for (inp, out) in basic_paths
            ],
            "generated_cases": [
                {"input_path": inp, "output_path": out, "passed": False}
                for (inp, out) in gen_paths
            ],
        }

        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(best_status, f, ensure_ascii=False, indent=2)
            

        return best_status, status_path

    def _get_signal_description(self, return_code: int) -> str:
        """
        Convert return code to human-readable signal description
        """
        signal_descriptions = {
            -1: "SIGHUP (Hangup)",
            -2: "SIGINT (Interrupt)",
            -3: "SIGQUIT (Quit)",
            -4: "SIGILL (Illegal Instruction)",
            -5: "SIGTRAP (Trace/Breakpoint Trap)",
            -6: "SIGABRT (Program Aborted)",
            -7: "SIGBUS (Bus Error)",
            -8: "SIGFPE (Floating Point Exception - Division by Zero/Arithmetic Error)",
            -9: "SIGKILL (Killed)",
            -10: "SIGUSR1 (User Signal 1)",
            -11: "SIGSEGV (Segmentation Fault - Invalid Memory Access)",
            -12: "SIGUSR2 (User Signal 2)",
            -13: "SIGPIPE (Broken Pipe)",
            -14: "SIGALRM (Alarm Clock)",
            -15: "SIGTERM (Termination)",
            -24: "SIGXCPU (CPU Time Limit Exceeded)",
            -25: "SIGXFSZ (File Size Limit Exceeded)",
            -31: "SIGSYS (Bad System Call)"
        }
        
        if return_code in signal_descriptions:
            return f"Return code {return_code}: {signal_descriptions[return_code]}"
        elif return_code < 0:
            return f"Return code {return_code}: Signal {abs(return_code)}"
        elif return_code > 0:
            return f"Return code {return_code}: Program exited with error code {return_code}"
        else:
            return "Return code 0: Success"

    def _truncate_output(self, output: str, max_chars: int = 4000) -> str:
        """Truncate output if it exceeds character limit to prevent excessive verbosity (roughly 1000 tokens)"""
        if len(output) > max_chars:
            truncated = output[:max_chars]
            return truncated + f"\n\n[Output truncated after {max_chars} characters to prevent excessive verbosity]"
        return output
    
    def _get_all_test_cases(self, test_data_dir: str) -> List[Tuple[str, str, int]]:
        """
        Get all available test cases from the directory
        
        Returns:
            List of (input_path, output_path, test_number) tuples
        """
        test_cases = []
        
        # Check for single test case (input/output)
        input_path = os.path.join(test_data_dir, "input")
        output_path = os.path.join(test_data_dir, "output")
        
        if os.path.exists(input_path) and os.path.exists(output_path):
            test_cases.append((input_path, output_path, 1))
            return test_cases
        
        # Check for multiple test cases (input_1, input_2, etc.)
        test_num = 1
        while True:
            input_path = os.path.join(test_data_dir, f"input_{test_num}")
            output_path = os.path.join(test_data_dir, f"output_{test_num}")
            
            if os.path.exists(input_path) and os.path.exists(output_path):
                test_cases.append((input_path, output_path, test_num))
                test_num += 1
            else:
                break
        
        return test_cases

    def _compile_spj(self, spj_code: str) -> Optional[str]:
        """Compile SPJ code and return path to executable"""
        import tempfile
        import subprocess
        
        try:
            # Create a temporary directory for the SPJ
            temp_dir = tempfile.mkdtemp(prefix="spj_")
            spj_src = os.path.join(temp_dir, "spj.cpp")
            spj_exe = os.path.join(temp_dir, "spj.exe") if os.name == 'nt' else os.path.join(temp_dir, "spj")
            
            with open(spj_src, "w", encoding="utf-8") as f:
                f.write(spj_code)
            
            # Compile SPJ
            compile_cmd = ["g++", "-std=c++17", "-O2", spj_src, "-o", spj_exe]
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return spj_exe
            else:
                print(f"SPJ Compilation failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error compiling SPJ: {e}")
            return None

    def _run_spj(self, spj_executable: str, expected_output: str, actual_output: str) -> bool:
        """Run SPJ and return True if Accepted, False otherwise"""
        import tempfile
        import subprocess
        
        try:
            # Create temporary files for outputs
            with tempfile.NamedTemporaryFile(mode='w', suffix='.expected', delete=False) as f_exp:
                f_exp.write(expected_output)
                exp_path = f_exp.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.actual', delete=False) as f_act:
                f_act.write(actual_output)
                act_path = f_act.name
            
            try:
                # Run SPJ: spj_exe expected_output actual_output
                result = subprocess.run([spj_executable, exp_path, act_path], capture_output=True, text=True, timeout=10)
                
                # Check output for "Accepted" or return code 0
                is_accepted = "Accepted" in result.stdout or result.returncode == 0
                return is_accepted
            finally:
                # Cleanup temp files
                if os.path.exists(exp_path): os.remove(exp_path)
                if os.path.exists(act_path): os.remove(act_path)
        except Exception as e:
            print(f"Error running SPJ: {e}")
            return False

    def _run_single_test_case(self, code: str, input_path: str, output_path: str, test_num: int, timeout: int, memory_limit: int = 256, spj_executable: str = None) -> Dict:
        """
        Run a single test case and return the result
        
        Returns:
            Dictionary with test result information
        """
        try:
            # Read test input and expected output
            with open(input_path, 'rb') as f:
                input_content = f.read()
            input_text = input_content.decode('utf-8')
            with open(output_path, 'r', encoding='utf-8') as f:
                expected_output = f.read()

            payload = {
                "code": code.strip(),
                "language": "cpp",
                "run_timeout": timeout,
                "memory_limit_MB": memory_limit,
                "files": {"input": base64.b64encode(input_content).decode('utf-8')},
            }

            # Submit to sandbox with infinite retry
            response = self._post_to_sandbox_with_retry(payload, timeout=timeout + 30)
            if response.status_code != 200:
                return {
                    'test_num': test_num,
                    'status': 'network_error',
                    'error': f"HTTP {response.status_code}: {response.text[:200]}",
                    'input': input_text.strip(),
                    'expected': expected_output.strip(),
                    'actual': ''
                }

            result = response.json()
            compile_result = result.get('compile_result', {}) or {}
            run_result = result.get('run_result', {}) or {}

            # Compilation error
            if compile_result.get('status') != 'Finished' or compile_result.get('return_code') not in (0, None):
                return {
                    'test_num': test_num,
                    'status': 'compile_error',
                    'error': f"Compilation failed:\n{compile_result.get('stderr', '')}",
                    'input': input_text.strip(),
                    'expected': expected_output.strip(),
                    'actual': ''
                }

            # Execution not finished or missing
            if run_result.get('status') != 'Finished':
                exec_status = run_result.get('status', 'Unknown')
                exec_stderr = run_result.get('stderr', '')
                exec_stdout = run_result.get('stdout', '')
                error_parts = [f"Execution failed - Status: {exec_status}"]
                if exec_stderr:
                    error_parts.append(f"Stderr: {exec_stderr}")
                if exec_stdout:
                    error_parts.append(f"Stdout: {exec_stdout}")
                return {
                    'test_num': test_num,
                    'status': 'runtime_error',
                    'error': '\n'.join(error_parts),
                    'input': input_text.strip(),
                    'expected': expected_output.strip(),
                    'actual': exec_stdout.strip() if exec_stdout else ''
                }

            # Non-zero return code
            if run_result.get('return_code', 0) != 0:
                exec_return_code = run_result.get('return_code', 0)
                exec_stderr = run_result.get('stderr', '')
                exec_stdout = run_result.get('stdout', '')
                error_parts = [self._get_signal_description(exec_return_code)]
                if exec_stderr:
                    error_parts.append(f"Stderr: {exec_stderr}")
                if exec_stdout:
                    error_parts.append(f"Stdout: {exec_stdout}")
                return {
                    'test_num': test_num,
                    'status': 'runtime_error',
                    'error': '\n'.join(error_parts),
                    'input': input_text.strip(),
                    'expected': expected_output.strip(),
                    'actual': exec_stdout.strip() if exec_stdout else ''
                }

            stdout = run_result.get('stdout', '')
            stderr = run_result.get('stderr', '')
            if stderr:
                return {
                    'test_num': test_num,
                    'status': 'runtime_error',
                    'error': f"Runtime error:\n{stderr}",
                    'input': input_text.strip(),
                    'expected': expected_output.strip(),
                    'actual': stdout.strip()
                }

            # Compare outputs
            if spj_executable:
                is_passed = self._run_spj(spj_executable, expected_output, stdout)
                actual_clean = stdout.strip()
                expected_clean = expected_output.strip()
            else:
                actual_clean = stdout.strip()
                expected_clean = expected_output.strip()
                is_passed = (actual_clean == expected_clean)
            
            if is_passed:
                return {
                    'test_num': test_num,
                    'status': 'passed',
                    'error': '',
                    'input': input_text.strip(),
                    'expected': expected_clean,
                    'actual': actual_clean
                }
            else:
                return {
                    'test_num': test_num,
                    'status': 'wrong_answer',
                    'error': 'Output mismatch',
                    'input': input_text.strip(),
                    'expected': expected_clean,
                    'actual': actual_clean
                }

        except Exception as e:
            return {
                'test_num': test_num,
                'status': 'error',
                'error': str(e),
                'input': '',
                'expected': '',
                'actual': ''
            }

    def _run_test_group(
        self,
        group: str,
        test_cases: List[Tuple[str, str, int]],
        code: str,
        timeout: int,
        memory_limit: int,
        max_workers: int,
        spj_executable: str = None
    ) -> List[Dict]:
        """
        Run a group of test cases in parallel and return sorted results with group metadata attached.
        """
        if not test_cases:
            return []

        if self.debug:
            print(f"Debug: Running {group.upper()} test cases...")

        results: List[Dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: List[Tuple[int, concurrent.futures.Future]] = []
            for input_path, output_path, local_idx in test_cases:
                fut = executor.submit(
                    self._run_single_test_case, code, input_path, output_path, local_idx, timeout, memory_limit, spj_executable
                )
                futures.append((local_idx, fut))

            for local_idx, fut in futures:
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        'test_num': local_idx,
                        'status': 'error',
                        'error': f"Parallel execution error: {str(e)}",
                        'input': '',
                        'expected': '',
                        'actual': ''
                    }
                r['group'] = group
                r['local_idx'] = local_idx
                results.append(r)

        results.sort(key=lambda r: r['test_num'])
        return results

    def _format_results_summary(self, results: List[Dict], code: str = "") -> str:
        """
        Format the aggregated test results into a readable summary
        """
        if not results:
            return "No test cases found"
        
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'passed'])
        failed_tests = total_tests - passed_tests
        
        # Summary header
        if passed_tests == total_tests:
            summary = f"ALL TESTS PASSED ({passed_tests}/{total_tests})"
        else:
            summary = f"TESTS FAILED ({passed_tests}/{total_tests} passed)"
        
        summary += f"\n{'='*50}\n"
        
        # Check if all tests have the same error (e.g., all compilation errors)
        common_errors = set()
        for result in results:
            if result['status'] != 'passed':
                common_errors.add(result['status'])
        
        # If all tests failed with the same type of error, show the error details once
        if len(common_errors) == 1 and failed_tests == total_tests:
            error_type = list(common_errors)[0]
            if error_type in ['compile_error', 'runtime_error']:
                summary += f"{error_type.replace('_', ' ').upper()} - All tests failed\n"
                # Show the full error message from the first test
                first_error = results[0]['error']
                summary += f"{first_error}\n\n"
                # Only show test numbers for this case
                for result in results:
                    summary += f"Test {result['test_num']}: FAILED\n"
                
                print(f"[DEBUG] _format_results_summary: code length={len(code) if code else 0}, enable_error_analysis={self.enable_error_analysis}")
                if code and self.enable_error_analysis:
                    error_type_mapping = {
                        'compile_error': 'compilation',
                        'runtime_error': 'runtime'
                    }
                    reminder = self._analyze_error(
                        first_error,
                        error_type_mapping.get(error_type, 'runtime'),
                        code
                    )
                    if reminder:
                        summary += reminder
                else:
                    if not code:
                        print(f"[DEBUG] Skipping error analysis: code is empty")
                    if not self.enable_error_analysis:
                        print(f"[DEBUG] Skipping error analysis: enable_error_analysis=False")
                
                return self._truncate_output(summary)
        
        # Details for each test case
        for result in results:
            test_num = result['test_num']
            status = result['status']
            
            if status == 'passed':
                summary += f"Test {test_num}: PASSED\n"
            else:
                status_display = {
                    'wrong_answer': 'WRONG ANSWER',
                    'compile_error': 'COMPILE ERROR',
                    'runtime_error': 'RUNTIME ERROR',
                    'sandbox_error': 'EXECUTION ERROR',
                    'network_error': 'NETWORK ERROR',
                    'error': 'ERROR'
                }.get(status, status.upper())
                
                summary += f"Test {test_num}: FAILED ({status_display})\n"
                
                # Show error details for individual failures
                if result['error']:
                    summary += f"  Error: {result['error']}\n"
                
                summary += f"  Input: {result['input'][:100]}{'...' if len(result['input']) > 100 else ''}\n"
                summary += f"  Expected: {result['expected'][:100]}{'...' if len(result['expected']) > 100 else ''}\n"
                
                if result['actual']:
                    summary += f"  Actual: {result['actual'][:100]}{'...' if len(result['actual']) > 100 else ''}\n"
                
                summary += "\n"
        
        if failed_tests > 0 and code and self.enable_error_analysis:
            has_wrong_answer = any(r.get('status') == 'wrong_answer' for r in results)
            
            if has_wrong_answer:
                print(f"[DEBUG] 检测到 wrong_answer 错误，准备进行深度分析...")
                
                first_wa = next((r for r in results if r.get('status') == 'wrong_answer'), None)
                if first_wa:
                    validation_error = f"""Output mismatch
Input: {first_wa['input']}
Expected: {first_wa['expected']}
Actual: {first_wa['actual']}"""
                    
                    reminder = self._analyze_error(
                        validation_error,
                        'validation',
                        code
                    )
                    if reminder:
                        summary += reminder
            
        return self._truncate_output(summary)

    def _validate_test_case(self, actual_output: str, expected_output: str, input_data: str, code: str = "") -> str:
        """Validate a single test case and return detailed result (legacy method)"""
        # Strip whitespace for comparison
        actual_clean = actual_output.strip()
        expected_clean = expected_output.strip()
        
        if actual_clean == expected_clean:
            return "Test PASSED: Output matches expected result"
        else:
            result = f"""Test FAILED: Output mismatch

Input Data:
{input_data.strip()}

Expected Output:
{expected_clean}

Actual Output:
{actual_clean}"""
            return self._truncate_output(result)
    # Removed legacy helpers (_get_all_test_cases, _validate_test_case) as they are unused

    def _fix_broken_char_literals(self, code: str) -> str:

        if not code:
            return code
            
        original_code = code

        code = re.sub(r"'\s*\n\s*'", r"'\\n'", code)
        code = re.sub(r"'\t'", r"'\\t'", code)
        code = re.sub(r"'\r'", r"'\\r'", code)
        
        def fix_broken_string(match):
            content = match.group(1)
            if '\n' in content or '\t' in content or '\r' in content:
                fixed = content.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                return f'"{fixed}"'
            return match.group(0)
        
        code = re.sub(r'"([^"]*?)"', fix_broken_string, code)
        
        def fix_broken_comment(match):
            comment_start = match.group(1)
            broken_content = match.group(2)
            cpp_indicators = [';', '{', '}', '#include', 'int ', 'void ', 'return', 'for', 'while', 'if', 'else']
            is_likely_code = any(ind in broken_content for ind in cpp_indicators)
            if not is_likely_code and broken_content.strip():
                return f"{comment_start} {broken_content.strip()}"
            return match.group(0)
        
        code = re.sub(r'(//[^\n]*)\n([^\n]*?)(?=\n|$)', fix_broken_comment, code)
        

        return code

    def _lazy_generate_test_cases(self, conversation_id: str) -> bool:

        if conversation_id in self._generated_cases_triggered_sessions:
            if self.debug:
                print(f"Debug: Test cases already generated for session {conversation_id}, skipping...")
            return False
        
        self._generated_cases_triggered_sessions.add(conversation_id)
        print("\n" + "=" * 60)
        print("=" * 60)
        
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
        
        if not global_dir:
            return False
        
        question_path = os.path.join(global_dir, "unit_test_data", "original_question.txt")
        if not os.path.exists(question_path):
            return False
        
        try:
            with open(question_path, 'r', encoding='utf-8') as f:
                question = f.read()
        except Exception as e:
            return False
        
        try:
            from agentflow.tools.cases_in_generator import Cases_In_Generator
            from agentflow.tools.bf_solution_generator import Bf_Solution_Generator
            from agentflow import config
            
            cases_in_generator = Cases_In_Generator(
                model_id=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key
            )
            cases_in_generator.forward(question, conversation_id=conversation_id, language="cpp")
            
            bf_solution_generator = Bf_Solution_Generator(
                model_id=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key
            )
            sample_n = getattr(config, 'BF_SOLUTION_SAMPLE_N', 1)
            bf_solution_generator.forward(sample_N=sample_n)
            
            calculator = Cases_Out_Calculator()
            calculator.forward()
            
            print("=" * 60 + "\n")
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("=" * 60 + "\n")
            return False

    def forward(self, code: str,) -> str:
        code = self._fix_broken_char_literals(code)
        
        try:
            from agentflow.tools.session_manager import get_current_session_id, get_run_timeout, get_memory_limit
            conversation_id = get_current_session_id()
            if not conversation_id:
                return "Error: No active conversation session. Please use run_coding_agent_with_session()."

            # Get global run timeout and memory limit settings
            global_timeout = get_run_timeout()
            global_memory_limit = get_memory_limit()
        except ImportError:
            return "Error: Cannot access conversation session management."
        
        # Load or initialize best_status.json (now using separated basic/generated cases)
        best_status, status_path = self._load_or_init_best_status(conversation_id, code)

        # Build per-group test case tuples WITHOUT merging
        basic_test_cases: List[Tuple[str, str, int]] = []
        generated_test_cases: List[Tuple[str, str, int]] = []

        def add_group(items: List[Dict], acc: List[Tuple[str, str, int]]):
            for local_idx, case in enumerate(items, start=1):
                inp = case.get("input_path")
                out = case.get("output_path")
                if inp and out and os.path.exists(inp) and os.path.exists(out):
                    acc.append((inp, out, local_idx))

        if "basic_cases" in best_status:
            add_group(best_status.get("basic_cases", []), basic_test_cases)
        if "generated_cases" in best_status:
            add_group(best_status.get("generated_cases", []), generated_test_cases)

        if not basic_test_cases and not generated_test_cases:
            return f"Error: No valid test cases found based on {status_path}"

        # Inject freopen into code for input redirection
        modified_code = self._inject_freopen(code)
        timeout = global_timeout

        # Check for SPJ
        spj_code = get_spj_code()
        spj_executable = None
        if spj_code:
            if self.debug:
                print("Debug: SPJ code found, compiling...")
            spj_executable = self._compile_spj(spj_code)
            if spj_executable and self.debug:
                print(f"Debug: SPJ compiled to {spj_executable}")

        # Always run with ThreadPoolExecutor; worker count based on CPU resources (cap at 8)
        try:
            cpu_workers = os.cpu_count() or 1
            max_workers = min(8, max(1, cpu_workers))
        except Exception:
            max_workers = 4

        if self.debug:
            total_cases = len(basic_test_cases) + len(generated_test_cases)
            print(f"Debug: Prepared {len(basic_test_cases)} basic and {len(generated_test_cases)} generated test cases "
                  f"(total {total_cases}), max_workers={max_workers}, status_path={status_path}")

        results_basic: List[Dict] = []
        results_generated: List[Dict] = []

        # 1) Run BASIC cases first
        if basic_test_cases:
            results_basic = self._run_test_group(
                'basic', basic_test_cases, modified_code, timeout, global_memory_limit, max_workers, spj_executable
            )

        # Determine if we can proceed to GENERATED cases
        can_run_generated = True
        if basic_test_cases:
            can_run_generated = all(r.get('status') == 'passed' for r in results_basic)

        if can_run_generated and not generated_test_cases and conversation_id not in self._generated_cases_triggered_sessions:
            if self._lazy_generate_test_cases(conversation_id):
                try:
                    _, gen_items = self._read_final_cases(conversation_id)
                    if gen_items:
                        _, final_test_root = self._final_cases_locations(conversation_id)
                        generated_dir = os.path.join(final_test_root, "generated_cases")
                        gen_paths = self._write_cases_to_dir(gen_items, generated_dir)
                        
                        with open(status_path, "r", encoding="utf-8") as f:
                            cur_status = json.load(f)
                        cur_status["generated_cases"] = [
                            {"input_path": inp, "output_path": out, "passed": False}
                            for (inp, out) in gen_paths
                        ]
                        with open(status_path, "w", encoding="utf-8") as f:
                            json.dump(cur_status, f, ensure_ascii=False, indent=2)
                        
                        add_group(cur_status.get("generated_cases", []), generated_test_cases)
                    else:
                        print("No generated_cases")
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        # 2) Run GENERATED cases only if BASIC all passed (or no basic cases exist)
        if generated_test_cases and can_run_generated:
            if self.debug:
                print("Debug: BASIC passed. Running GENERATED test cases...")
            results_generated = self._run_test_group(
                'generated', generated_test_cases, modified_code, timeout, global_memory_limit, max_workers, spj_executable
            )
        elif generated_test_cases and not can_run_generated and self.debug:
            print("Debug: BASIC did not fully pass. Skipping GENERATED test cases.")

        # Build grouped summary
        parts = []
        if results_basic or basic_test_cases:
            parts.append("BASIC CASES\n" + (self._format_results_summary(results_basic, code) if results_basic else "No test cases found"))
        if generated_test_cases:
            parts.append("\n")
            if results_generated:
                parts.append("GENERATED CASES\n" + self._format_results_summary(results_generated, code))
            else:
                parts.append("GENERATED CASES\nSKIPPED: Basic cases must all pass before running generated cases.")
        summary = "".join(parts)

        # Update best_status.json; only apply results for groups that actually ran
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                cur_status = json.load(f)

            basic_list = cur_status.get("basic_cases", [])
            gen_list = cur_status.get("generated_cases", [])

            def apply_results(results: List[Dict], lst: List[Dict]):
                for r in results:
                    idx0 = r['local_idx'] - 1  # 0-based
                    if 0 <= idx0 < len(lst):
                        prev = lst[idx0].get("passed", False)
                        now = (r.get('status') == 'passed')
                        if not (prev and not now):  # do not flip True -> False
                            lst[idx0]['passed'] = now

            if results_basic:
                apply_results(results_basic, basic_list)
            if results_generated:
                apply_results(results_generated, gen_list)

            def count_pass(lst: List[Dict]) -> Tuple[int, int]:
                return sum(1 for c in lst if c.get("passed")), len(lst)

            b_pass, b_total = count_pass(basic_list)
            g_pass, g_total = count_pass(gen_list)
            total = b_total + g_total
            cur_status["pass_rate"] = ((b_pass + g_pass) / total * 100.0) if total else 0.0

            cur_status["basic_cases"] = basic_list
            cur_status["generated_cases"] = gen_list

            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(cur_status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            summary += f"\n\nError updating best_status.json: {str(e)}"

        # Update last_run_stats
        self.last_run_stats = {
            "generated_run": bool(results_generated),
            "generated_passed": bool(results_generated) and all(r.get('status') == 'passed' for r in results_generated),
            "has_generated_cases": bool(generated_test_cases)
        }

        # Cleanup SPJ
        if spj_executable:
            try:
                import shutil
                if os.path.exists(os.path.dirname(spj_executable)):
                    shutil.rmtree(os.path.dirname(spj_executable))
            except: pass

        return summary
    
def create_cpp_validation_tool(
    sandbox_url,
    debug: bool = False,
    default_run_timeout: int = 30,
    enable_error_analysis: bool = True,
    model_id: str = None,
    api_base: str = None,
    api_key: str = None
) -> CppValidationTool:
    """
    Create a C++ validation tool instance
    
    Args:
        sandbox_url: The URL address of the sandbox service
        debug: Enable debug output for troubleshooting
        default_run_timeout: Default timeout for code execution in seconds
        enable_error_analysis: Whether to enable automatic error analysis
        model_id: LLM model ID for error analysis
        api_base: API endpoint for error analysis
        api_key: API key for error analysis
        
    Returns:
        CppValidationTool instance
    """
    return CppValidationTool(
        sandbox_url=sandbox_url,
        debug=debug,
        default_run_timeout=default_run_timeout,
        enable_error_analysis=enable_error_analysis,
        model_id=model_id,
        api_base=api_base,
        api_key=api_key
    )


# Example usage
if __name__ == "__main__":
    """
    Ad-hoc tester for CppValidationTool.

    What this does:
    - Sets a session_id and global work_dir so the tool can locate final_cases.json and test_data.
    - Runs the validator with a provided C++ code snippet.
    - Prints a concise summary (pass/fail by groups).

    Notes:
    - The work_dir must contain unit_test_data/final_cases.json OR test_data/ with input/output files.
    - The code under test should read from stdin; the tool auto-injects freopen("input", "r", stdin).
    """

    from agentflow.tools.session_manager import (
        set_current_session_id,
        set_global_work_dir,
        set_run_timeout,
    )

    # 1) Configure session and workspace
    # You can change these two lines to point to your own prepared session folder
    session_id = "test_data"
    from agentflow import config
    work_dir_base = config.WORK_DIR
    if not os.path.isabs(work_dir_base):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        work_dir_base = os.path.join(project_root, work_dir_base)
    work_dir = os.path.join(work_dir_base, session_id)

    set_current_session_id(session_id)
    set_global_work_dir(work_dir)
    set_run_timeout(10)  # seconds

    # 2) Provide a test C++ program
    # This demo program reads the problem input but outputs a placeholder result.
    # It is intended to verify the validation pipeline works end-to-end.
    demo_code = """
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

class Solution {
public:
    int shortestMatchingSubstring(string s, string p) {
        int n = s.length();
        int m = p.length();
        
        // Find positions of '*' in pattern
        vector<int> star_pos;
        for (int i = 0; i < m; i++) {
            if (p[i] == '*') {
                star_pos.push_back(i);
            }
        }
        
        // There should be exactly two '*'
        if (star_pos.size() != 2) {
            return -1;
        }
        
        int min_length = INT_MAX;
        
        // Split pattern into three parts: before first *, between *, and after second *
        string part1 = p.substr(0, star_pos[0]);
        string part2 = p.substr(star_pos[0] + 1, star_pos[1] - star_pos[0] - 1);
        string part3 = p.substr(star_pos[1] + 1);
        
        // Check empty substring case - if all parts are empty
        if (part1.empty() && part2.empty() && part3.empty()) {
            return 0;
        }
        
        // Try all possible starting positions for the substring
        for (int i = 0; i < n; i++) {
            // Try all possible ending positions
            for (int j = i; j <= n; j++) {
                int len = j - i;
                string substr = s.substr(i, len);
                
                // Check if this substring matches the pattern
                if (matchesPattern(substr, part1, part2, part3)) {
                    min_length = min(min_length, len);
                }
            }
        }
        
        return min_length == INT_MAX ? -1 : min_length;
    }
    
private:
    bool matchesPattern(const string& str, const string& part1, const string& part2, const string& part3) {
        int len = str.length();
        int idx = 0;
        
        // Match part1 at the beginning
        for (char c : part1) {
            if (idx >= len || str[idx] != c) return false;
            idx++;
        }
        
        // Match part3 at the end
        if (len < idx + part3.length()) return false;
        for (int i = 0; i < part3.length(); i++) {
            if (str[len - part3.length() + i] != part3[i]) return false;
        }
        int before_part3 = len - part3.length();
        
        // Now we need to find part2 somewhere between idx and before_part3
        if (part2.empty()) {
            // Empty part2 always matches
            return true;
        }
        
        // Check if part2 appears in the middle section
        for (int pos = idx; pos <= before_part3 - part2.length(); pos++) {
            bool match = true;
            for (int i = 0; i < part2.length(); i++) {
                if (str[pos + i] != part2[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return true;
            }
        }
        
        return false;
    }
};

// ---

int main() {
    Solution solution;
    string s, p;
    
    // Read input
    cin >> s >> p;
    
    // Compute and output result
    int result = solution.shortestMatchingSubstring(s, p);
    cout << result << endl;
    
    return 0;
}
"""

    # Optionally allow overriding the demo code via an environment variable path
    import os as _os
    override_code_path = _os.environ.get("CPP_VALIDATION_CODE_PATH", "")
    if override_code_path and _os.path.exists(override_code_path):
        try:
            with open(override_code_path, "r", encoding="utf-8") as f:
                demo_code = f.read()
        except Exception as e:
            print(f"Failed to read override code from {override_code_path}: {e}")

    # 3) Create tool (enable debug to see more details)
    cpp_tool = create_cpp_validation_tool(debug=True, enable_error_analysis=False)

    # Sanity checks for paths
    print(f"[MAIN] session_id: {session_id}")
    print(f"[MAIN] work_dir:   {work_dir}")
    print(f"[MAIN] final_cases.json exists: {_os.path.exists(_os.path.join(work_dir, 'unit_test_data', 'final_cases.json'))}")
    print(f"[MAIN] test_data dir exists:    {_os.path.exists(_os.path.join(work_dir, 'test_data'))}")

    # 4) Run validation
    print("\n[MAIN] Running validation...\n")
    summary = cpp_tool.forward(demo_code)
    print(summary)

