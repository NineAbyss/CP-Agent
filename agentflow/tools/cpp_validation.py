from smolagents import Tool
import requests
import base64
import os
import re
from typing import Optional, List, Dict, Tuple
import concurrent.futures
import threading
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
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        self._error_summarizer = None

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
                print(f"Error loading error_summarizer: {e}")
        return self._error_summarizer

    def _post_to_sandbox_with_retry(self, payload: dict, timeout: int = None):

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
            print(f" {e}")
        
        return ""

    def _normalize_output(self, text: str) -> str:

        lines = text.splitlines()
        normalized_lines = [line.rstrip() for line in lines]
        result = '\n'.join(normalized_lines).strip()
        return result
    
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
        """Truncate output if it exceeds character limit to prevent excessive verbosity (roughly 1000 tokens)
        
        Note: Protects error reminders from truncation
        """
        if len(output) > max_chars:
            reminder_marker = "Things I should notice while correcting my code:"
            if reminder_marker in output:
                parts = output.split(reminder_marker, 1)
                error_part = parts[0]
                reminder_part = reminder_marker + parts[1] if len(parts) > 1 else ""
                
                if len(error_part) > max_chars - len(reminder_part):
                    error_part = error_part[:max_chars - len(reminder_part) - 100] + "\n\n[Output truncated to prevent excessive verbosity]"
                
                return error_part + reminder_part
            else:
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
            # Read test input file
            with open(input_path, 'rb') as f:
                input_content = f.read()
            input_base64 = base64.b64encode(input_content).decode('utf-8')
            input_text = input_content.decode('utf-8')
            
            # Read expected output file
            with open(output_path, 'r', encoding='utf-8') as f:
                expected_output = f.read()
            
            payload = {
                "code": code.strip(),
                "language": "cpp",
                "run_timeout": timeout,
                "memory_limit_MB": memory_limit,
                "files": {"input": input_base64}
            }
            
            # Submit to sandbox with infinite retry
            response = self._post_to_sandbox_with_retry(payload, timeout=timeout + 30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'Success':
                    compile_result = result.get('compile_result', {})
                    run_result = result.get('run_result', {})
                    
                    # Check compilation
                    if compile_result.get('status') != 'Finished' or compile_result.get('return_code') != 0:
                        return {
                            'test_num': test_num,
                            'status': 'compile_error',
                            'error': f"Compilation failed:\n{compile_result.get('stderr', '')}",
                            'input': input_text.strip(),
                            'expected': expected_output.strip(),
                            'actual': ''
                        }
                    
                    # Check execution
                    if run_result.get('status') != 'Finished':
                        # Get detailed execution error information
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
                    
                    # Check if execution finished but with non-zero return code
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
                    
                    # Check for runtime errors
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
                        # Use normalization method to handle trailing spaces
                        actual_clean = self._normalize_output(stdout)
                        expected_clean = self._normalize_output(expected_output)
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
                else:
                    # Check if it's a compilation failure
                    compile_result = result.get('compile_result', {})
                    if compile_result and compile_result.get('return_code') != 0:
                        return {
                            'test_num': test_num,
                            'status': 'compile_error',
                            'error': f"Compilation failed:\n{compile_result.get('stderr', '')}",
                            'input': input_text.strip(),
                            'expected': expected_output.strip(),
                            'actual': ''
                        }
                    
                    # Check if it's a runtime failure (execution finished but with error)
                    run_result = result.get('run_result', {})
                    if run_result and run_result.get('return_code', 0) != 0:
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
                    
                    return {
                        'test_num': test_num,
                        'status': 'sandbox_error',
                        'error': result.get('message', 'Unknown sandbox error'),
                        'input': input_text.strip(),
                        'expected': expected_output.strip(),
                        'actual': ''
                    }
            else:
                return {
                    'test_num': test_num,
                    'status': 'network_error',
                    'error': f"HTTP {response.status_code}: {response.text[:200]}",
                    'input': input_text.strip(),
                    'expected': expected_output.strip(),
                    'actual': ''
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
            summary = f"✅ ALL TESTS PASSED ({passed_tests}/{total_tests})"
        else:
            summary = f"❌ TESTS FAILED ({passed_tests}/{total_tests} passed)"
        
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
                
                if failed_tests > 0 and code:
                    error_type_mapping = {
                        'compile_error': 'compilation',
                        'runtime_error': 'runtime'
                    }
                    reminder = self._analyze_error(
                        first_error, 
                        error_type_mapping.get(error_type, 'runtime'), 
                        code
                    )
                    summary += reminder
                
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
            try:
                first_failed = next((r for r in results if r['status'] != 'passed'), None)
                if first_failed:
                    print(f"[DEBUG] Analyzing first failed test, type: {first_failed['status']}")
                    
                    error_type_mapping = {
                        'compile_error': 'compilation',
                        'runtime_error': 'runtime',
                        'wrong_answer': 'validation',
                        'sandbox_error': 'runtime',
                        'network_error': 'runtime'
                    }
                    error_type = error_type_mapping.get(first_failed['status'], 'validation')
                    
                    if error_type == 'validation':
                        error_msg = f"Output mismatch\nInput: {first_failed['input'][:200]}\nExpected: {first_failed['expected'][:200]}\nActual: {first_failed['actual'][:200]}"
                    else:
                        error_msg = first_failed['error']
                    
                    reminder = self._analyze_error(error_msg, error_type, code)
            except Exception as e:
                print(f"[DEBUG] Error analysis failed: {e}")
                pass  
        
        return self._truncate_output(summary)

    def _validate_test_case(self, actual_output: str, expected_output: str, input_data: str) -> str:
        """Validate a single test case and return detailed result (legacy method)"""
        actual_clean = self._normalize_output(actual_output)
        expected_clean = self._normalize_output(expected_output)
        
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

    def forward(self, code: str) -> str:
        """
        Validate C++ code against local test cases
        
        Args:
            code: C++ source code to validate
            
        Returns:
            Validation result with detailed I/O information for failed tests
        """
        code = self._fix_broken_char_literals(code)
        
        # Get current conversation ID for test data location
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
        
        # Locate test data files
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
        if global_dir:
            test_data_dir = os.path.join(global_dir, "test_data")
        else:
            test_data_dir = os.path.join(project_root, "workdir", conversation_id, "test_data")
        
        # Ensure test data directory exists
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir, exist_ok=True)
        
        # Get all available test cases
        test_cases = self._get_all_test_cases(test_data_dir)
        
        if not test_cases:
            return f"Error: No test files found in {test_data_dir}"
        
        # Inject freopen into code for input redirection
        modified_code = self._inject_freopen(code)
        
        # Use global timeout setting
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
        
        # Run test cases
        try:
            results = []
            if len(test_cases) == 1:
                # Single test case
                input_path, output_path, test_num = test_cases[0]
                result = self._run_single_test_case(
                    modified_code,
                    input_path,
                    output_path,
                    test_num,
                    timeout,
                    global_memory_limit,
                    spj_executable
                )
                results.append(result)
            else:
                # Multiple test cases - run in parallel
                if self.debug:
                    print(f"Debug: Running {len(test_cases)} test cases in parallel")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(test_cases))) as executor:
                    futures = []
                    for input_path, output_path, test_num in test_cases:
                        future = executor.submit(
                            self._run_single_test_case,
                            modified_code,
                            input_path,
                            output_path,
                            test_num,
                            timeout,
                            global_memory_limit,
                            spj_executable
                        )
                        futures.append(future)
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            results.append(future.result())
                        except Exception as e:
                            results.append({
                                'test_num': 0,
                                'status': 'error',
                                'error': f"Parallel execution error: {str(e)}",
                                'input': '',
                                'expected': '',
                                'actual': ''
                            })
            
            # Sort results by test number
            results.sort(key=lambda x: x['test_num'])
            return self._format_results_summary(results, code)
        except Exception as e:
            return f"Error during validation: {str(e)}"
        finally:
            # Final safety cleanup for SPJ
            if spj_executable:
                try:
                    import shutil
                    if os.path.exists(os.path.dirname(spj_executable)):
                        shutil.rmtree(os.path.dirname(spj_executable))
                except: pass


# Convenience function for creating tool instance
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
    # Create tool instance
    cpp_tool = create_cpp_validation_tool()
    
    # Test with simple C++ code (requires test files)
    test_code = """
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
    
    # Note: This tool validates code against local test cases
    # You need to provide a conversation_id with corresponding test files
    print("CppValidationTool validates code against local test cases.")
    print("Requires input and output files in test_data/{conversation_id}/ directory.")

