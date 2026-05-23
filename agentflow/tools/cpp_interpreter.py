from smolagents import Tool
import requests
from typing import Optional
import os
import re

class CppInterpreterTool(Tool):
    """
    A tool for compiling and executing C++ code using a local sandbox environment.
    Provides direct execution of C++ code without validation.
    """
    
    name = "cpp_interpreter"
    description = """
This tool compiles and executes C++ code in a local sandbox environment.
It takes C++ code as input, compiles it, and returns the execution output. The tool can be used for quick numerical computation and verification: write small C++ programs to compute results (e.g., prefix sums, combinations, matrix operations, simulations, randomized experiments), and treat the printed output as a reliable computed reference to support your reasoning.
**When input is needed, you should hard-code the input directly into the C++ program** (e.g., embed test data in the code). This is not a tool for validating the code you intend to submit.
If any compilation or runtime errors occur, it will return the corresponding error messages, along with historical error reminders.
    """
    
    inputs = {
        "code": {
            "type": "string", 
            "description": "The C++ code to compile and execute."
        }
    }
    output_type = "string"

    def __init__(
        self, 
        sandbox_url,
        run_timeout: int = 60,
        compile_timeout: int = 60,
        enable_error_analysis: bool = True,
        model_id: str = None,
        api_base: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize the C++ interpreter tool
        
        Args:
            sandbox_url: The URL address of the sandbox service
            run_timeout: Timeout for code execution in seconds (default: 30)
            compile_timeout: Timeout for compilation in seconds (default: 60)
            enable_error_analysis: Whether to enable automatic error analysis
            model_id: LLM model ID for error analysis
            api_base: API endpoint for error analysis
            api_key: API key for error analysis
        """
        super().__init__(**kwargs)
        self.sandbox_url = sandbox_url
        self.run_timeout = run_timeout
        self.compile_timeout = compile_timeout
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
                self.enable_error_analysis = False
        return self._error_summarizer

    def _post_to_sandbox_with_retry(self, payload: dict, timeout: int = None):

        if timeout is None:
            timeout = self.compile_timeout + self.run_timeout + 30
        
        try:
            response = requests.post(self.sandbox_url, json=payload, timeout=timeout)
            return response
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            print(f"[CppInterpreter Sandbox Error] {repr(e)}")
            print("Sleeping for 30 seconds before retry...")
            from time import sleep
            sleep(30)
            return self._post_to_sandbox_with_retry(payload, timeout)
        except Exception as e:
            print(f"[CppInterpreter Sandbox] Fatal error (non-retryable): {repr(e)}")
            raise e
    
    def _analyze_error(self, error_message: str, error_type: str, code: str):
        if not self.enable_error_analysis:
            return ""
        
        try:
            summarizer = self._get_error_summarizer()
            if summarizer:
                print("\n" + "=" * 60)
                
                # For validation errors, pass the full code; for compilation/runtime errors, pass the first 500 characters
                code_to_analyze = code if error_type == 'validation' else code[:500]
                
                summarizer.forward(
                    error_message=error_message,
                    error_type=error_type,
                    code_snippet=code_to_analyze
                )
                print("=" * 60 + "\n")
        except Exception as e:
            print(f"Failed: {e}")
        
        return ""
    
    def _truncate_output(self, output: str, max_length: int = 2000) -> str:
        """Truncate output if it's too long and add truncation notice
        
        Note: Increased max_length to 2000 to accommodate error reminders
        """
        if len(output) > max_length:
            reminder_marker = "[Error message reminder]"
            if reminder_marker in output:
                parts = output.split(reminder_marker, 1)
                error_part = parts[0]
                reminder_part = reminder_marker + parts[1] if len(parts) > 1 else ""
                
                if len(error_part) > max_length - len(reminder_part):
                    error_part = error_part[:max_length - len(reminder_part) - 50] + "\n\n[Error message truncated]"
                
                return error_part + reminder_part
            else:
                return output[:max_length] + "\n\n[Output truncated]"
        return output

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
        Execute C++ code compilation and execution
        
        Args:
            code: C++ source code
            
        Returns:
            Compilation execution result or error message
        """
        code = self._fix_broken_char_literals(code)
        
        payload = {
            "code": code.strip(),
            "language": "cpp",
            "run_timeout": self.run_timeout,
            "compile_timeout": self.compile_timeout,
            "files": {}
        }
        
        try:
            response = self._post_to_sandbox_with_retry(payload)
            if response.status_code == 200:
                result = response.json()
                
                # Parse sandbox response format
                if result.get('status') == 'Success':
                    compile_result = result.get('compile_result', {})
                    run_result = result.get('run_result', {})
                    
                    # Check compilation status
                    if compile_result.get('status') != 'Finished' or compile_result.get('return_code') != 0:
                        compile_stderr = compile_result.get('stderr', '')
                        error_msg = f"Error: Compilation failed:\n{compile_stderr}"
                        
                        reminder = self._analyze_error(compile_stderr, "compilation", code)
                        
                        return self._truncate_output(error_msg + reminder)
                    
                    # Check execution status
                    if run_result.get('status') != 'Finished':
                        error_msg = f"Error: Execution failed: {run_result.get('status')}"
                        
                        reminder = self._analyze_error(error_msg, "runtime", code)
                        
                        return self._truncate_output(error_msg + reminder)
                    
                    # Return execution output
                    stdout = run_result.get('stdout', '')
                    stderr = run_result.get('stderr', '')
                    return_code = run_result.get('return_code', 0)
                    
                    # For execution task, return raw output
                    output = stdout
                    if stderr:
                        output += f"\nStderr: {stderr}"
                    if return_code != 0:
                        output += f"\nExit code: {return_code}"
                    
                    final_output = output if output else "Program executed successfully with no output"
                    return self._truncate_output(final_output)
                else:
                    # Get detailed error information
                    error_details = []
                    compile_stderr = None
                    run_stderr = None
                    
                    if 'message' in result and result['message']:
                        error_details.append(f"Message: {result['message']}")
                    if 'compile_result' in result:
                        compile_result = result['compile_result']
                        if compile_result and compile_result.get('stderr'):
                            compile_stderr = compile_result['stderr']
                            error_details.append(f"Compile stderr: {compile_stderr}")
                        if compile_result and compile_result.get('stdout'):
                            error_details.append(f"Compile stdout: {compile_result['stdout']}")
                    if 'run_result' in result:
                        run_result = result['run_result']
                        if run_result and run_result.get('stderr'):
                            run_stderr = run_result['stderr']
                            error_details.append(f"Run stderr: {run_stderr}")
                        if run_result and run_result.get('stdout'):
                            error_details.append(f"Run stdout: {run_result['stdout']}")
                    
                    if error_details:
                        error_msg = f"Error: Sandbox error:\n" + "\n".join(error_details)
                    else:
                        error_msg = f"Error: Sandbox error: {result}"
                    
                    reminder = ""
                    if compile_stderr:
                        reminder = self._analyze_error(compile_stderr, "compilation", code)
                    elif run_stderr:
                        reminder = self._analyze_error(run_stderr, "runtime", code)
                    
                    return self._truncate_output(error_msg + reminder)
            else:
                error_msg = f"Error: HTTP Error: {response.status_code} - {response.text}"
                return self._truncate_output(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Request error: {str(e)}"
            return self._truncate_output(error_msg)


# Convenience function for creating tool instance
def create_cpp_interpreter_tool(
    sandbox_url,
    run_timeout: int = 60,
    compile_timeout: int = 60,
    enable_error_analysis: bool = True,
    model_id: str = None,
    api_base: str = None,
    api_key: str = None
) -> CppInterpreterTool:
    """
    Create a C++ interpreter tool instance
    
    Args:
        sandbox_url: The URL address of the sandbox service
        run_timeout: Timeout for code execution in seconds (default: 30)
        compile_timeout: Timeout for compilation in seconds (default: 60)
        enable_error_analysis: Whether to enable automatic error analysis
        model_id: LLM model ID for error analysis
        api_base: API endpoint for error analysis
        api_key: API key for error analysis
        
    Returns:
        CppInterpreterTool instance
    """
    return CppInterpreterTool(
        sandbox_url=sandbox_url,
        run_timeout=run_timeout,
        compile_timeout=compile_timeout,
        enable_error_analysis=enable_error_analysis,
        model_id=model_id,
        api_base=api_base,
        api_key=api_key
    )


# Example usage
if __name__ == "__main__":
    # Create tool instance
    cpp_tool = create_cpp_interpreter_tool()
    
    # Test with simple C++ code
    test_code = """
#include <iostream>
int a;
int main() {
		std::cerr << "This is an error message." << std::endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
    
    # Execute code
    result = cpp_tool.forward(test_code)
    print("Execution result:")
    print(result)
