from smolagents import Tool
import requests
import base64
import os
import re
from typing import Optional


class CppCompilerTool(Tool):
    """
    A tool for compiling and executing C++ code using a local sandbox environment.
    Can compile C++ code and return execution results, also supports automated test validation.
    """
    
    name = "cpp_compiler"
    description = """
This tool compiles and executes C++ code in a local sandbox.
It takes C++ code as input, compiles it, and returns the program’s output.
If compilation fails, it returns the compilation error message.
    """
    
    inputs = {
        "code": {
            "type": "string", 
            "description": "The C++ code to compile and execute."
        },
        "task": {
            "type": "string",
            "description": "The task type: 'execution' for direct compilation and execution, 'validate' for automatic input injection and output validation using test cases.",
            "default": "validate",
            "nullable": True
        },
        "conversation_id": {
            "type": "string",
            "description": "The conversation ID, required for 'validate' task to locate the input/output files in test_data/{conversation_id}/.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, sandbox_url, **kwargs):
        """
        Initialize the C++ compiler tool
        
        Args:
            sandbox_url: The URL address of the sandbox service
        """
        super().__init__(**kwargs)
        self.sandbox_url = sandbox_url

    def _post_to_sandbox_with_retry(self, payload: dict, timeout: int = 30):

        try:
            response = requests.post(self.sandbox_url, json=payload, timeout=timeout)
            return response
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            print(f"[CppCompiler Sandbox Error] {repr(e)}")
            print("Sleeping for 30 seconds before retry...")
            from time import sleep
            sleep(30)
            return self._post_to_sandbox_with_retry(payload, timeout)
        except Exception as e:
            print(f"[CppCompiler Sandbox] Fatal error (non-retryable): {repr(e)}")
            raise e

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
    
    def _truncate_output(self, output: str, max_length: int = 500) -> str:
        """Truncate output if it's too long and add truncation notice"""
        if len(output) > max_length:
            return output[:max_length] + "\n\nThe output is too long and has been truncated."
        return output
    
    def _compare_outputs(self, actual_output: str, expected_output: str, input_data: str) -> str:
        """Compare actual output with expected output and return comparison result"""
        # Strip whitespace for comparison
        actual_clean = actual_output.strip()
        expected_clean = expected_output.strip()
        
        if actual_clean == expected_clean:
            return "Test PASSED: Output matches expected result"
        else:
            result = f"""Test FAILED: Output mismatch
            
Input Data:
{input_data}

Expected Output:
{expected_clean}

Actual Output:
{actual_clean}"""
            return result

    def forward(self, code: str, task: str = "execution", conversation_id: Optional[str] = None) -> str:
        """
        Execute C++ code compilation and execution
        
        Args:
            code: C++ source code
            task: Task type, 'execution' or 'validate'
            conversation_id: Conversation ID, used for validate mode
            
        Returns:
            Compilation execution result or error message
        """
        if task == "execution":
            # Direct execution mode
            payload = {
                "code": code.strip(),
                "language": "cpp",
                "files": {}
            }
        elif task == "validate":
            # Validation mode
            if not conversation_id:
                return "Error: conversation_id is required for validation task"
            
            # Read input and expected output files
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            test_data_dir = os.path.join(project_root, "tools", "cpp", "test_data", conversation_id)
            input_path = os.path.join(test_data_dir, "input")
            output_path = os.path.join(test_data_dir, "output")
            
            if not os.path.exists(test_data_dir):
                os.makedirs(test_data_dir, exist_ok=True)
            
            try:
                # Read input file
                with open(input_path, 'rb') as f:
                    input_content = f.read()
                input_base64 = base64.b64encode(input_content).decode('utf-8')
                input_text = input_content.decode('utf-8')
                
                # Read expected output file
                with open(output_path, 'r', encoding='utf-8') as f:
                    expected_output = f.read()
                
                # Inject freopen into code
                modified_code = self._inject_freopen(code)
                
                payload = {
                    "code": modified_code.strip(),
                    "language": "cpp",
                    "files": {"input": input_base64}
                }
            except FileNotFoundError as e:
                return f"Error: Required test files not found. Make sure both test_data/{conversation_id}/input and test_data/{conversation_id}/output exist."
            except Exception as e:
                return f"Error: reading test files: {str(e)}"
        else:
            return "Error: Invalid task type. Use 'execution' or 'validate'"
        
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
                        return self._truncate_output(error_msg)
                    
                    # Check execution status
                    if run_result.get('status') != 'Finished':
                        error_msg = f"Error: Execution failed: {run_result.get('status')}"
                        return self._truncate_output(error_msg)
                    
                    # Return execution output
                    stdout = run_result.get('stdout', '')
                    stderr = run_result.get('stderr', '')
                    return_code = run_result.get('return_code', 0)
                    
                    if task == "validate":
                        # Compare output with expected result
                        result_output = self._compare_outputs(stdout, expected_output, input_text)
                        return self._truncate_output(result_output)
                    else:
                        # For execution task, return raw output
                        output = stdout
                        if stderr:
                            output += f"\nStderr: {stderr}"
                        if return_code != 0:
                            output += f"\nExit code: {return_code}"
                        
                        final_output = output if output else "Program executed successfully with no output"
                        return self._truncate_output(final_output)
                else:
                    error_msg = f"Error: Sandbox error: {result.get('message', 'Unknown error')}"
                    return self._truncate_output(error_msg)
            else:
                error_msg = f"Error: HTTP Error: {response.status_code} - {response.text}"
                return self._truncate_output(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: Request error: {str(e)}"
            return self._truncate_output(error_msg)


# Convenience function for creating tool instance
def create_cpp_compiler_tool(sandbox_url) -> CppCompilerTool:
    """
    Create a C++ compiler tool instance
    
    Args:
        sandbox_url: The URL address of the sandbox service
        
    Returns:
        CppCompilerTool instance
    """
    return CppCompilerTool(sandbox_url=sandbox_url)


# Example usage
if __name__ == "__main__":
    # Create tool instance
    cpp_tool = create_cpp_compiler_tool()
    
    # Test with simple C++ code
    test_code = """
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
"""
    
    # Execute code
    result = cpp_tool.forward(test_code, task="execution")
    print("Execution result:")
    print(result)

