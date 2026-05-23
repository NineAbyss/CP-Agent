"""
Error Summarizer Tool - Use LLM to analyze and summarize code errors
"""
from smolagents import Tool
from typing import Optional
import os
import json
import re
from agentflow.tools.tracked_completion import tracked_completion


class ErrorSummarizerTool(Tool):

    
    name = "error_summarizer"
    description = """
This tool analyzes compilation or runtime errors and provides a structured error summary.
It uses an LLM to interpret the error messages and generate actionable remediation suggestions.
The analysis results are automatically stored in error memory for future reference.
    """
    
    inputs = {
        "error_message": {
            "type": "string",
            "description": "The error message from cpp_validate or cpp_interpreter"
        },
        "code_snippet": {
            "type": "string",
            "description": "The relevant code snippet that caused the error (optional)",
            "nullable": True
        },
        "error_type": {
            "type": "string",
            "description": "Type of error: 'compilation', 'runtime', or 'validation'",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(
        self,
        model_id: str = None,
        api_base: str = None,
        api_key: str = None,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.model_id = model_id or os.getenv("MODEL_ID", "deepseek-chat")
        self.api_base = api_base or os.getenv("API_BASE", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
    
    def _create_analysis_prompt(self, error_message: str, code_snippet: Optional[str], error_type: str) -> str:
        
        if error_type == "validation":
            return self._create_validation_prompt(error_message, code_snippet)
        
        prompt = f"""You are a C++ programming expert. Please analyze the following {'compilation' if error_type == 'compilation' else 'runtime'} error and provide a structured analysis.

Error type: {error_type}

Error message:
{error_message}
"""

if code_snippet:
    prompt += f"""
Relevant code:
```cpp
{code_snippet}

```
"""
        
        prompt += """
Please return your analysis in JSON format with only one field:
{
"reminder": "A one-sentence reminder in the format: When doing xxxx, you must not xxxx / you need to xxxx"
}

Requirements:

1. The reminder must be concise and clear, in a single sentence

2. The reminder should be actionable and specific

3. Avoid overly technical jargon; use plain and easy-to-understand language

4. Example: "When using cout, you need to include the <iostream> header"

Make sure to return valid JSON.
"""
        return prompt
    
    def _create_validation_prompt(self, error_message: str, code_snippet: Optional[str]) -> str:
        
        problem_description = self._get_problem_description()
        
        input_match = re.search(r'Input:\s*(.+?)(?:\n|Expected:)', error_message, re.DOTALL)
        expected_match = re.search(r'Expected:\s*(.+?)(?:\n|Actual:)', error_message, re.DOTALL)
        actual_match = re.search(r'Actual:\s*(.+?)$', error_message, re.DOTALL)
        
        input_data = input_match.group(1).strip() if input_match else "UNK"
        expected_output = expected_match.group(1).strip() if expected_match else "UNK"
        actual_output = actual_match.group(1).strip() if actual_match else "UNK"
        
        prompt = f"""You are a competitive programming expert. Please analyze why the code’s output does not match on this test case.

        """

        if problem_description:
            prompt += f"""[Problem Description]
        {problem_description}...

        """

        prompt += f"""[Test Case Information]
        Input:
        {input_data}

        Expected output:
        {expected_output}

        Actual output:
        {actual_output}

        """

        if code_snippet:
            prompt += f"""[Code Snippet]
        ```cpp
        {code_snippet}


        """

        prompt += f"""[In-Depth Analysis Requirements]

Please analyze carefully following these steps:

Step 1: Understand the test case

What scenario is this test case targeting? (boundary case, normal case, special case?)

What characteristics does the input data have?

Step 2: Trace code execution

For this input, what is the code’s execution flow?

What key computations or decisions does the code perform?

Step 3: Identify the failure cause

The expected output is "{expected_output}", but the actual output is "{actual_output}"

Why does the code produce this incorrect output?

What did the code miss or fail to handle for this input?

Step 4: Generalize the error pattern

What type of situation does this error indicate the code did not consider?

For example: boundary conditions, special inputs, algorithmic logic flaws, incorrect data structure usage, etc.

Please return a one-sentence reminder in JSON format:
{{
"reminder": "When handling [specific scenario], you need to pay attention to [specific issue], otherwise it may cause [specific consequence]."
}}

Reminder requirements:

Must be specific to this test failure cause

Concise and clear, one sentence

Actionable and helpful for fixing the code

Example: "When handling boundary cases, check whether the array index goes out of bounds."

Use professional algorithm terms such as “array”, “element”, “shortest path”, etc., rather than story-like terms such as “cows”, “dandelions”, “farm”, etc.

Make sure to return valid JSON.
"""

        return prompt
    
    def _get_problem_description(self) -> Optional[str]:
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            work_dir = get_global_work_dir()
            if work_dir:
                question_file = os.path.join(work_dir, "unit_test_data", "original_question.txt")
                if os.path.exists(question_file):
                    with open(question_file, 'r', encoding='utf-8') as f:
                        return f.read()
        except Exception:
            pass
        return None
    
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> dict:
        response = tracked_completion(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are a professional C++ programming expert, skilled at analyzing and fixing programming errors. Please provide a concise, one-sentence reminder."},
                {"role": "user", "content": prompt}
            ],
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=0.2,  
            max_tokens=max_tokens,
            source="error_summarizer",
        )
        
        content = response.choices[0].message.content
        print(content)
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {
                "reminder": content.strip()
            }
    
    def forward(self, error_message: str, error_type: str = "runtime", code_snippet: Optional[str] = None) -> str:

        prompt = self._create_analysis_prompt(error_message, code_snippet, error_type)
        

        max_tokens = 600 if error_type == 'validation' else 200
        
        analysis = self._call_llm(prompt, max_tokens=max_tokens)
        
        reminder = analysis.get("reminder", "An error occurred, please check the code carefully")
        
        result = reminder
        
        return result


def create_error_summarizer_tool(
    model_id: str = None,
    api_base: str = None,
    api_key: str = None
) -> ErrorSummarizerTool:

    return ErrorSummarizerTool(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key
    )


# Example usage
if __name__ == "__main__":
    tool = create_error_summarizer_tool()
    
    # Test compilation error
    test_error = """
main.cpp: In function 'int main()':
main.cpp:5:5: error: 'cout' was not declared in this scope
    5 |     cout << "Hello" << endl;
      |     ^~~~
main.cpp:2:1: note: 'std::cout' is defined in header '<iostream>'; did you forget to '#include <iostream>'?
    1 | #include <string>
  +++ |+#include <iostream>
    2 | using namespace std;
"""
    
    result = tool.forward(
        error_message=test_error,
        error_type="compilation",
        code_snippet="#include <string>\nusing namespace std;\nint main() {\n    cout << \"Hello\" << endl;\n    return 0;\n}"
    )
    
    print(result)

