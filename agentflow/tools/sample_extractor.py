from smolagents import Tool, LiteLLMModel, CodeAgent
from agentflow.models import TokenTrackingModel
import os
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional


class SampleExtractorTool(Tool):

    
    name = "sample_extractor"
    description = """
This tool uses an LLM to extract sample input and sample output from the problem description.
It leverages a sub-agent to intelligently parse the problem and prepare test files for cpp_validation.
    """
    
    inputs = {
        "problem_text": {
            "type": "string",
            "description": "The problem description text containing sample inputs and outputs."
        },
        "conversation_id": {
            "type": "string",
            "description": "The conversation ID for this session. If provided, will use this ID to save test files.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        base_model = LiteLLMModel(
            model_id="deepseek-chat", 
            api_base="https://api.deepseek.com/v1",  
            api_key=os.getenv("DEEPSEEK_API_KEY") 
        )
        model = TokenTrackingModel(base_model, source="sample_extractor")
        
        self.agent = CodeAgent(
            tools=[], 
            model=model,
            max_steps=1,  
            verbosity_level=0,  
            add_base_tools=False,
        )
        
        self._current_session_id = None

    def set_session_id(self, session_id: str):
        self._current_session_id = session_id

    def get_session_id(self) -> Optional[str]:
        return self._current_session_id

    def _generate_conversation_id(self, problem_text: str) -> str:
        timestamp = str(int(time.time() * 1000))
        problem_hash = hashlib.md5(problem_text.encode()).hexdigest()[:8]
        return f"auto_{timestamp}_{problem_hash}"

    def _extract_samples_with_llm(self, problem_text: str) -> Dict:
        
        system_prompt = """You are a professional algorithm problem parsing assistant. Your task is to extract the Sample Input and Sample Output from the given problem statement.

Please return the result strictly in the following JSON format, and do not add any other explanatory text:

{
"samples": [
{
"input": "Input of the first sample test case",
"output": "Output of the first sample test case"
},
{
"input": "Input of the second sample test case",
"output": "Output of the second sample test case"
}
]
}

Requirements:

1. Carefully analyze the Sample Input and Sample Output sections in the problem statement
2. Correctly extract all sample test cases
3. Preserve the original formatting of input and output, including newlines and spaces
4. If there are multiple sample test cases, extract all of them
5. Ensure each input-output pair is matched correctly
6. If the problem contains multiple test cases, the first line is often the number of test cases and must be included in the input
7. Return JSON only; do not add `json` fences or any other text

Note: Some samples may include escape sequences (e.g., \n). Please convert them into actual newline characters.
"""

        user_prompt = f"""Extract Sample Input and Sample Output from the following problem description:

{problem_text}"""

        try:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.agent.run(prompt)
            
            response_text = str(response).strip()
            
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
                else:
                    return {"error": f"Failed to parse LLM response: {response_text}"}
                    
        except Exception as e:
            return {"error": f"LLM extraction failed: {str(e)}"}

    def _save_test_files(self, samples: List[Dict], conversation_id: str) -> str:

        if not samples:
            return "No samples found to save"
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_data_dir = os.path.join(project_root, "tools", "cpp", "test_data", conversation_id)
        
        os.makedirs(test_data_dir, exist_ok=True)
        
        input_path = os.path.join(test_data_dir, "input")
        output_path = os.path.join(test_data_dir, "output")
        
        try:
            if len(samples) == 1:
                sample = samples[0]
                input_content = sample.get("input", "")
                output_content = sample.get("output", "")
                
                with open(input_path, 'w', encoding='utf-8') as f:
                    f.write(input_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)
            
            else:
                combined_input = []
                combined_output = []
                
                first_sample = samples[0]
                first_input = first_sample.get("input", "").strip()
                
                if first_input:
                    first_line = first_input.split('\n')[0].strip()
                    
                    if first_line.isdigit() and int(first_line) == len(samples):
                        combined_input = first_input.split('\n')
                        
                        for i in range(1, len(samples)):
                            sample = samples[i]
                            input_content = sample.get("input", "").strip()
                            if input_content:
                                lines = input_content.split('\n')
                                if lines and lines[0].strip().isdigit():
                                    lines = lines[1:]
                                combined_input.extend(lines)
                    else:
                        combined_input.append(str(len(samples)))
                        
                        for sample in samples:
                            input_content = sample.get("input", "").strip()
                            if input_content:
                                lines = input_content.split('\n')
                                if lines and lines[0].strip() == '1':
                                    lines = lines[1:]
                                combined_input.extend(lines)
                
                for sample in samples:
                    output_content = sample.get("output", "").strip()
                    if output_content:
                        combined_output.append(output_content)
                
                with open(input_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(combined_input))
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(combined_output))
            
            return f"Successfully saved {len(samples)} samples to test_data/{conversation_id}/"
        
        except Exception as e:
            return f"Error saving test files: {str(e)}"

    def forward(self, problem_text: str, conversation_id: Optional[str] = None) -> str:

        if not problem_text:
            return "Error: Empty problem text provided"
        
        if not conversation_id:
            conversation_id = self.get_session_id()
            
            if not conversation_id:
                try:
                    from agentflow.tools.session_manager import get_current_session_id
                    conversation_id = get_current_session_id()
                except ImportError:
                    pass
            
            if not conversation_id:
                conversation_id = self._generate_conversation_id(problem_text)
        
        self.set_session_id(conversation_id)
        
        extraction_result = self._extract_samples_with_llm(problem_text)
        
        if "error" in extraction_result:
            return f"Error: {extraction_result['error']}"
        
        samples = extraction_result.get("samples", [])
        
        if not samples:
            return "No sample data found in the problem text."
        
        result_lines = [f"Found {len(samples)} sample(s) using LLM extraction:"]
        
        for i, sample in enumerate(samples, 1):
            input_content = sample.get("input", "")
            output_content = sample.get("output", "")
            
            result_lines.append(f"\nSample {i}:")
            result_lines.append("Input:")
            result_lines.append(input_content if input_content else "(empty)")
            result_lines.append("Output:")
            result_lines.append(output_content if output_content else "(empty)")
        
        save_result = self._save_test_files(samples, conversation_id)
        result_lines.append(f"\n{save_result}")
        result_lines.append(f"Conversation ID: {conversation_id}")
        
        return '\n'.join(result_lines)


def create_sample_extractor_tool() -> SampleExtractorTool:

    return SampleExtractorTool()


if __name__ == "__main__":
    from agentflow.tools.session_manager import set_current_session_id, generate_session_id
    
    extractor = create_sample_extractor_tool()
    
    test_text = """
**Problem D. Destruction of the Dandelion Fields**

*Time limit per test:* $2$ seconds  
*Memory limit per test:* $256$ megabytes

Farmer John has a lawnmower, initially turned **off**. He also has $ n $ fields, where the $ i $-th field contains $ a_i $ dandelions. He will visit **all** the fields **exactly once**, in any order he chooses.

The lawnmower behaves as follows:
- **Before visiting a field**, it checks whether the number of dandelions in that field is **odd** or **even**.
  - If the number is **odd**, the lawnmower **toggles its state**:
    - If it was **off**, it turns **on**.
    - If it was **on**, it turns **off**.
- **After possibly toggling**, if the lawnmower is **on**, it cuts **all** dandelions in that field.
- If the lawnmower is **off**, no dandelions are cut (FJ just visits the field).

Determine the **maximum total number of dandelions** Farmer John can cut by choosing an optimal visiting order.

---

### Input

The first line contains an integer $ t $ ($ 1 \\leq t \\leq 10^4 $) — the number of test cases.

Each test case is described as follows:
- The first line contains an integer $ n $ ($ 1 \\leq n \\leq 2 \\cdot 10^5 $) — the number of fields.
- The second line contains $ n $ space-separated integers $ a_1, a_2, \\dots, a_n $ ($ 1 \\leq a_i \\leq 10^9 $) — the number of dandelions in each field.

It is guaranteed that the **sum of $ n $** over all test cases does not exceed $ 2 \\cdot 10^5 $.

---

### Output

For each test case, output a single integer: the **maximum number of dandelions** FJ can cut when visiting the fields in an optimal order.

---

### Examples

**Input:**
```
3
3
2 4 6
4
4 2 1 6
4
1000000000 999999999 1000000000 999999999
```

**Output:**
```
0
13
2999999999
```
"""
    
    session_id = generate_session_id(test_text)
    set_current_session_id(session_id)
    extractor.set_session_id(session_id)
    
    
    result = extractor.forward(test_text)
    print(result)