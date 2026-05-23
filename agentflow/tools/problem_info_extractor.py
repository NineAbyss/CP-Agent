from smolagents import Tool, LiteLLMModel
from agentflow.models import TokenTrackingModel
import os
import json
import time
import hashlib
import re
from typing import Dict, List, Tuple, Optional


class ProblemInfoExtractorTool(Tool):
    """
    Extracts complete problem information from the problem description, including time limit, memory limit, input/output format, and sample data.
    Uses regular expressions to parse basic information and LLM to extract sample data.
    """
    
    name = "problem_info_extractor"
    description = """
    This tool extracts comprehensive problem information from the problem description.
    It extracts time limit, memory limit, input/output format, and sample data.
    The time limit is automatically set as the global run_timeout for cpp_validation.
    """
    
    inputs = {
        "problem_text": {
            "type": "string",
            "description": "The problem description text containing all problem information."
        },
        "conversation_id": {
            "type": "string",
            "description": "The conversation ID for this session. If provided, will use this ID to save test files.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model_id: str = None, api_base: str = None, api_key: str = None, **kwargs):
        """Initialize problem info extraction tool"""
        super().__init__(**kwargs)
        
        # Use provided parameters or default config
        model_id = model_id or os.getenv("MODEL_ID", "deepseek-chat")
        api_base = api_base or os.getenv("API_BASE", "https://api.deepseek.com/v1")
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
        
        # Use create_model factory function to create the model (automatically handles openai/ prefix for custom APIs)
        self.model = create_model(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            track_tokens=True,
            token_source="problem_info_extractor"
        )
        
        # Store the current runtime session ID
        self._current_session_id = None

    def set_session_id(self, session_id: str):
        """Set current session ID"""
        self._current_session_id = session_id

    def get_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self._current_session_id

    def _generate_conversation_id(self, problem_text: str) -> str:
        """Generate conversation ID"""
        timestamp = str(int(time.time() * 1000))
        problem_hash = hashlib.md5(problem_text.encode()).hexdigest()[:8]
        return f"auto_{timestamp}_{problem_hash}"

    def _extract_basic_info_with_regex(self, problem_text: str) -> Dict:
        """Extract basic info using regular expressions"""
        info = {
            "time_limit": "10 seconds",
            "memory_limit": "256 megabytes", 
            "input_format": "Not specified",
            "output_format": "Not specified"
        }
        
        # Extract time limit
        time_patterns = [
            r'Time limit per test[:\s]*([^\\n]+)',
            r'Time limit[:\s]*([^\\n]+)',
            r'Time[:\s]*([^\\n]*(?:second|ms|minute)[^\\n]*)'
        ]
        
        for pattern in time_patterns:
            time_match = re.search(pattern, problem_text, re.IGNORECASE)
            if time_match:
                info["time_limit"] = time_match.group(1).strip()
                break
        
        # Extract memory limit
        memory_patterns = [
            r'Memory limit per test[:\s]*([^\\n]+)',
            r'Memory limit[:\s]*([^\\n]+)',
            r'Memory[:\s]*([^\\n]*(?:MB|megabyte|GB|gigabyte)[^\\n]*)'
        ]
        
        for pattern in memory_patterns:
            memory_match = re.search(pattern, problem_text, re.IGNORECASE)
            if memory_match:
                info["memory_limit"] = memory_match.group(1).strip()
                break
        
        # Extract input format
        input_section = re.search(r'### Input(.*?)(?=### Output|### Examples|---|\Z)', problem_text, re.DOTALL | re.IGNORECASE)
        if input_section:
            info["input_format"] = input_section.group(1).strip()
        
        # Extract output format  
        output_section = re.search(r'### Output(.*?)(?=### Examples|---|\Z)', problem_text, re.DOTALL | re.IGNORECASE)
        if output_section:
            info["output_format"] = output_section.group(1).strip()
        
        return info

    def _extract_samples_with_llm(self, problem_text: str) -> List[Dict]:
        """Extract sample data using LLM"""
        
        # First check if it's AtCoder-style independent samples
        # AtCoder samples should be extracted directly using regex to avoid LLM error merging
        if self._is_atcoder_style(problem_text):
            samples = self._extract_atcoder_samples(problem_text)
            if samples:
                return samples
        
        system_prompt = """You are a professional algorithm problem analysis assistant. Your task is to extract complete Sample Input and Sample Output from the given problem description.

Important rules:
1. Look for the **Input:** and **Output:** sections in the problem, or similar markers.
2. These sections usually contain complete test data within code blocks (```).
3. For Codeforces-style multi-testcase problems (where the first line of the Input section is the number of test cases t), extract all data as a single complete INPUT/OUTPUT.
4. For independent samples (where each Sample is an independent test case), output multiple INPUT/OUTPUT pairs separately.
5. You must extract the **complete** Input and Output, including all test data.
6. Do not add any explanatory notes; output only INPUT and OUTPUT.
7. **Important**: If the problem has independent sample markers like "Sample Input 1", "Sample Input 2", please output multiple INPUT/OUTPUT pairs separately, do not merge!

Output format (merged multi-testcases):
INPUT: [complete input content]
OUTPUT: [complete output content]

Output format (independent samples):
INPUT: [sample 1 input]
OUTPUT: [sample 1 output]
INPUT: [sample 2 input]
OUTPUT: [sample 2 output]
...

Example 1 (Codeforces multi-testcase style - merged):
If the problem contains:
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

Then you should output:
INPUT: 3
3
2 4 6
4
4 2 1 6
4
1000000000 999999999 1000000000 999999999
OUTPUT: 0
13
2999999999

Example 2 (AtCoder independent sample style - not merged):
If the problem contains:
Sample Input 1
Kyoto
Sample Output 1
KUPC
Sample Input 2
Tohoku
Sample Output 2
TUPC

Then you should output:
INPUT: Kyoto
OUTPUT: KUPC
INPUT: Tohoku
OUTPUT: TUPC

Note: Judge whether merging is needed based on the problem's sample format!"""

        user_prompt = f"""Please extract the complete Sample Input and Sample Output from the following problem description:

{problem_text}

Please judge based on the problem sample format:
- If it is Codeforces-style multi-testcase (the first line is the test count), output one pair of INPUT/OUTPUT
- If they are independent samples (Sample Input 1, Sample Input 2, etc.), output multiple pairs of INPUT/OUTPUT respectively."""

        try:
            # Directly use LLM model for sample extraction
            from smolagents.models import MessageRole
            
            messages = [
                {"role": MessageRole.SYSTEM, "content": system_prompt},
                {"role": MessageRole.USER, "content": user_prompt}
            ]
            
            # Call model
            response = self.model(messages)
            
            # Parse response - extract actual content
            if hasattr(response, 'content') and response.content:
                response_text = str(response.content).strip()
            else:
                response_text = str(response).strip()
            
            # Remove possible code block markers
            response_text = re.sub(r'```[a-z]*\n?', '', response_text)
            response_text = re.sub(r'```', '', response_text)
            
            samples = []
            
            # First attempt to match single INPUT/OUTPUT pair (expected format)
            input_match = re.search(r'INPUT:\s*(.*?)(?=OUTPUT:|$)', response_text, re.DOTALL)
            output_match = re.search(r'OUTPUT:\s*(.*?)(?=INPUT:|$)', response_text, re.DOTALL)
            
            if input_match and output_match:
                input_content = input_match.group(1).strip()
                output_content = output_match.group(1).strip()
                
                # Check for multiple INPUT/OUTPUT pairs (LLM didn't merge as requested)
                all_inputs = re.findall(r'INPUT:\s*(.*?)(?=OUTPUT:|INPUT:|$)', response_text, re.DOTALL)
                all_outputs = re.findall(r'OUTPUT:\s*(.*?)(?=INPUT:|OUTPUT:|$)', response_text, re.DOTALL)
                
                # Clean blank items
                all_inputs = [inp.strip() for inp in all_inputs if inp.strip()]
                all_outputs = [out.strip() for out in all_outputs if out.strip()]
                
                if len(all_inputs) > 1 and len(all_inputs) == len(all_outputs):
                    # LLM returned multiple INPUT/OUTPUT pairs, need to merge
                    # Save them as multiple independent samples
                    for inp, out in zip(all_inputs, all_outputs):
                        samples.append({
                            "input": inp,
                            "output": out
                        })
                else:
                    # Single complete sample
                    samples.append({
                        "input": input_content,
                        "output": output_content
                    })
            
            return samples
                    
        except Exception as e:
            # If LLM extraction fails, attempt regular expressions
            return self._extract_samples_with_regex(problem_text)

    def _is_atcoder_style(self, problem_text: str) -> bool:
        """
        Detect if the problem is AtCoder style (independent Sample Input 1, Sample Input 2, etc.)
        
        AtCoder style characteristics:
        - Contains tags like "Sample Input 1", "Sample Output 1"
        - Each sample is independent, not multi-testcase format
        """
        # Detect if there are multiple independent Sample Input/Output
        sample_input_count = len(re.findall(r'Sample Input \d+', problem_text, re.IGNORECASE))
        sample_output_count = len(re.findall(r'Sample Output \d+', problem_text, re.IGNORECASE))
        
        # If there are multiple Sample Input/Output pairs, consider it AtCoder style
        return sample_input_count >= 2 and sample_output_count >= 2

    def _extract_atcoder_samples(self, problem_text: str) -> List[Dict]:
        """
        Extract AtCoder-style independent samples
        
        The sample format for AtCoder problems is usually:
        Sample Input 1
        <input content>
        
        Sample Output 1
        <output content>
        
        Sample Input 2
        ...
        """
        samples = []
        
        # Find all Sample Input N and corresponding Sample Output N
        # Matching pattern: Content after Sample Input N until Sample Output N or next Sample Input
        input_pattern = r'Sample Input (\d+)\s*\n(.*?)(?=Sample Output \d+|Sample Input \d+|\Z)'
        output_pattern = r'Sample Output (\d+)\s*\n(.*?)(?=Sample Input \d+|Sample Output \d+|\Z)'
        
        # Extract all inputs
        input_matches = re.findall(input_pattern, problem_text, re.DOTALL | re.IGNORECASE)
        # Extract all outputs  
        output_matches = re.findall(output_pattern, problem_text, re.DOTALL | re.IGNORECASE)
        
        # Build mapping from index to content
        inputs_dict = {}
        outputs_dict = {}
        
        for idx, content in input_matches:
            # Clean content: remove leading newline and trailing explanatory text
            content = content.strip()
            # If content contains multiple lines, there might be explanatory text at the end, attempt to keep only data part
            lines = content.split('\n')
            data_lines = self._filter_explanation_lines(lines)
            inputs_dict[idx] = '\n'.join(data_lines)
        
        for idx, content in output_matches:
            content = content.strip()
            lines = content.split('\n')
            data_lines = self._filter_explanation_lines(lines)
            outputs_dict[idx] = '\n'.join(data_lines)
        
        # Pair by index
        for idx in sorted(inputs_dict.keys(), key=int):
            if idx in outputs_dict:
                samples.append({
                    "input": inputs_dict[idx],
                    "output": outputs_dict[idx]
                })
        
        return samples

    def _is_explanation_line(self, line: str) -> bool:
        """
        Judge whether a line is explanatory text (rather than actual input/output data)
        
        Characteristics of explanatory text:
        1. Mathematical expression explanation starting with parentheses, e.g., (20+25)^2=2025.
        2. Contains a complete sentence (ending with a period and having multiple words)
        3. Contains a mathematical equation explanation, e.g., xxx^2=xxx or xxx+xxx=xxx
        4. Starts with a common explanatory word, such as "The", "Note", "Since", "Because", etc.
        5. Contains many English words
        """
        line = line.strip()
        if not line:
            return False
        
        # Characteristic 1: Mathematical expression starting with parentheses, e.g., (20+25)^2=2025.
        if re.match(r'^\([^)]+\)[^=]*=.*$', line):
            return True
        
        # Characteristic 2: Mathematical expression explanation containing ^Number=, e.g., xxx^2=xxx
        if re.search(r'\^\d+\s*=', line):
            return True
        
        # Characteristic 3: Ending with a period and containing multiple words (likely an explanatory sentence)
        if line.endswith('.') and len(re.findall(r'\b[a-zA-Z]+\b', line)) >= 2:
            return True
        
        # Characteristic 4: Starting with common explanatory words
        explanation_starters = [
            r'^The\s+', r'^This\s+', r'^Note\s+', r'^Since\s+', r'^Because\s+',
            r'^In\s+this\s+', r'^For\s+', r'^When\s+', r'^If\s+', r'^As\s+',
            r'^We\s+', r'^Here\s+', r'^It\s+', r'^There\s+'
        ]
        for pattern in explanation_starters:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Characteristic 5: More than 3 long words (4+ letters), likely explanatory text
        word_count = len(re.findall(r'\b[a-zA-Z]{4,}\b', line))
        if word_count >= 3:
            return True
        
        return False

    def _filter_explanation_lines(self, lines: List[str]) -> List[str]:
        """
        Filters explanatory text from input/output lines
        
        Strategy:
        - After encountering an empty line, check if the subsequent content is an explanation
        - Once an explanation line is encountered, stop collecting
        """
        data_lines = []
        saw_empty_line = False
        
        for line in lines:
            stripped = line.strip()
            
            # Track if empty line is encountered
            if not stripped:
                saw_empty_line = True
                continue
            
            # Check if it's an explanation line
            if self._is_explanation_line(stripped):
                # If explanation line is encountered after empty line, stop collecting
                if saw_empty_line:
                    break
                # If first line is explanation (rare), also stop
                if not data_lines:
                    break
                # Otherwise, this might be an explanation following data, stop collecting
                break
            
            # This is a data line
            data_lines.append(stripped)
            saw_empty_line = False
        
        return data_lines

    def _extract_samples_with_regex(self, problem_text: str) -> List[Dict]:
        """Extract sample data using regular expressions (fallback scheme)"""
        samples = []
        
        # First check if it's AtCoder-style independent samples
        if self._is_atcoder_style(problem_text):
            return self._extract_atcoder_samples(problem_text)
        
        # Find Examples section
        examples_section = re.search(r'### Examples(.*?)(?=---|\Z)', problem_text, re.DOTALL | re.IGNORECASE)
        if not examples_section:
            # Try other possible markers
            examples_section = re.search(r'(Examples?|Sample|Test Cases?)(.*?)(?=---|\Z)', problem_text, re.DOTALL | re.IGNORECASE)
        
        if examples_section:
            examples_text = examples_section.group()
            
            # Find all code blocks
            code_blocks = re.findall(r'```\s*(.*?)\s*```', examples_text, re.DOTALL)
            
            if len(code_blocks) >= 2:
                # Usually first is input, second is output
                input_text = code_blocks[0].strip()
                output_text = code_blocks[1].strip()
                
                samples.append({
                    "input": input_text,
                    "output": output_text
                })
            else:
                # Attempt to find **Input:** and **Output:** markers
                input_match = re.search(r'\*\*Input:\*\*\s*```\s*(.*?)\s*```', examples_text, re.DOTALL)
                output_match = re.search(r'\*\*Output:\*\*\s*```\s*(.*?)\s*```', examples_text, re.DOTALL)
                
                if input_match and output_match:
                    input_text = input_match.group(1).strip()
                    output_text = output_match.group(1).strip()
                    
                    samples.append({
                        "input": input_text,
                        "output": output_text
                    })
        
        return samples

    def _save_test_files(self, samples: List[Dict], conversation_id: str) -> str:
        """
        Saves the extracted samples as test files
        
        Args:
            samples: List of samples, each element is {"input": str, "output": str}
            conversation_id: Conversation ID
            
        Returns:
            Description of the saving result
        """
        if not samples:
            return "No samples found to save"
        
        # Unified global work directory priority
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            work_root = get_global_work_dir()
        except Exception:
            print("Failed to set global work_dir")
        
        if not work_root:
            # Fallback legacy logic
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            work_root = os.path.join(project_root, "tools", "cpp", "unit_test_data", conversation_id)
        os.makedirs(work_root, exist_ok=True)
        
        # Organize subdirectories under global work directory
        test_data_dir = os.path.join(work_root, "test_data")
        test_data_dir_unit = os.path.join(work_root, "unit_test_data")
        
        # Create directories
        os.makedirs(test_data_dir, exist_ok=True)
        os.makedirs(test_data_dir_unit, exist_ok=True)
        
        try:
            if len(samples) == 1:
                # Single sample: save directly
                sample = samples[0]
                input_content = sample.get("input", "")
                output_content = sample.get("output", "")
                
                # Save to test_data directory
                input_path = os.path.join(test_data_dir, "input")
                output_path = os.path.join(test_data_dir, "output")
                
                with open(input_path, 'w', encoding='utf-8') as f:
                    f.write(input_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output_content)

                # Save to unit_test_data directory
                input_path_unit = os.path.join(test_data_dir_unit, "sample_test_input")
                output_path_unit = os.path.join(test_data_dir_unit, "sample_test_output")

                with open(input_path_unit, 'w', encoding='utf-8') as f:
                    f.write(input_content)

                with open(output_path_unit, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                    
                return f"Successfully saved 1 sample to {test_data_dir}/ and {test_data_dir_unit}/."
            else:
                # Multiple samples - LLM returns multiple INPUT/OUTPUT pairs representing independent samples
                # Note: Codeforces multi-testcase format is already merged into single sample result in LLM extraction phase
                # So if it reaches here with multiple samples, it must be independent samples (AtCoder style)
                # 
                # [BUG FIX] Removed the previous is_multi_testcase auto-detection logic
                # This logic had false positives: when the first line of an independent sample input
                # happened to be a number equal to the number of samples, it would be 
                # incorrectly judged as a multi-testcase format and merged, leading to test case corruption.
                # For example: 2 independent samples, where the first line of each input is "2", it would be incorrectly merged.
                
                # Save directly as independent samples input_1, input_2, output_1, output_2 etc.
                for i, sample in enumerate(samples, 1):
                    input_content = sample.get("input", "")
                    output_content = sample.get("output", "")
                    
                    input_path = os.path.join(test_data_dir, f"input_{i}")
                    output_path = os.path.join(test_data_dir, f"output_{i}")
                    
                    with open(input_path, 'w', encoding='utf-8') as f:
                        f.write(input_content)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output_content)

                    # Save to unit_test_data directory
                    input_path_unit = os.path.join(test_data_dir_unit, f"sample_test_input_{i}")
                    output_path_unit = os.path.join(test_data_dir_unit, f"sample_test_output_{i}")

                    with open(input_path_unit, 'w', encoding='utf-8') as f:
                        f.write(input_content)

                    with open(output_path_unit, 'w', encoding='utf-8') as f:
                        f.write(output_content)                            
                
                return f"Successfully saved {len(samples)} independent samples as separate files to test_data/{conversation_id}/ and unit_test_data/{conversation_id}/."
        
        except Exception as e:
            return f"Error saving test files: {str(e)}"

    def forward(self, problem_text: str, conversation_id: Optional[str] = None) -> str:
        """
        Extract complete problem information from the problem description
        
        Args:
            problem_text: Problem description text
            conversation_id: Conversation ID, will be auto-generated if not provided
            
        Returns:
            Description of the extraction result
        """
        if not problem_text:
            return "Error: Empty problem text provided"
        
        # If conversation_id not provided, attempt to get from multiple sources
        if not conversation_id:
            # First attempt to get from within the tool
            conversation_id = self.get_session_id()
            
            # If still not found, attempt to get from main agent
            if not conversation_id:
                try:
                    from agentflow.tools.session_manager import get_current_session_id
                    conversation_id = get_current_session_id()
                except ImportError:
                    pass
            
            # Finally, auto-generate one
            if not conversation_id:
                conversation_id = self._generate_conversation_id(problem_text)
        
        # Update tool's session ID
        self.set_session_id(conversation_id)
        
        # Use regex to extract basic information
        basic_info = self._extract_basic_info_with_regex(problem_text)
        
        # Extract sample data using LLM
        samples = self._extract_samples_with_llm(problem_text)
        
        # Get extracted information
        time_limit = basic_info.get("time_limit", "10 seconds")
        memory_limit = basic_info.get("memory_limit", "256 megabytes")
        input_format = basic_info.get("input_format", "Not specified")
        output_format = basic_info.get("output_format", "Not specified")
        
        # Parse time limit and memory limit and set global config
        try:
            from agentflow.tools.session_manager import (
                parse_time_limit, set_run_timeout,
                parse_memory_limit, set_memory_limit
            )
            time_limit_seconds = parse_time_limit(time_limit)
            set_run_timeout(time_limit_seconds)
            timeout_update_result = f"Set global run_timeout to {time_limit_seconds} seconds"
            
            # Parse memory limit and set global memory limit
            memory_limit_mb = parse_memory_limit(memory_limit)
            set_memory_limit(memory_limit_mb)
            memory_update_result = f"Set global memory_limit to {memory_limit_mb} MB"
        except ImportError:
            time_limit_seconds = 30
            timeout_update_result = "Failed to set global run_timeout (session_manager not available)"
            memory_limit_mb = 256
            memory_update_result = "Failed to set global memory_limit (session_manager not available)"
        
        # Build result description
        result_lines = ["=== Problem Information Extraction ==="]
        result_lines.append(f"Time limit per test: {time_limit} (converted to {time_limit_seconds} seconds for run_timeout)")
        result_lines.append(f"Memory limit per test: {memory_limit} (converted to {memory_limit_mb} MB for memory_limit)")
        result_lines.append(f"Input format: {input_format}")
        result_lines.append(f"Output format: {output_format}")
        result_lines.append("")
        
        if samples:
            result_lines.append(f"Found {len(samples)} sample(s):")
            
            for i, sample in enumerate(samples, 1):
                input_content = sample.get("input", "")
                output_content = sample.get("output", "")
                
                result_lines.append(f"\nSample {i}:")
                result_lines.append("Input:")
                result_lines.append(input_content if input_content else "(empty)")
                result_lines.append("Output:")
                result_lines.append(output_content if output_content else "(empty)")
            
            # Save test files
            save_result = self._save_test_files(samples, conversation_id)
            result_lines.append(f"\n{save_result}")
        else:
            result_lines.append("No sample data found in the problem text.")
        
        result_lines.append(f"\nConversation ID: {conversation_id}")
        result_lines.append(f"Timeout update: {timeout_update_result}")
        result_lines.append(f"Memory update: {memory_update_result}")
        
        return '\n'.join(result_lines)


# Helper functions
def create_problem_info_extractor_tool() -> ProblemInfoExtractorTool:
    """
    Creates an instance of the problem info extraction tool
    
    Returns:
        ProblemInfoExtractorTool instance
    """
    return ProblemInfoExtractorTool()


# Example usage
if __name__ == "__main__":
    # When running standalone, setup session management
    try:
        from agentflow.tools.session_manager import set_current_session_id, generate_session_id
        
        # Create tool instance
        extractor = create_problem_info_extractor_tool()
        
        # Test sample text
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
        
        # Generate and set session ID
        session_id = generate_session_id(test_text)
        set_current_session_id(session_id)
        extractor.set_session_id(session_id)
        
        print(f"Standalone run mode, session ID: {session_id}")
        
        # Extract problem information
        result = extractor.forward(test_text)
        print("Extraction Result:")
        print(result)
        
    except ImportError:
        print("Standalone run mode requires session_manager module")
        print("Please run this tool in the complete project environment")
