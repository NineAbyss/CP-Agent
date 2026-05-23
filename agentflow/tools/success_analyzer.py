
import os
import re
import json
from typing import Dict, Optional
from agentflow.tools.tracked_completion import tracked_completion


class SuccessAnalyzer:

    def __init__(
        self,
        model_id: str = None,
        api_base: str = None,
        api_key: str = None
    ):
        self.model_id = model_id or os.getenv("MODEL_ID", "deepseek-chat")
        self.api_base = api_base or os.getenv("API_BASE", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
    
    def _extract_error_summary(self, failure_result: str, tool_type: str = "validation") -> str:
        if tool_type == "interpreter":
            if "Compile error" in failure_result or "Compilation failed" in failure_result or "Sandbox error" in failure_result:
                return "Complie Error"
            elif "Runtime error" in failure_result or "RUNTIME ERROR" in failure_result:
                return "Runtime Error"
            else:
                return "Excution Error"
        else:
            if "FAILED" in failure_result or "Output mismatch" in failure_result or "WRONG ANSWER" in failure_result:
                return "Validation Failed (Output Mismatch)"
            elif "Compilation failed" in failure_result:
                return "Compilation Error"
            elif "RUNTIME ERROR" in failure_result:
                return "Runtime Error"
            else:
                return "Validation Error"
    
    def _format_intermediate_steps(self, steps: list) -> str:
        if not steps:
            return "No"
        
        lines = []
        for step in steps[:10]: 
            step_type = step.get("type", "unknown")
            if step_type == "agent_output":
                content = step.get("content", "")
                if "Thought:" in content:
                    thought_match = re.search(r'Thought:(.*?)(?:Action:|$)', content, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        lines.append(f"- Thought: {thought[:200]}")
                else:
                    lines.append(f"- Agent Output: {content[:200]}")
            elif step_type == "tool_call":
                tool_name = step.get("tool", "unknown")
                lines.append(f"- Tool Call: {tool_name}")
        
        return "\n".join(lines)
    
    def _create_analysis_prompt(self, context: Dict) -> str:
        failure = context.get("failure", {})
        success = context.get("success", {})
        intermediate_steps = context.get("intermediate_steps", [])
        
        failure_code = failure.get("code", "")  
        success_code = success.get("code", "")
        failure_result = failure.get("result", "")
        failure_tool_type = failure.get("tool_type", "validation")
        success_tool_type = success.get("tool_type", "validation")
        
        error_summary = self._extract_error_summary(failure_result, failure_tool_type)
        steps_text = self._format_intermediate_steps(intermediate_steps)
        
        prompt = f"""# Code Fix Experience Extraction Expert

## Your Task
As a code fix experience extraction expert, you need to analyze a debugging process that goes from failure to success, and distill reusable debugging experiences and fix strategies.

## Core Responsibilities
1. **Accuracy**: Ensure the extracted experience is accurate and verified
2. **Reusability**: The summarized experience should apply to similar problems
3. **Practicality**: Provide concrete and actionable fix methods
4. **Conciseness**: Express the key point in the most concise language

---

## Fix Process to Analyze

### First Validation (Failed)
**Error type**: {error_summary}

**Error message**:
{failure_result}

**Code snippet at failure**:
```cpp
{failure_code}

```

### Intermediate Improvement Steps
{steps_text}

### Second Validation (Succeeded)
**Code snippet after success**:
```cpp
{success_code}
```

---

Analysis Requirements
1. Identify the root cause

What is the fundamental cause of this error?

Is it an algorithmic logic issue, a boundary condition issue, or a syntax/library usage issue?

2. Extract the fix strategy

What specific method did the agent use to fix the problem?

What are the key steps of this fix?

Are there any debugging tips or ideas worth recording?

3. Summarize reusable experience

If a similar issue occurs in the future, what should be done?

What is the most critical insight from this fix process?

What general cautions or best practices can be drawn?
---

## Output Format Requirements

Return JSON with the following fields:

```json
{{
    "error_context": "When/where the error happened (what the code was doing when the error occurred)",
    "error_cause": "The specific cause of the error",
    "fix_method": "The concrete change made to fix it",
    "fix_result": "The effect after the fix",
    "key_insight": "A reusable key insight (one-sentence lesson learned)"
}}

```

Output Guidelines

1. error_context (required):

Describe the scenario and timing of the error

Examples: "While implementing divide-and-conquer optimized DP", "While handling large-scale array input", "While enumerating subsets with bit operations"

Length: 10-30 words

2. error_cause (required):

Explain the cause precisely

Example: "Incorrect computation order of recursive subproblems, so state dependencies were not satisfied"

Example: "Using <= in the loop bound caused an out-of-bounds access"

Length: 15-50 words
3. fix_method (required):

Describe the concrete fix

Example: "Move the recursive call solve(l, mid-1) to execute before computing dp[mid]"

Example: "Change the loop condition from i<=n to i<n"

Length: 15-50 words

4. fix_result (required):

Describe the outcome after modification

Example: "State dependencies are correct and outputs match"

Example: "Array access stays within bounds; no segmentation fault"

Length: 10-30 words

5. key_insight (required):

The reusable key lesson

This is the most important field: it should help others avoid the same error at a glance

Example: "In divide-and-conquer DP optimization, compute all dependent subproblems before computing the current state"

Example: "Array indices range from 0 to n-1, so loops should use strict < n"

Length: 20-60 words

---

## Example Outputs

```json
{{
    "error_context": "While implementing divide-and-conquer optimized DP",
    "error_cause": "Before computing dp[mid], the recursion for dp[l..mid-1] was not executed, breaking state dependencies",
    "fix_method": "Call solve(l, mid-1) before computing dp[mid]",
    "fix_result": "Dependencies are correct; all test cases pass",
    "key_insight": "Divide-and-conquer DP optimization requires all dependent subproblems to be computed before the current state"
}}
```

```json
{{
    "error_context": "While iterating over an array",
    "error_cause": "The loop used i<=n instead of i<n, causing an out-of-bounds access",
    "fix_method": "Change for(int i=0; i<=n; i++) to for(int i=0; i<n; i++)",
    "fix_result": "All array accesses are in range; the program runs normally",
    "key_insight": "Array indices are 0..n-1, so loop bounds should use strict less-than n"

}}
```

---

Please perform an in-depth analysis based on the information above and extract the most valuable reusable experience. Ensure you return valid JSON.
"""
        
        return prompt
    
    def analyze(self, context: Dict) -> Optional[Dict]:

        if not context or not context.get("failure") or not context.get("success"):
            return None
        
        print("[SuccessAnalyzer]")
        
        try:
            prompt = self._create_analysis_prompt(context)
            
            response = tracked_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a professional code-fix experience extraction expert, skilled at distilling reusable lessons and strategies from debugging processes."},
                    {"role": "user", "content": prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=500,  
                source="success_analyzer",
            )
            
            content = response.choices[0].message.content
            print(f"[SuccessAnalyzer] :\n{content}\n")
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                error_context = result.get("error_context", "")
                error_cause = result.get("error_cause", "")
                fix_method = result.get("fix_method", "")
                fix_result = result.get("fix_result", "")
                key_insight = result.get("key_insight", "")
                
                if not error_context and result.get("fix_summary"):
                    error_context = "While implementing the code"
                    error_cause = result.get("root_cause", result.get("original_error", "Unknown error"))
                    fix_method = result.get("fix_summary", "")
                    fix_result = "The problem has been fixed"
                    key_insight = result.get("key_insight", fix_method)
                
                if error_context and key_insight:
                    print(f"[SuccessAnalyzer]  Analysis successful:")
                    print(f"[SuccessAnalyzer]   Error Context: {error_context}")
                    print(f"[SuccessAnalyzer]   Error Cause: {error_cause}")
                    print(f"[SuccessAnalyzer]   Fix Method: {fix_method}")
                    print(f"[SuccessAnalyzer]   Fix Result: {fix_result}")
                    print(f"[SuccessAnalyzer]   Key Insight: {key_insight}")
                    
                    success_data = {
                        "type": "success_fix",
                        "error_context": error_context,
                        "error_cause": error_cause,
                        "fix_method": fix_method,
                        "fix_result": fix_result,
                        "key_insight": key_insight,
                        "fix_summary": f"{error_context}, {fix_method}",
                        "original_error": error_cause,
                        "improvement_steps": self._extract_key_steps(context),
                    }
                    
                    return success_data
                else:
                    print("[SuccessAnalyzer] ✗ The key fields returned by LLM are empty")
            else:
                print("[SuccessAnalyzer] ✗ Unable to extract JSON from the LLM return")
        
        except Exception as e:
            print(f"[SuccessAnalyzer] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _extract_key_steps(self, context: Dict) -> list:
        steps = context.get("intermediate_steps", [])
        key_steps = []
        
        for step in steps[:5]:  
            step_type = step.get("type", "")
            if step_type == "agent_output":
                content = step.get("content", "")
                if "Thought:" in content:
                    thought_match = re.search(r'Thought:(.*?)(?:Action:|$)', content, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()[:100]
                        key_steps.append(f"Thought: {thought}")
            elif step_type == "tool_call":
                tool_name = step.get("tool", "unknown")
                key_steps.append(f" {tool_name}")
        
        return key_steps


def create_success_analyzer(
    model_id: str = None,
    api_base: str = None,
    api_key: str = None
) -> SuccessAnalyzer:
    return SuccessAnalyzer(model_id, api_base, api_key)


