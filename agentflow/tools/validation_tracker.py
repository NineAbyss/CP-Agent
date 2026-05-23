from typing import List, Dict, Optional
from datetime import datetime


class ValidationRecord:
    def __init__(self, step_idx: int, code: str, result: str, is_success: bool, tool_type: str = "validation", generated_stats: Optional[Dict] = None):
        self.step_idx = step_idx
        self.code = code
        self.result = result
        self.is_success = is_success
        self.tool_type = tool_type  
        self.generated_stats = generated_stats
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "step_idx": self.step_idx,
            "code": self.code,
            "result": self.result,
            "is_success": self.is_success,
            "tool_type": self.tool_type,
            "timestamp": self.timestamp
        }


class ValidationTracker:

    
    def __init__(self):
        self.records: List[ValidationRecord] = []
        self._last_failure_idx: Optional[int] = None
    
    def record_validation(self, step_idx: int, code: str, result: str, is_success: bool, tool_type: str = "validation", generated_stats: Optional[Dict] = None):

        record = ValidationRecord(step_idx, code, result, is_success, tool_type, generated_stats)
        self.records.append(record)
        
        tool_name = "Validation" if tool_type == "validation" else "Compile/Run"
        print(f"[ValidationTracker] Record {tool_name}: step={step_idx}, success={is_success}")
        
        if not is_success:
            self._last_failure_idx = len(self.records) - 1
    
    def check_success_after_failure(self) -> bool:

        if len(self.records) < 2:
            return False
        
        if not self.records[-1].is_success:
            return False
        
        if self._last_failure_idx is None:
            return False
        
        if self._last_failure_idx >= len(self.records) - 1:
            return False

        return True
    
    def get_last_failure_and_success(self) -> tuple[Optional[ValidationRecord], Optional[ValidationRecord]]:

        if not self.check_success_after_failure():
            return None, None
        
        failure_record = self.records[self._last_failure_idx]
        success_record = self.records[-1]
        
        return failure_record, success_record
    
    def extract_improvement_context(self, agent) -> Dict:

        failure_record, success_record = self.get_last_failure_and_success()
        
        if not failure_record or not success_record:
            return {}
        
        intermediate_steps = []
        
        if hasattr(agent, 'memory') and agent.memory and hasattr(agent.memory, 'steps'):
            start_step = failure_record.step_idx
            end_step = success_record.step_idx
            
            
            for step_idx in range(start_step, end_step + 1):
                if step_idx < len(agent.memory.steps):
                    step = agent.memory.steps[step_idx]
                    
                    if hasattr(step, "model_output_message") and step.model_output_message:
                        content = step.model_output_message.content
                        if content:
                            intermediate_steps.append({
                                "step_idx": step_idx,
                                "type": "agent_output",
                                "content": content
                            })
                    
                    if hasattr(step, "tool_calls") and step.tool_calls:
                        for tool_call in step.tool_calls:
                            tool_name = getattr(tool_call, 'name', 'unknown')
                            intermediate_steps.append({
                                "step_idx": step_idx,
                                "type": "tool_call",
                                "tool": tool_name
                            })
        
        context = {
            "failure": {
                "step_idx": failure_record.step_idx,
                "code": failure_record.code,
                "result": failure_record.result,
                "tool_type": failure_record.tool_type,
                "timestamp": failure_record.timestamp
            },
            "success": {
                "step_idx": success_record.step_idx,
                "code": success_record.code,
                "result": success_record.result,
                "tool_type": success_record.tool_type,
                "timestamp": success_record.timestamp
            },
            "intermediate_steps": intermediate_steps,
            "steps_count": len(intermediate_steps)
        }
        
        
        return context
    
    def reset(self):
        self.records.clear()
        self._last_failure_idx = None

    def check_generated_cases_improvement(self) -> bool:

        has_failure = False
        has_success_after_failure = False
        
        for record in self.records:
            stats = record.generated_stats
            if not stats or not stats.get("has_generated_cases"):
                continue
                
            if not stats.get("generated_run"):

                pass
            
            generated_passed = stats.get("generated_passed", False)
            
            if not generated_passed:
                has_failure = True
            elif has_failure and generated_passed:
                has_success_after_failure = True
                break
                
        if has_success_after_failure:
            return True
            
        return False


