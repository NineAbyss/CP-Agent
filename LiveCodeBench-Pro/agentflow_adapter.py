import sys
import os
import re

import litellm
litellm.drop_params = True

litellm.callbacks = []
litellm._async_success_callback = []
litellm._async_failure_callback = []

agentflow_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, agentflow_path)

try:
    import smolagents.models as _smolagents_models
    _original_supports_stop = _smolagents_models.supports_stop_parameter
    def _patched_supports_stop_parameter(model_id: str) -> bool:
        model_name = model_id.split("/")[-1].lower()
        if re.match(r"gpt-5\.\d+", model_name):
            return False
        return _original_supports_stop(model_id)
    _smolagents_models.supports_stop_parameter = _patched_supports_stop_parameter
except ImportError:
    pass

from typing import Any, Tuple
from api_interface import LLMInterface


class AgentflowLLM(LLMInterface):
    
    def __init__(
        self, 
        config_path: str = None,
        model_id: str = None,
        api_base: str = None,
        api_key: str = None,
        tools_config: str = None,
        unit_test_cases_on: bool = None,
        preprocess_config: str = None):
        if config_path:
            from agentflow import config as agentflow_config
            agentflow_config.reload_config(config_path)
            print(f"AgentflowLLM: Loaded config file {config_path}")
        
        from agentflow import config as agentflow_config
        
        self.model_id = model_id or agentflow_config.MODEL_ID
        self.api_base = api_base or agentflow_config.API_BASE
        self.api_key = api_key or agentflow_config.API_KEY
        self.tools_config = tools_config
        self.unit_test_cases_on = unit_test_cases_on
        self.preprocess_config = preprocess_config
        
        self.name = f"agentflow-{self.model_id}"
        if config_path:
            config_name = os.path.basename(config_path).replace(".yaml", "")
            self.name = f"agentflow-{config_name}"
        
        print(f"AgentflowLLM initialized")
        print(f"   Name: {self.name}")
        print(f"   Model: {self.model_id}")
        print(f"   API: {self.api_base}")

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        from agentflow.main_agent import run_coding_agent_with_session
        
        result, tracker_stats = run_coding_agent_with_session(
            question=user_prompt,
            model_id=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            tools_config=self.tools_config,
            unit_test_cases_on=self.unit_test_cases_on,
            preprocess_config=self.preprocess_config
        )
        
        metadata = {
            "tracker_stats": tracker_stats,
            "model_id": self.model_id,
            "config_name": self.name,
            "total_prompt_tokens": tracker_stats.get("total_prompt_tokens", 0),
            "total_completion_tokens": tracker_stats.get("total_completion_tokens", 0),
            "total_tokens": tracker_stats.get("total_tokens", 0),
            "step_count": tracker_stats.get("step_count", 0),
            "memory_size": tracker_stats.get("memory_size", 0),
            "general_exp_count": tracker_stats.get("general_exp_count", 0),
            "algo_exp_count": tracker_stats.get("algo_exp_count", 0),
        }
        
        return str(result), metadata
    
    def generate_solution(self, problem_statement: str) -> Tuple[str, Any]:
        return self.call_llm(problem_statement)


def create_agentflow_llm(config_name: str = "ds-chat_reflection_only") -> AgentflowLLM:
    if "/" not in config_name:
        possible_paths = [
            f"agentflow/configs/{config_name}.yaml",
            f"agentflow/configs/ds-chat/{config_name}.yaml",
            f"agentflow/configs/ds-reasoner/{config_name}.yaml",
            f"agentflow/configs/4o/{config_name}.yaml",
            f"agentflow/configs/o4mini/{config_name}.yaml",
        ]
        
        for path in possible_paths:
            full_path = os.path.join(agentflow_path, path)
            if os.path.exists(full_path):
                return AgentflowLLM(config_path=path)
        
        raise FileNotFoundError(f"Config file not found: {config_name}.yaml")
    else:
        return AgentflowLLM(config_path=config_name)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing AgentflowLLM adapter")
    print("=" * 60)
    
    llm = create_agentflow_llm("ds-chat_reflection_only")
    
    test_problem = """
    Given an integer n, output the sum of all integers from 1 to n.
    
    Input: An integer n (1 <= n <= 1000)
    Output: The sum from 1 to n
    
    Example:
    Input: 5
    Output: 15
    """
    
    print("\nTest problem:")
    print(test_problem)
    print("\nGenerating solution...")
    
    response, meta = llm.generate_solution(test_problem)
    
    print("\n" + "=" * 60)
    print("Generated response:")
    print(response[:1000] + "..." if len(response) > 1000 else response)
    print("\nMetadata:")
    print(meta)
