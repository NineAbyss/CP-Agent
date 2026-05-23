import os
import yaml
from pathlib import Path

# =============================================================================
# Configuration Loader
# =============================================================================

def load_yaml_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.
    
    Priority: Environment variable CONFIG_PATH > parameter > default
    """
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "agentflow/configs/default.yaml")
    
    config_file = Path(config_path)
    if not config_file.is_absolute():
        # Try to find the file relative to the project root
        project_root = Path(__file__).parent.parent
        config_file = project_root / config_path
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    else:
        print(f"Warning: Config file not found: {config_file}, using defaults")
        return {}


def get_config_value(yaml_config: dict, keys: list, env_var: str = None, type_cast=None):
    """Get config value with priority: env_var > yaml"""
    # First check environment variable
    if env_var and os.getenv(env_var) is not None:
        value = os.getenv(env_var)
        if type_cast:
            return type_cast(value)
        return value
    
    # Then check YAML config
    value = yaml_config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    return value


# Load YAML configuration
_yaml_config = load_yaml_config()

# =============================================================================
# Global Service Configuration
# =============================================================================

# Embedding API Configuration
EMBEDDING_API_URL = get_config_value(_yaml_config, ['global', 'embedding_api_url'])

# Phoenix Monitoring Configuration
PHOENIX_COLLECTOR_ENDPOINT = get_config_value(_yaml_config, ['global', 'phoenix_collector_endpoint'])
PHOENIX_PROJECT_NAME = get_config_value(_yaml_config, ['global', 'phoenix_project_name'],
                                         env_var="PHOENIX_PROJECT_NAME")

# Work Directory Configuration
# Session data will be stored in {WORK_DIR}/{session_id}/
WORK_DIR = get_config_value(_yaml_config, ['global', 'work_dir'],
                            env_var="WORK_DIR")

# Sandbox Configuration
SANDBOX_URL = get_config_value(_yaml_config, ['global', 'sandbox_url'],
                               env_var="SANDBOX_URL")

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_ID = get_config_value(_yaml_config, ['model', 'model_id'], 
                            env_var="MODEL_ID")
API_BASE = get_config_value(_yaml_config, ['model', 'api_base'], 
                            env_var="API_BASE")

_api_key_from_yaml = get_config_value(_yaml_config, ['model', 'api_key'])
API_KEY = _api_key_from_yaml or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
MODEL_TEMPERATURE = get_config_value(_yaml_config, ['model', 'temperature'], 
                                      env_var="MODEL_TEMPERATURE", type_cast=float)
MODEL_MAX_TOKENS = get_config_value(_yaml_config, ['model', 'max_tokens'], 
                                     env_var="MODEL_MAX_TOKENS", type_cast=int)
INCLUDE_REASONING_IN_CONTEXT = get_config_value(_yaml_config, ['model', 'include_reasoning_in_context'], 
                                                 env_var="INCLUDE_REASONING_IN_CONTEXT")

# =============================================================================
# Tool Switches & Configuration
# =============================================================================

# List of enabled tools for the agent
# Options: "cpp_validation", "cpp_interpreter", "oi_wiki_retrieval", "luogu_retrieval", "oeis", "error_memory"
TOOLS_ENABLED = get_config_value(_yaml_config, ['tools', 'enabled'])

# List of enabled preprocessing tools (run before the main agent loop)
# Options: "simplifier", "info_extractor"
PREPROCESS_TOOLS_ENABLED = get_config_value(_yaml_config, ['tools', 'preprocess_enabled'])

# Feature Flags
UNIT_TEST_CASES_ON = get_config_value(_yaml_config, ['features', 'unit_test_cases_on'])
LONG_TERM_EXP_ENABLED = get_config_value(_yaml_config, ['features', 'long_term_exp_enabled'])
MEMORY_STORE_PATH = get_config_value(_yaml_config, ['features', 'memory_store_path'])
ENABLE_EXPERIENCE_DEDUP = get_config_value(_yaml_config, ['features', 'enable_experience_dedup'])


COMPRESS_SIGNATURE = get_config_value(_yaml_config, ['features', 'compress_signature'])
PROBLEM_SIGNATURE_ENABLED = get_config_value(_yaml_config, ['features', 'problem_signature_enabled'])

# =============================================================================
# Resource Paths
# =============================================================================


# Unit Test Generation Config
BF_SOLUTION_SAMPLE_N = get_config_value(_yaml_config, ['resources', 'bf_solution_sample_n'], 
                                         env_var="BF_SOLUTION_SAMPLE_N", type_cast=int)
BF_SOLUTION_TEMPERATURE = get_config_value(_yaml_config, ['resources', 'bf_solution_temperature'], 
                                            env_var="BF_SOLUTION_TEMPERATURE", type_cast=float)

# =============================================================================
# Agent Behavior Configuration
# =============================================================================

# Agent reasoning steps limit
AGENT_MAX_STEPS = get_config_value(_yaml_config, ['agent', 'max_steps'], 
                                    env_var="AGENT_MAX_STEPS", type_cast=int)

# Agent verbosity level (0-3, higher = more verbose)
AGENT_VERBOSITY_LEVEL = get_config_value(_yaml_config, ['agent', 'verbosity_level'], 
                                          env_var="AGENT_VERBOSITY_LEVEL", type_cast=int)

# Agent prompt template path
AGENT_PROMPT_PATH = get_config_value(_yaml_config, ['agent', 'prompt_path'], 
                                      env_var="AGENT_PROMPT_PATH")

# Agent type: "code" for CodeAgent, "tool_calling" for ToolCallingAgent
AGENT_TYPE = get_config_value(_yaml_config, ['agent', 'agent_type'], 
                               env_var="AGENT_TYPE")


# =============================================================================
# Utility Function to Reload Config
# =============================================================================

def reload_config(config_path: str = None):
    """Reload configuration from a different YAML file.
    
    Usage:
        from agentflow.config import reload_config
        reload_config("agentflow/configs/my_config.yaml")
    """
    global _yaml_config
    global EMBEDDING_API_URL, PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME, WORK_DIR
    global MODEL_ID, API_BASE, API_KEY, MODEL_TEMPERATURE, MODEL_MAX_TOKENS, INCLUDE_REASONING_IN_CONTEXT
    global TOOLS_ENABLED, PREPROCESS_TOOLS_ENABLED
    global UNIT_TEST_CASES_ON, LONG_TERM_EXP_ENABLED, MEMORY_STORE_PATH, ENABLE_EXPERIENCE_DEDUP
    global COMPRESS_SIGNATURE, PROBLEM_SIGNATURE_ENABLED
    global BF_SOLUTION_SAMPLE_N, BF_SOLUTION_TEMPERATURE
    global AGENT_MAX_STEPS, AGENT_VERBOSITY_LEVEL, AGENT_PROMPT_PATH, AGENT_TYPE
    global SANDBOX_URL

    _yaml_config = load_yaml_config(config_path)
    
    # Reload all values
    EMBEDDING_API_URL = get_config_value(_yaml_config, ['global', 'embedding_api_url'])
    PHOENIX_COLLECTOR_ENDPOINT = get_config_value(_yaml_config, ['global', 'phoenix_collector_endpoint'])
    PHOENIX_PROJECT_NAME = get_config_value(_yaml_config, ['global', 'phoenix_project_name'],
                                             env_var="PHOENIX_PROJECT_NAME")
    WORK_DIR = get_config_value(_yaml_config, ['global', 'work_dir'],
                                env_var="WORK_DIR")
    SANDBOX_URL = get_config_value(_yaml_config, ['global', 'sandbox_url'],
                                   env_var="SANDBOX_URL")
    MODEL_ID = get_config_value(_yaml_config, ['model', 'model_id'], 
                                env_var="MODEL_ID")
    API_BASE = get_config_value(_yaml_config, ['model', 'api_base'], 
                                env_var="API_BASE")
    _api_key_from_yaml = get_config_value(_yaml_config, ['model', 'api_key'])
    API_KEY = _api_key_from_yaml or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
    MODEL_TEMPERATURE = get_config_value(_yaml_config, ['model', 'temperature'], 
                                          env_var="MODEL_TEMPERATURE", type_cast=float)
    MODEL_MAX_TOKENS = get_config_value(_yaml_config, ['model', 'max_tokens'], 
                                         env_var="MODEL_MAX_TOKENS", type_cast=int)
    INCLUDE_REASONING_IN_CONTEXT = get_config_value(_yaml_config, ['model', 'include_reasoning_in_context'], 
                                                     env_var="INCLUDE_REASONING_IN_CONTEXT")
    TOOLS_ENABLED = get_config_value(_yaml_config, ['tools', 'enabled'])
    PREPROCESS_TOOLS_ENABLED = get_config_value(_yaml_config, ['tools', 'preprocess_enabled'])
    UNIT_TEST_CASES_ON = get_config_value(_yaml_config, ['features', 'unit_test_cases_on'])
    LONG_TERM_EXP_ENABLED = get_config_value(_yaml_config, ['features', 'long_term_exp_enabled'])
    MEMORY_STORE_PATH = get_config_value(_yaml_config, ['features', 'memory_store_path'])
    ENABLE_EXPERIENCE_DEDUP = get_config_value(_yaml_config, ['features', 'enable_experience_dedup'])

    COMPRESS_SIGNATURE = get_config_value(_yaml_config, ['features', 'compress_signature'])
    PROBLEM_SIGNATURE_ENABLED = get_config_value(_yaml_config, ['features', 'problem_signature_enabled'])

    BF_SOLUTION_SAMPLE_N = get_config_value(_yaml_config, ['resources', 'bf_solution_sample_n'], 
                                             env_var="BF_SOLUTION_SAMPLE_N", type_cast=int)
    BF_SOLUTION_TEMPERATURE = get_config_value(_yaml_config, ['resources', 'bf_solution_temperature'], 
                                                env_var="BF_SOLUTION_TEMPERATURE", type_cast=float)
    AGENT_MAX_STEPS = get_config_value(_yaml_config, ['agent', 'max_steps'], 
                                        env_var="AGENT_MAX_STEPS", type_cast=int)
    AGENT_VERBOSITY_LEVEL = get_config_value(_yaml_config, ['agent', 'verbosity_level'], 
                                              env_var="AGENT_VERBOSITY_LEVEL", type_cast=int)
    AGENT_PROMPT_PATH = get_config_value(_yaml_config, ['agent', 'prompt_path'], 
                                          env_var="AGENT_PROMPT_PATH")
    AGENT_TYPE = get_config_value(_yaml_config, ['agent', 'agent_type'], 
                                   env_var="AGENT_TYPE")
