

from litellm import completion as litellm_completion
from agentflow.tools.session_manager import add_token_usage


def tracked_completion(*args, source: str = "unknown", **kwargs):

    if 'model' in kwargs:
        model_name = kwargs['model']
        if model_name and "/" not in model_name:
            known_providers = ["gpt-", "o1", "claude", "gemini"]
            needs_prefix = True
            model_lower = model_name.lower()
            
            for provider in known_providers:
                if provider in model_lower:
                    needs_prefix = False
                    break
            
            if needs_prefix:
                if "deepseek" in model_lower:
                    kwargs['model'] = f"deepseek/{model_name}"
                else:
                    kwargs['model'] = f"openai/{model_name}"
                    print(f"[TrackedCompletion] Custom API model, adding prefix: {kwargs['model']}")
    
    response = litellm_completion(*args, **kwargs)
    
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            if hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens or 0
            if hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens or 0
        elif isinstance(response, dict) and 'usage' in response:
            usage = response['usage']
            prompt_tokens = usage.get('prompt_tokens', 0) or 0
            completion_tokens = usage.get('completion_tokens', 0) or 0
    except Exception as e:
        print(f"[TrackedCompletion] : {e}")
    
    if prompt_tokens > 0 or completion_tokens > 0:
        add_token_usage(prompt_tokens, completion_tokens, source)
    
    return response


def extract_token_usage_from_response(response, source: str = "unknown"):

    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        if hasattr(response, 'token_usage') and response.token_usage:
            usage = response.token_usage
            if hasattr(usage, 'input_tokens'):
                prompt_tokens = usage.input_tokens or 0
                completion_tokens = usage.output_tokens or 0
            elif hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
            elif isinstance(usage, dict):
                prompt_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0
                completion_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0
        
        elif hasattr(response, 'usage') and response.usage:
            usage = response.usage
            if hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
            elif hasattr(usage, 'input_tokens'):
                prompt_tokens = usage.input_tokens or 0
                completion_tokens = usage.output_tokens or 0
            elif isinstance(usage, dict):
                prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0
                completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0
        
        elif hasattr(response, 'raw') and response.raw:
            raw = response.raw
            if hasattr(raw, 'usage') and raw.usage:
                usage = raw.usage
                if hasattr(usage, 'prompt_tokens'):
                    prompt_tokens = usage.prompt_tokens or 0
                    completion_tokens = usage.completion_tokens or 0
        
        elif isinstance(response, dict):
            if 'usage' in response:
                usage = response['usage']
                prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0
                completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0
        
        if prompt_tokens > 0 or completion_tokens > 0:
            add_token_usage(prompt_tokens, completion_tokens, source)
            
    except Exception as e:
        print(f"[TrackedCompletion] Failed: {e}")
    
    return prompt_tokens, completion_tokens

