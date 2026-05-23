

from typing import Any, Optional
from smolagents import LiteLLMModel
from smolagents.models import ChatMessage

class TokenTrackingModel:
    """
    Model wrapper with token usage tracking.
    
    Wraps any LiteLLMModel and automatically records token usage for each call.
    """
    
    def __init__(self, model: LiteLLMModel, source: str = "agent"):
        self.model = model
        self.source = source
        # Copy attributes from the original model
        self.model_id = model.model_id if hasattr(model, 'model_id') else None
    
    def generate(self, *args, **kwargs):
        """Wrap generate method to record token usage"""
        response = self.model.generate(*args, **kwargs)
        self._record_token_usage(response)
        return response
    
    def __call__(self, *args, **kwargs):
        """Wrap __call__ method to record token usage"""
        response = self.model(*args, **kwargs)
        self._record_token_usage(response)
        return response
    
    def _record_token_usage(self, response):
        """Extract and record token usage from response"""
        try:
            from agentflow.tools.session_manager import add_token_usage
            
            prompt_tokens = 0
            completion_tokens = 0
            
            # Try to get from smolagents ChatMessage (uses input_tokens/output_tokens)
            if hasattr(response, 'token_usage') and response.token_usage:
                usage = response.token_usage
                # smolagents TokenUsage uses input_tokens/output_tokens
                if hasattr(usage, 'input_tokens'):
                    prompt_tokens = usage.input_tokens or 0
                    completion_tokens = usage.output_tokens or 0
                elif hasattr(usage, 'prompt_tokens'):
                    prompt_tokens = usage.prompt_tokens or 0
                    completion_tokens = usage.completion_tokens or 0
                elif isinstance(usage, dict):
                    prompt_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0
                    completion_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0
            
            # Try to get from usage attribute (LiteLLM direct response format)
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
            
            if prompt_tokens > 0 or completion_tokens > 0:
                add_token_usage(prompt_tokens, completion_tokens, self.source)
                
        except Exception as e:
            print(f"[TokenTrackingModel] Warning: Failed to record tokens: {e}")
    
    def __getattr__(self, name):
        """Proxy other attributes to the original model"""
        return getattr(self.model, name)

def create_model(
    model_id: str,
    api_base: str,
    api_key: str,
    include_reasoning_in_context: bool = False,
    track_tokens: bool = True,  # Whether to enable token tracking
    token_source: str = "agent",  # Token source identifier
    **kwargs
) -> LiteLLMModel:
    """
    Factory function: Create appropriate model instance based on model ID.
    
    If the model is deepseek-reasoner and include_reasoning_in_context=True,
    returns DeepSeekReasonerModel; otherwise returns a regular LiteLLMModel.
    
    For non-DeepSeek models using custom api_base, automatically adds openai/ prefix
    to let litellm know to use OpenAI-compatible protocol.
    
    Args:
        model_id: Model ID
        api_base: API base URL
        api_key: API key
        include_reasoning_in_context: Whether to include reasoning_content in context
        track_tokens: Whether to enable token tracking (default enabled)
        token_source: Token source identifier (default "agent")
        **kwargs: Other model parameters
        
    Returns:
        LiteLLMModel or DeepSeekReasonerModel instance (may be wrapped by TokenTrackingModel)
    """
    # Known providers natively supported by litellm (no openai/ prefix needed)
    KNOWN_PROVIDERS = [
        "deepseek",  # deepseek-chat, deepseek-reasoner
        "gpt-",      # gpt-4, gpt-4o, gpt-3.5-turbo
        "o1",        # o1-preview, o1-mini
        "claude",    # claude-3-xxx
        "gemini",    # gemini-xxx
    ]
    
    # Check if openai/ prefix is needed
    # If model_id is not from a known provider and uses custom api_base, add prefix
    needs_openai_prefix = True
    model_id_lower = model_id.lower()
    
    # If already has provider prefix, no need to add
    if "/" in model_id:
        needs_openai_prefix = False
    else:
        # Check if it's a model from a known provider
        for provider in KNOWN_PROVIDERS:
            if provider in model_id_lower:
                needs_openai_prefix = False
                break
    
    # Add openai/ prefix for custom API
    effective_model_id = model_id
    if needs_openai_prefix and api_base:
        effective_model_id = f"openai/{model_id}"
        print(f"Custom API detected, using model_id: {effective_model_id}")
    
    if "deepseek-reasoner" in model_id.lower() and include_reasoning_in_context:
        print(f"Using DeepSeekReasonerModel (include_reasoning=True)")
        model = DeepSeekReasonerModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            include_reasoning=True,
            **kwargs
        )
    else:
        model = LiteLLMModel(
            model_id=effective_model_id,
            api_base=api_base,
            api_key=api_key,
            **kwargs
        )
    
    # If token tracking is enabled, wrap the model
    if track_tokens:
        print(f"Token tracking enabled (source: {token_source})")
        return TokenTrackingModel(model, source=token_source)
    
    return model
