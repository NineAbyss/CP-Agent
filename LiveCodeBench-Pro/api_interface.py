from abc import ABC, abstractmethod
from typing import Any, Tuple
from openai import OpenAI


class LLMInterface(ABC):
    """
    Abstract base class for integrating Large Language Models (LLMs) into a competitive programming context.

    Attributes:
        name (str): Name of the LLM model.
        prompt (str): Initial prompt to set the context for the LLM.
    """

    name: str = "base"  # Override in subclass

    def __init__(self):
        """
        Initialize the LLMInterface with a predefined prompt for generating competitive programming solutions.
        """
        self.prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.
        """

    @abstractmethod
    def call_llm(user_prompt: str) -> Tuple[str, Any]:
        """
        Abstract method to interact with the LLM.

        Args:
            user_prompt (str): The prompt containing the problem statement for the LLM.

        Returns:
            Tuple[str, Any]: A tuple containing the generated solution and additional metadata.
        """
        pass

    def generate_solution(self, problem_statement: str) -> Tuple[str, Any]:
        """
        Generates a solution to a given competitive programming problem using the LLM.

        Args:
            problem_statement (str): The competitive programming problem statement.

        Returns:
            Tuple[str, Any]: The generated solution and associated metadata.
        """
        user_prompt = self.prompt + problem_statement
        response, meta = self.call_llm(user_prompt)
        return response, meta


class ExampleLLM(LLMInterface):
    """
    Concrete implementation of LLMInterface using OpenAI's GPT-4o model.

    Attributes:
        client (OpenAI): Client instance for interacting with the OpenAI API.
    """

    name = "gpt-4o"

    def __init__(self):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        super().__init__()
        self.client = OpenAI()

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenAI's GPT-4o model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, Any]: The LLM's response and metadata about the completion.
        """
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=8192
        )
        return completion.choices[0].message.content, str(completion)


class DeepSeekLLM(LLMInterface):
    """
    Concrete implementation of LLMInterface using DeepSeek's model.
    
    Supported models:
        - deepseek-chat: max_tokens=8192
        - deepseek-reasoner: max_tokens=65536
    """

    # Model configurations: model_name -> max_tokens
    MODEL_CONFIG = {
        "deepseek-chat": 8192,
        "deepseek-reasoner": 65536,
    }

    def __init__(self, model: str = "deepseek-reasoner"):
        super().__init__()
        if model not in self.MODEL_CONFIG:
            raise ValueError(f"Unsupported model: {model}. Supported: {list(self.MODEL_CONFIG.keys())}")
        self.model = model
        self.name = model
        self.max_tokens = self.MODEL_CONFIG[model]
        self.client = OpenAI(
            api_key="",
            base_url="https://api.deepseek.com"
        )

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content, str(completion)


class QwenLLM(LLMInterface):
    """
    Concrete implementation of LLMInterface using Qwen's model via DashScope.
    
    Supports models like:
        - qwen3-235b-a22b-instruct-2507
        - qwen-turbo
        - qwen-plus
        - etc.
    """

    def __init__(
        self, 
        model: str = "qwen3-235b-a22b-instruct-2507",
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens: int = 65536
    ):
        super().__init__()
        self.model = model
        self.name = model
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content, str(completion)


if __name__ == "__main__":
    """
    Example execution demonstrating how to use the DeepSeekLLM class to generate solutions.
    """
    llm = DeepSeekLLM()
    response, meta = llm.generate_solution("Hello world")
    print(response)
    print(meta)
