from __future__ import annotations

import os
import re
import json
import time
import datetime
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import openai
from openai import OpenAI

from smolagents import Tool
from agentflow.tools.session_manager import get_current_session_id
import requests

# Model configuration (aligned with other tools, but we call the API directly)
MODEL_NAME = "deepseek-chat"
API_BASE = "https://api.deepseek.com/v1"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


class ValidatorTool(Tool):
    """
    Validator tool that orchestrates multi-turn prompt construction to synthesize
    a C++ validator program for an algorithm problem, compiles it, and runs it
    against a sample input located under tools/cpp/unit_test_data/<conversation_id>/.

    The tool iteratively calls the LLM API (no agent.run) to fix compilation/runtime
    issues until the program compiles and runs without exceptions, then saves the
    final code and the full conversation history to JSON.
    """

    name = "validator"
    description = (
"Construct multi-round prompts to synthesize a C++ validator, compile it and run it on the sample inputs under the `unit_test_data/` directory, and iterate upon errors until it succeeds."
    )
    inputs = {
        "question": {
            "type": "string",
            "description": "Algorithm problem description and constraints."
        },
        "language": {
            "type": "string",
            "description": "Language hint (unused)."
        },
        "conversation_id": {
            "type": "string",
            "description": "Optional conversation ID to scope work directory.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """Initialize conversation state and defaults."""
        super().__init__(**kwargs)
        self._current_session_id: Optional[str] = None
        self._work_dir: Optional[str] = None
        self._conversation: List[Dict[str, str]] = []
        print("ValidatorTool initialized (API mode, no agent.run)")

    # ------------------------- Path/session helpers ------------------------- #

    def _set_work_dir(self, conversation_id: str) -> str:
        """
        Resolve and create the work directory under tools/cpp/unit_test_data/<conversation_id>/.
        This mirrors the unit test data layout used elsewhere for consistency.
        """
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
            print("global_dir is None")
        if global_dir:
            unit_test_data_dir = os.path.join(global_dir, "unit_test_data")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            unit_test_data_dir = os.path.join(project_root, "tools", "cpp", "unit_test_data", conversation_id)
        os.makedirs(unit_test_data_dir, exist_ok=True)
        self._work_dir = unit_test_data_dir
        print(f"Set validator output directory to: {unit_test_data_dir}")
        return unit_test_data_dir

    def _set_session_id(self, session_id: str):
        """Set current conversation/session identifier for scoping outputs."""
        self._current_session_id = session_id

    def _get_session_id(self) -> Optional[str]:
        """Get the active conversation/session id if any."""
        return self._current_session_id

    def _generate_conversation_id(self, problem_text: str) -> str:
        """Generate a stable conversation id based on timestamp and hash of problem text."""
        timestamp = str(int(time.time() * 1000))
        problem_hash = re.sub(r"[^0-9a-f]", "", str(hash(problem_text)))[:8]
        return f"auto_{timestamp}_{problem_hash}"

    # ----------------------------- Prompt builders ----------------------------- #

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt to set the LLM role and strict output format.
        The LLM must return a complete C++ program containing a validate function
        and a main that reads from stdin and prints ONLY "VALID" or "INVALID".
        """
        return (
            "You are an expert input validator author for competitive programming. "
            "Write robust, efficient C++17 code. Your output MUST be a single C++ "
            "program in one fenced code block ```cpp ... ```. The program must: "
            "(1) define a function bool validate(/* problem-specific signature */) that checks input format and constraints; "
            "(2) read the entire input from stdin in main(), call validate, and print EXACTLY 'VALID' or 'INVALID' followed by a newline; "
            "(3) avoid external dependencies; (4) compile with g++ -O2 -std=c++17."
        )

    def _build_user_prompt(self, question: str) -> str:
        """
        Build the initial user prompt with the problem statement and strict instructions
        for the validator behavior and I/O contract.
        """
        return (
            "Problem statement and constraints:\n\n"
            f"{question}\n\n"
            "Task: Implement a C++17 program containing a bool validate(...) function that checks whether the given input strictly follows the problem's input format and constraints.\n"
            "- If the input violates format or constraints, return false.\n"
            "- If the input is valid, return true.\n\n"
            "Main() must: read from stdin, call validate, and print ONLY 'VALID' if true else 'INVALID', then return 0.\n"
            "Do NOT include any comments or explanations outside the code block."
        )

    # --------------------------- Conversation helpers --------------------------- #

    def _append_message(self, role: str, content: str):
        """
        Append a message to in-memory conversation and persist to disk for continuity.
        The conversation is saved to a JSON file under the work directory.
        """
        self._conversation.append({"role": role, "content": content})
        try:
            work_dir = self._work_dir or "."
            os.makedirs(work_dir, exist_ok=True)
            log_dir = os.path.join(work_dir, "logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "validator_chat_log.json")
            existing: List[Dict[str, str]]
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []
            existing.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "role": role,
                "content": content,
                "model": MODEL_NAME,
            })
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: failed to persist conversation log: {e}")

    # ----------------------------- API integration ----------------------------- #

    def _api_chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 60) -> str:
        """
        Call the OpenAI-style chat completions API and return the content string.
        Implements infinite retry with 30-second delay on API errors.
        """

        # Create OpenAI client with API key and base URL
        client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                stream=False,
            )
            return response.choices[0].message.content
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ) as e:
            print(f"[Validator API Error] {repr(e)}")
            print("Sleeping for 30 seconds before retry...")
            print("Consider reducing the number of parallel processes.")
            from time import sleep
            sleep(30)
            return self._api_chat(messages, temperature, timeout)
        except Exception as e:
            print(f"[Validator] Fatal error (non-retryable): {repr(e)}")
            raise e

    def _call_llm_and_extract_cpp(self) -> str:
        """
        Call the LLM API with the full conversation, expecting a single ```cpp ... ``` fenced block.
        Extract and return the C++ source code. If not found, return an informative stub.
        """
        response_text = self._api_chat(self._conversation)
        print(response_text)
        self._append_message("assistant", response_text)
        try:
            pattern = r"```cpp\n(.*?)```"
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                return matches[-1].strip()
            return "// Failed to extract C++ code from the response."
        except Exception as e:
            return f"// Exception while extracting C++ code: {e}"

    # --------------------------- Build / Run utilities -------------------------- #

    def _write_cpp(self, code: str, filename: str = "validate.cpp") -> str:
        """Write the current C++ source to the work directory and return absolute path."""
        assert self._work_dir is not None
        cpp_path = os.path.join(self._work_dir, filename)
        with open(cpp_path, "w", encoding="utf-8") as f:
            f.write(code)
        return os.path.abspath(cpp_path)

    def _compile_cpp(self, cpp_file: str) -> Tuple[bool, str, str]:
        """
        Compile the provided C++ source file into an executable in the work directory.
        Returns (success, error_message, exec_path).
        """
        compile_success = False
        compile_error = ""
        exec_path = ""
        try:
            base_name = os.path.splitext(os.path.basename(cpp_file))[0]
            exec_dir = self._work_dir or os.path.dirname(cpp_file)
            exec_path = os.path.join(exec_dir, base_name)
            compile_cmd = [
                "g++", "-O2", "-std=c++17",
                "-o", exec_path,
                cpp_file,
            ]
            print(f"compile_cmd: {compile_cmd}")
            result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=exec_dir)
            if result.returncode == 0:
                compile_success = True
            else:
                compile_error = result.stderr
                exec_path = ""
        except Exception as e:
            compile_error = str(e)
            exec_path = ""
        print(f"compile_success: {compile_success}")
        print(f"compile_error: {compile_error}")
        print(f"exec_path: {exec_path}")
        return compile_success, compile_error, exec_path

    def _detect_sample_input_path(self) -> Optional[str]:
        """
        Locate the sample input file to validate against. Priority order under the
        work directory: 'sample_test_input', 'input', 'input_1'. Returns absolute path or None.
        """
        assert self._work_dir is not None
        # Use regex to match filenames prefixed with "sample"
        pattern = re.compile(r"^sample.*")
        candidates = []
        sample_input_path = self._work_dir
        for fname in os.listdir(sample_input_path):
            if pattern.match(fname):
                fpath = os.path.join(sample_input_path, fname)
                if os.path.isfile(fpath):
                    candidates.append(fpath)
        print(f"candidates: {candidates}")
        for p in candidates:
            if os.path.exists(p) and os.path.isfile(p):
                return os.path.abspath(p)
        return None

    def _run_with_input(self, exec_path: str, input_path: str, timeout_seconds: int = 60) -> Tuple[int, str, str]:
        """
        Run the compiled executable with the given input file piped to stdin.
        Returns (exit_code, stdout, stderr). Captures timeouts and exceptions as non-zero exit.
        """
        try:
            with open(input_path, "rb") as fin:
                completed = subprocess.run(
                    [exec_path],
                    stdin=fin,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=self._work_dir,
                )
            return completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as e:
            return 124, e.stdout or "", e.stderr or f"Command timed out after {timeout_seconds}s"
        except Exception as e:
            return 1, "", str(e)

    # --------------------------------- Public API -------------------------------- #

    def forward(self, question: str, language: str, conversation_id: Optional[str] = None) -> str:
        """
        Entry point for the validator tool (API mode).
        - Constructs system and user prompts and stores them as a multi-turn conversation.
        - Iteratively calls LLM API to obtain a C++ validator, compiles and runs it against sample input.
        - On any exception (compile/run), appends the error as a new user turn and retries.
        - Stops when compile and run both succeed without exceptions. Saves final code to work dir.
        The full conversation is saved as JSON under the work directory.
        Returns a status string including output directory and file paths.
        """
        try:
            if not question:
                return "Error: Empty question text provided"

            # Resolve conversation/session id
            conversation_id = self._get_session_id() or conversation_id
            if not conversation_id:
                try:
                    conversation_id = get_current_session_id()
                except ImportError:
                    conversation_id = None
            if not conversation_id:
                conversation_id = self._generate_conversation_id(question)

            self._set_session_id(conversation_id)
            work_dir = self._set_work_dir(conversation_id)

            # Persist the original problem for traceability
            original_question_path = os.path.join(work_dir, "original_question.txt")
            with open(original_question_path, "w", encoding="utf-8") as f:
                f.write(question)
            print(f"Original question saved to {original_question_path}")

            # Build initial conversation (system + user)
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(question)
            self._append_message("system", system_prompt)
            self._append_message("user", user_prompt)

            # Locate sample input file
            input_path = self._detect_sample_input_path()
            if not input_path:
                return (
                    "Error: No sample input found. Expected one or more files with prefix 'sample' "
                    f"under {work_dir}."
                )

            # Iterative refinement loop
            max_attempts = 8
            attempt = 0
            cpp_path: Optional[str] = None
            exec_path: Optional[str] = None

            while attempt < max_attempts:
                print(f"==================== Synthesizing validator... attempt {attempt} ====================")
                cpp_code = self._call_llm_and_extract_cpp()

                # Save candidate code
                cpp_path = self._write_cpp(cpp_code, filename="validate.cpp")

                # Compile
                ok, compile_err, exec_path = self._compile_cpp(cpp_path)
                if not ok:
                    # Append compile error and retry
                    error_msg = (
                        "The previous C++ program failed to compile with g++ -O2 -std=c++17.\n"
                        f"Compiler errors:\n{compile_err}\n\n"
                        "Please provide a corrected COMPLETE C++17 program in a single ```cpp code block, "
                        "following the requirements strictly (print ONLY 'VALID' or 'INVALID')."
                    )
                    self._append_message("user", error_msg)
                    attempt += 1
                    continue

                # Run
                code, out, err = self._run_with_input(exec_path, input_path)
                if code != 0:
                    # Append runtime error and retry
                    error_msg = (
                        "The compiled program failed during execution.\n"
                        f"Exit code: {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}\n\n"
                        "Please fix the program and reply with a COMPLETE C++17 program in one ```cpp block."
                    )
                    self._append_message("user", error_msg)
                    attempt += 1
                    continue

                # Success path
                print("Validator compile and run succeeded.")
                break
            else:
                return "Failed: Could not synthesize a working validator after multiple attempts."

            # Finalize and report
            final_cpp_path = os.path.join(work_dir, "validate.cpp")
            return f"OK: validator created. dir={work_dir}, validate_cpp={final_cpp_path}, executable={exec_path}"

        except Exception as e:
            print(f"Validator error: {e}")
            return f"Failed: {str(e)}"
