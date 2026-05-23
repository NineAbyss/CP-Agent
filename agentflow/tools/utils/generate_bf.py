#!/usr/bin/env python3

import argparse
import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional, Dict, Any
from openai import OpenAI
import re

user_prompt = """
You are an expert competitive programming assistant.

Your task is to write a correct and *brute-force* C++ solution for the following problem. The program must:
- Read input from standard input and write output to standard output, strictly following the problem's input/output format.
- Implement a straightforward, brute-force (but correct) algorithm, prioritizing clarity and correctness over efficiency.
- Avoid any optimizations or advanced algorithms; use only simple, exhaustive approaches.
- Ensure the code is self-contained and compiles with GNU G++23 14.2 (64bit, msys2).

---

{Question}

---

**Language and Compiler**

* **Language:** cpp
* **Compiler:** GNU G++23 14.2 (64bit, msys2)

**Output:**  
Provide only the complete C++ source code, enclosed in a single code block using the syntax ```cpp ... ```. Do not include any explanations or comments outside the code.
"""

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate brute-force C++ solutions for competitive programming problems",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-q', '--question',
        required=True,
        help="Programming question (raw text or path to .txt file)"
    )
    
    parser.add_argument(
        '-m', '--model_name',
        default='gpt-4o-mini',
        help="Model to use (default: gpt-4o-mini)"
    )

    parser.add_argument(
        '-b', '--api_base',
        default='https://api.deepseek.com/v1',
        help="Base URL for the API (default: https://api.deepseek.com/v1)"
    )

    parser.add_argument(
        '-k', '--api_key',
        help="API key for the API"
    )
    
    parser.add_argument(
        '-t', '--temperature',
        type=float,
        default=0.6,
        help="Temperature for generation (default: 0.2)"
    )
    
    parser.add_argument(
        '-o', '--output',
        default='bf_solution.cpp',
        help="Output filename (default: bf_solution.cpp)"
    )
    
    return parser.parse_args()


def read_question(question_arg: str) -> str:
    if os.path.isfile(question_arg):
        try:
            with open(question_arg, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except IOError as e:
            print(f"Error reading file '{question_arg}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        return question_arg.strip()


def send_api_request(question: str, model: str, temperature: float, api_key: str, base_url: str) -> str:
    system_prompt = "You are an elite competitive-programming assistant.\nProduce only valid C++17 code.\nUse standard I/O (cin/cout).\nInclude every helper necessary for a brute-force solution.\nDo NOT write explanations or markdown fences."
    formatted_user_prompt = user_prompt.format(Question=question)
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': formatted_user_prompt}
            ],
            temperature=temperature
        )
        
        response_content = completion.choices[0].message.content
        
        log_data = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "request": {
                "system_prompt": system_prompt,
                "user_prompt": formatted_user_prompt,
                "model": model,
                "temperature": temperature
            },
            "response": {
                "content": response_content
            }
        }
        
        log_filename = "api_conversation_log.json"
        try:
            if os.path.exists(log_filename):
                with open(log_filename, 'r', encoding='utf-8') as f:
                    log_history = json.load(f)
            else:
                log_history = []
            
            log_history.append(log_data)
            
            with open(log_filename, 'w', encoding='utf-8') as f:
                json.dump(log_history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}", file=sys.stderr)
        
        return response_content
        
    except (
        __import__('openai').APIError,
        __import__('openai').RateLimitError,
        __import__('openai').InternalServerError,
        __import__('openai').OpenAIError,
        __import__('openai').APIStatusError,
        __import__('openai').APITimeoutError,
        __import__('openai').APIConnectionError,
    ) as e:
        print(f"[generate_bf] API Error: {repr(e)}", file=sys.stderr)
        print("Sleeping for 30 seconds before retry...", file=sys.stderr)
        print("Consider reducing the number of parallel processes.", file=sys.stderr)
        __import__('time').sleep(30)
        return send_api_request(question, model, temperature, api_key, base_url)
    except Exception as e:
        print(f"[generate_bf] Fatal error (non-retryable): {repr(e)}", file=sys.stderr)
        sys.exit(1)


def extract_code_from_response(response_text: str) -> str:
    try:
        if not isinstance(response_text, str):
            return "We can not extract the code in the output. (raw_response is not a string)"
        
        pattern = r"```cpp\n(.*?)```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            code_output = matches[-1].strip()
        else:
            pattern_fallback = r"```\n(.*?)```"
            matches_fallback = re.findall(pattern_fallback, response_text, re.DOTALL)
            if matches_fallback:
                code_output = matches_fallback[-1].strip()
            else:
                code_output = "We can not extract the code in the output."
        
        return code_output
        
    except Exception as e:
        return f"We can not extract the code in the output. (Exception: {e})"


def write_code_to_file(code: str, output_file: str) -> None:
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    try:
        args = parse_arguments()
        
        question = read_question(args.question)
        
        api_key = os.getenv('DEEPSEEK_API_KEY') 
        if not api_key:
            print("Error: DEEPSEEK_API_KEY environment variable is required", file=sys.stderr)
            sys.exit(1)
        
        base_url = "https://api.deepseek.com/v1"
        
        response_content = send_api_request(
            question, args.model, args.temperature, api_key, base_url
        )
        
        code = extract_code_from_response(response_content)
        write_code_to_file(code, args.output)
        
        print("Brute-force C++ code written to", args.output)
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
