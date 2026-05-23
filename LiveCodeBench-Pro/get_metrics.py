import json
import sys
import os

def get_metrics(json_path):
    if not os.path.exists(json_path):
        return "Error: File not found"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        return "Error: Empty data"
    
    total_problems = len(data)
    accepted = sum(1 for item in data if item.get('judge_result') == 'Accepted')
    acc = accepted / total_problems if total_problems > 0 else 0
    
    total_prompt_tokens = sum(item.get('total_prompt_tokens', 0) for item in data)
    total_completion_tokens = sum(item.get('total_completion_tokens', 0) for item in data)
    
    avg_prompt = total_prompt_tokens / total_problems if total_problems > 0 else 0
    avg_completion = total_completion_tokens / total_problems if total_problems > 0 else 0
    
    return f"Average acc: {acc:.4f}, Average input token: {avg_prompt:.2f}, Average output token: {avg_completion:.2f}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_metrics.py <json_path>")
    else:
        print(get_metrics(sys.argv[1]))
