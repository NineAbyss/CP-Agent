
import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from datasets import load_dataset, DatasetDict
from multiprocessing import Pool, Manager
from typing import Tuple, Any
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code
from benchmark import BenchmarkResult, ProblemTestState, get_problem_set, print_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_WORKER = 8

_LLM_CONFIG = {}


def init_worker(llm_config: dict):

    global _LLM_CONFIG
    _LLM_CONFIG = llm_config


def create_llm_instance(config: dict):

    llm_type = config.get('llm_type', 'qwen')
    
    if llm_type == 'qwen':
        from api_interface import QwenLLM
        return QwenLLM(
            model=config.get('model', 'qwen3-235b-a22b-instruct-2507'),
            api_key=config.get('api_key'),
            base_url=config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            max_tokens=config.get('max_tokens', 65536)
        )
    elif llm_type == 'openai':
        from api_interface import QwenLLM
        return QwenLLM(
            model=config.get('model', 'gpt-4o'),
            api_key=config.get('api_key'),
            base_url=config.get('base_url', 'https://api.openai.com/v1'),
            max_tokens=config.get('max_tokens', 8192)
        )
    elif llm_type == 'deepseek':
        from api_interface import DeepSeekLLM
        return DeepSeekLLM(model=config.get('model', 'deepseek-reasoner'))
    else:
        raise ValueError(f" {llm_type}")


def process_single_problem(args: Tuple[dict, str]) -> dict:

    global _LLM_CONFIG
    problem_dict, judge_url = args
    
    problem_id = problem_dict['problem_id']
    problem_title = problem_dict['problem_title']
    problem_statement = problem_dict['problem_statement']
    
    try:
        llm_instance = create_llm_instance(_LLM_CONFIG)
    except Exception as e:
        logger.error(f" : {e}")
        problem_dict['text_response'] = f"Error: LLM init failed: {e}"
        problem_dict['judge_result'] = "LLM Init Error"
        return problem_dict
    
    logger.info(f"Processing problem: {problem_id} - {problem_title}")
    

    try:
        response, meta = llm_instance.generate_solution(problem_statement)
        problem_dict['text_response'] = response
        problem_dict['code'] = extract_longest_cpp_code(response)
        problem_dict['response_meta'] = meta
        logger.info(f"[{problem_id}] ")
    except Exception as e:
        logger.error(f"[{problem_id}] : {e}")
        traceback.print_exc()
        problem_dict['text_response'] = f"Error: {e}"
        problem_dict['judge_result'] = "Generation Error"
        return problem_dict
    
    if problem_dict['code']:
        try:
            import requests
            
            submit_resp = requests.post(
                f"{judge_url}/submit",
                json={
                    "pid": problem_id,
                    "lang": "cpp",
                    "code": problem_dict['code']
                },
                timeout=30
            )
            submit_resp.raise_for_status()
            submission_id = submit_resp.json()["sid"]
            problem_dict['submission_id'] = submission_id
            
            while True:
                result_resp = requests.get(f"{judge_url}/result/{submission_id}", timeout=10)
                if result_resp.status_code == 404:
                    time.sleep(0.5)
                    continue
                result = result_resp.json()
                if result["status"] == "queued":
                    time.sleep(0.5)
                    continue
                if result["status"] == "error":
                    problem_dict['judge_result'] = "Judge Failed"
                    break
                problem_dict['judge_result'] = result["result"]
                break
            
            logger.info(f"[{problem_id}] : {problem_dict['judge_result']}")
            
        except Exception as e:
            logger.error(f"[{problem_id}] : {e}")
            problem_dict['judge_result'] = "Submission Error"
    else:
        problem_dict['judge_result'] = "No Code Extracted"
        logger.warning(f"[{problem_id}] ")
    
    return problem_dict


def save_results(results, result_filename, lock):
    with lock:
        output = []
        for r in results:
            if r.get('text_response') is not None:
                output.append({
                    "problem_id": r['problem_id'],
                    "problem_title": r['problem_title'],
                    "difficulty": r['difficulty'],
                    "platform": r['platform'],
                    "text_response": r['text_response'],
                    "code": r.get('code'),
                    "judge_result": r.get('judge_result', 'Judging'),
                    "response_meta": r.get('response_meta')
                })
        
        with open(result_filename, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3-235b-a22b-instruct-2507",
        help=""
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API Key"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="API Base URL"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help=""
    )
    parser.add_argument(
        "--llm-type",
        type=str,
        default="qwen",
        choices=["qwen", "openai", "deepseek"],
        help=""
    )
    parser.add_argument(
        "--split", 
        nargs="+", 
        help=""
    )
    parser.add_argument(
        "--difficulty", 
        nargs="+", 
        help=""
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help=""
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=DEFAULT_WORKER,
        help=f"Judge worker : {DEFAULT_WORKER})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help=""
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=""
    )
    
    parser.add_argument(
        "--judge-instance-id",
        type=str,
        default=None,
        help=""
    )
    
    args = parser.parse_args()
    
    llm_config = {
        'llm_type': args.llm_type,
        'model': args.model,
        'api_key': args.api_key,
        'base_url': args.base_url,
        'max_tokens': args.max_tokens,
    }
    

    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.split:
        available_splits = list(dataset.keys())
        selected_splits = [s for s in args.split if s in available_splits]
        if not selected_splits:

            sys.exit(1)
        dataset = DatasetDict({k: dataset[k] for k in selected_splits})
    
    if args.difficulty:
        selected_difficulties = set(args.difficulty)
        new_splits = {}
        for split_name, split_data in dataset.items():
            filtered_split = split_data.filter(lambda x: x["difficulty"] in selected_difficulties)
            if len(filtered_split) > 0:
                new_splits[split_name] = filtered_split
        if not new_splits:
            sys.exit(1)
        dataset = DatasetDict(new_splits)
    
    problem_set = get_problem_set(dataset)
    
    completed_problem_ids = set()
    existing_results = []
    if args.resume:
        if not os.path.exists(args.resume):
            sys.exit(1)
        
        try:
            with open(args.resume, "r") as f:
                existing_results = json.load(f)
            
            for item in existing_results:
                problem_id = item.get("problem_id")
                if problem_id:
                    completed_problem_ids.add(problem_id)
            
            result_filename = args.resume 
        except Exception as e:
            sys.exit(1)
    else:

        model_name_clean = args.model.replace("/", "_").replace(":", "_")
        splits_str = "_".join(sorted(dataset.keys())) if args.split else "all"
        difficulty_str = "_".join(sorted(args.difficulty)) if args.difficulty else "all"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{args.output_dir}/{model_name_clean}_{splits_str}_{difficulty_str}_{timestamp}.json"
    
    
    problems_list = [p.model_dump() for p in problem_set.values() if p.problem_id not in completed_problem_ids]
    
    
    with LightCPVerifierJudge(worker=args.worker, instance_id=args.judge_instance_id) as judge:
        judge_url = judge.base_url
        
        task_args = [(p, judge_url) for p in problems_list]
        
        manager = Manager()
        lock = manager.Lock()
        results = manager.list()
        
        for item in existing_results:
            results.append(item)
        
        total_problems = len(problem_set)
        
        if args.parallel > 1:

            with Pool(
                processes=args.parallel,
                initializer=init_worker,
                initargs=(llm_config,)
            ) as pool:
                for i, result in enumerate(pool.imap_unordered(process_single_problem, task_args)):
                    results.append(result)
                    save_results(list(results), result_filename, lock)
        else:
            import tqdm
            init_worker(llm_config)
            for task in tqdm.tqdm(task_args, desc="Benchmarking"):
                result = process_single_problem(task)
                results.append(result)
                save_results(list(results), result_filename, lock)
    
    for r in results:
        if r['problem_id'] in problem_set:
            problem_set[r['problem_id']].text_response = r.get('text_response')
            problem_set[r['problem_id']].code = r.get('code')
            problem_set[r['problem_id']].judge_result = r.get('judge_result', 'Judging')
            problem_set[r['problem_id']].response_meta = r.get('response_meta')
    
    print_stats(dataset, problem_set)
    


if __name__ == "__main__":
    main()

