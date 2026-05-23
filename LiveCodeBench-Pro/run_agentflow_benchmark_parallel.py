"""
Multi-process parallel version of LiveCodeBench-Pro benchmark

Usage:
    # Use 4 parallel processes
    python run_agentflow_benchmark_parallel.py --parallel 4
    
    # Full example
    python run_agentflow_benchmark_parallel.py \
        --config agentflow/configs/ds-chat/ds-chat_reflection_and_testcase_and_exp.yaml \
        --split quater_2025_4_6 \
        --difficulty easy medium \
        --parallel 4
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from datasets import load_dataset, DatasetDict
from multiprocessing import Pool, Manager, Lock
from functools import partial
import traceback

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code
from benchmark import BenchmarkResult, ProblemTestState, get_problem_set, print_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Judge worker count
WORKER = 8


def process_single_problem(args):
    """
    Process a single problem (runs in a child process)
    
    Args:
        args: (problem_dict, config_path, judge_url)
    
    Returns:
        Processed problem_dict
    """
    problem_dict, config_path, judge_url = args
    
    problem_id = problem_dict['problem_id']
    
    # Set process-specific work directory to avoids multi-process conflicts
    try:
        import tempfile
        process_work_dir = tempfile.mkdtemp(prefix=f"lcbpro_{problem_id}_")
        os.environ['AGENTFLOW_WORK_DIR'] = process_work_dir
    except Exception:
        pass
    
    # Re-create LLM instance in child process (avoids shared state issues)
    try:
        from agentflow_adapter import AgentflowLLM
        llm_instance = AgentflowLLM(config_path=config_path)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        problem_dict['text_response'] = f"Error: LLM init failed: {e}"
        problem_dict['judge_result'] = "LLM Init Error"
        return problem_dict
    
    problem_id = problem_dict['problem_id']
    problem_title = problem_dict['problem_title']
    problem_statement = problem_dict['problem_statement']
    
    logger.info(f"Processing problem: {problem_id} - {problem_title} (Difficulty: {problem_dict.get('difficulty')})")
    
    # 1. Generate solution
    try:
        response, meta = llm_instance.generate_solution(problem_statement)
        problem_dict['text_response'] = response
        problem_dict['code'] = extract_longest_cpp_code(response)
        problem_dict['response_meta'] = meta
        # 📊 Token statistics
        problem_dict['total_prompt_tokens'] = meta.get('total_prompt_tokens', 0)
        problem_dict['total_completion_tokens'] = meta.get('total_completion_tokens', 0)
        logger.info(f"[{problem_id}] Code generation completed, tokens: {problem_dict['total_prompt_tokens']:,}+{problem_dict['total_completion_tokens']:,}")
    except Exception as e:
        logger.error(f"[{problem_id}] Failed to generate solution: {e}")
        traceback.print_exc()
        problem_dict['text_response'] = f"Error: {e}"
        problem_dict['judge_result'] = "Generation Error"
        return problem_dict
    
    # 2. Submit for evaluation
    if problem_dict['code']:
        try:
            import requests
            
            # Submit code
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
            
            # Wait for results
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
            
            logger.info(f"[{problem_id}] Evaluation result: {problem_dict['judge_result']}")
            
        except Exception as e:
            logger.error(f"[{problem_id}] Submission failed: {e}")
            problem_dict['judge_result'] = "Submission Error"
    else:
        problem_dict['judge_result'] = "No Code Extracted"
        logger.warning(f"[{problem_id}] Failed to extract code")
    
    # Cleanup process-specific temporary directory
    try:
        process_work_dir = os.environ.get('AGENTFLOW_WORK_DIR')
        if process_work_dir and os.path.exists(process_work_dir):
            import shutil
            shutil.rmtree(process_work_dir, ignore_errors=True)
    except Exception:
        pass
    
    return problem_dict


def save_results(results, result_filename, lock):
    """Thread-safe result saving"""
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
                    "response_meta": r.get('response_meta'),
                    "total_prompt_tokens": r.get('total_prompt_tokens', 0),
                    "total_completion_tokens": r.get('total_completion_tokens', 0),
                })
        
        with open(result_filename, "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-process parallel version of LiveCodeBench-Pro benchmark"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="agentflow config file path"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="ds-chat_reflection_only",
        help="Config name (if --config is not specified)"
    )
    parser.add_argument(
        "--split", 
        nargs="+", 
        help="Dataset split"
    )
    parser.add_argument(
        "--difficulty", 
        nargs="+", 
        help="Difficulty level"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)"
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=WORKER,
        help=f"Judge worker count (default: {WORKER})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Result output directory"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing result file path"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Specify output result filename (if specified, --output-dir and auto-generated name will be ignored)"
    )
    parser.add_argument(
        "--judge-instance-id",
        type=str,
        default=None,
        help="Judge container instance ID (for multi-instance parallel runs)"
    )
    args = parser.parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        from agentflow_adapter import create_agentflow_llm
        # Get config path
        agentflow_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            f"agentflow/configs/{args.config_name}.yaml",
            f"agentflow/configs/ds-chat/{args.config_name}.yaml",
            f"agentflow/configs/ds-reasoner/{args.config_name}.yaml",
        ]
        config_path = None
        for path in possible_paths:
            full_path = os.path.join(agentflow_path, path)
            if os.path.exists(full_path):
                config_path = path
                break
        if not config_path:
            logger.error(f"Config file not found: {args.config_name}")
            sys.exit(1)
    
    logger.info(f"Using config: {config_path}")
    logger.info(f"Parallel processes: {args.parallel}")

    # Load dataset
    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter by split
    if args.split:
        available_splits = list(dataset.keys())
        selected_splits = [s for s in args.split if s in available_splits]
        if not selected_splits:
            logger.error("No valid splits found")
            sys.exit(1)
        dataset = DatasetDict({k: dataset[k] for k in selected_splits})

    # Filter by difficulty
    if args.difficulty:
        selected_difficulties = set(args.difficulty)
        new_splits = {}
        for split_name, split_data in dataset.items():
            filtered_split = split_data.filter(lambda x: x["difficulty"] in selected_difficulties)
            if len(filtered_split) > 0:
                new_splits[split_name] = filtered_split
        if not new_splits:
            logger.error("No problems found after filtering")
            sys.exit(1)
        dataset = DatasetDict(new_splits)

    # Get problem set
    problem_set = get_problem_set(dataset)
    
    # Handle resume logic
    completed_problem_ids = set()
    existing_results = []
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Resume file does not exist: {args.resume}")
            sys.exit(1)
        
        try:
            with open(args.resume, "r") as f:
                existing_results = json.load(f)
            
            for item in existing_results:
                problem_id = item.get("problem_id")
                judge_result = item.get("judge_result")
                # If generation error or other transient error, not considered completed, needs redo
                if problem_id and judge_result not in ["Generation Error", "LLM Init Error", "Submission Error", "Judge Failed"]:
                    completed_problem_ids.add(problem_id)
            
            logger.info(f"Restored {len(completed_problem_ids)} completed problems from {args.resume}")
            result_filename = args.resume  # Continue writing to same file
        except Exception as e:
            logger.error(f"Failed to read resume file: {e}")
            sys.exit(1)
    elif args.output_file:
        result_filename = args.output_file
    else:
        # Generate new result filename
        config_name = os.path.basename(config_path).replace(".yaml", "")
        model_name = f"agentflow-{config_name}"
        splits_str = "_".join(sorted(dataset.keys())) if args.split else "all"
        difficulty_str = "_".join(sorted(args.difficulty)) if args.difficulty else "all"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{args.output_dir}/{model_name}_{splits_str}_{difficulty_str}_{timestamp}.json"
    
    # Filter out completed problems
    problems_list = [p.model_dump() for p in problem_set.values() if p.problem_id not in completed_problem_ids]
    
    # Sort by the order specified in --difficulty
    if args.difficulty:
        diff_order = {d: i for i, d in enumerate(args.difficulty)}
        problems_list.sort(key=lambda p: diff_order.get(p.get('difficulty'), 999))
    
    logger.info(f"Results will be saved to: {result_filename}")
    logger.info(f"Total problems: {len(problem_set)}, Completed: {len(completed_problem_ids)}, Pending: {len(problems_list)}")

    # Start Judge container
    with LightCPVerifierJudge(worker=args.worker, instance_id=args.judge_instance_id) as judge:
        judge_url = judge.base_url
        logger.info(f"Judge service URL: {judge_url}")
        
        # Prepare task arguments
        task_args = [(p, config_path, judge_url) for p in problems_list]
        
        # Use Manager to share results
        manager = Manager()
        lock = manager.Lock()
        results = manager.list()
        
        # Add existing results to list
        for item in existing_results:
            # Keep only completed problem results (old results of redo problems will be discarded)
            if item.get("problem_id") in completed_problem_ids:
                results.append(item)
        
        total_problems = len(problem_set)
        
        if args.parallel > 1:
            # Multi-process mode
            logger.info(f"Starting {args.parallel} parallel processes...")
            
            with Pool(processes=args.parallel) as pool:
                for i, result in enumerate(pool.imap_unordered(process_single_problem, task_args)):
                    results.append(result)
                    # Save after each completion
                    save_results(list(results), result_filename, lock)
                    logger.info(f"Progress: {len(results)}/{total_problems} ({len(results)/total_problems*100:.1f}%)")
        else:
            # Single-process mode (compatible with existing behavior)
            import tqdm
            for task in tqdm.tqdm(task_args, desc="Benchmarking"):
                result = process_single_problem(task)
                results.append(result)
                save_results(list(results), result_filename, lock)

    # Reconstruct problem_set for statistics
    for r in results:
        if r['problem_id'] in problem_set:
            problem_set[r['problem_id']].text_response = r.get('text_response')
            problem_set[r['problem_id']].code = r.get('code')
            problem_set[r['problem_id']].judge_result = r.get('judge_result', 'Judging')
            problem_set[r['problem_id']].response_meta = r.get('response_meta')

    # Print statistics
    print_stats(dataset, problem_set)
    
    # 📊 Print Token statistics summary
    total_prompt_tokens = sum(r.get('total_prompt_tokens', 0) for r in results)
    total_completion_tokens = sum(r.get('total_completion_tokens', 0) for r in results)
    total_tokens = total_prompt_tokens + total_completion_tokens
    logger.info(f"\n📊 Token Statistics Summary:")
    logger.info(f"   Total prompt tokens: {total_prompt_tokens:,}")
    logger.info(f"   Total completion tokens: {total_completion_tokens:,}")
    logger.info(f"   Total tokens: {total_tokens:,}")
    if len(results) > 0:
        logger.info(f"   Average tokens per problem: {total_tokens // len(results):,}")
    
    logger.info(f"\n✅ Benchmark completed! Results saved to: {result_filename}")


if __name__ == "__main__":
    main()

