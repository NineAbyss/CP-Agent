#!/usr/bin/env python3
"""
Run benchmark of agentflow agent on LiveCodeBench-Pro

Usage:
    # Use default config (ds-chat_reflection_only)
    python run_agentflow_benchmark.py
    
    # Specify config file
    python run_agentflow_benchmark.py --config agentflow/configs/ds-chat/ds-chat_reflection_and_testcase.yaml
    
    # Specify dataset split and difficulty
    python run_agentflow_benchmark.py --split quater_2025_4_6 --difficulty easy medium
    
    # Full example
    python run_agentflow_benchmark.py \
        --config agentflow/configs/ds-chat/ds-chat_reflection_and_testcase_and_exp.yaml \
        --split quater_2025_4_6 \
        --difficulty easy medium hard
    
    # Resume from existing result file
    python run_agentflow_benchmark.py --resume results/xxx.json
"""

import argparse
import json
import os
import sys
import tqdm
import time
import logging
from datetime import datetime
from datasets import load_dataset, DatasetDict

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code
from agentflow_adapter import AgentflowLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models and stats functions from benchmark.py
from benchmark import BenchmarkResult, ProblemTestState, get_problem_set, print_stats

# Judge worker count
WORKER = 8


def main():
    parser = argparse.ArgumentParser(
        description="Run agentflow agent benchmark on LiveCodeBench-Pro"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="agentflow config file path (e.g., agentflow/configs/ds-chat/ds-chat_reflection_only.yaml)"
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
        help="Dataset split (e.g., quater_2025_4_6)"
    )
    parser.add_argument(
        "--difficulty", 
        nargs="+", 
        help="Difficulty level (e.g., easy medium hard)"
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
        help="Result output directory (default: results)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing result file path (e.g., results/xxx.json)"
    )
    args = parser.parse_args()

    # Create AgentflowLLM instance
    if args.config:
        llm_instance = AgentflowLLM(config_path=args.config)
    else:
        from agentflow_adapter import create_agentflow_llm
        llm_instance = create_agentflow_llm(args.config_name)
    
    logger.info(f"Using agent: {llm_instance.name}")

    # Load dataset
    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter by split
    if args.split:
        available_splits = list(dataset.keys())
        selected_splits = []
        for s in args.split:
            if s in available_splits:
                selected_splits.append(s)
            else:
                logger.warning(f"Split '{s}' does not exist. Available: {available_splits}")
        
        if not selected_splits:
            logger.error("No valid splits found")
            sys.exit(1)
            
        dataset = DatasetDict({k: dataset[k] for k in selected_splits})

    # Filter by difficulty
    if args.difficulty:
        valid_difficulties = {"easy", "medium", "hard"}
        selected_difficulties = set(args.difficulty)
        
        if not selected_difficulties.issubset(valid_difficulties):
            logger.warning(f"Unknown difficulties: {selected_difficulties - valid_difficulties}")

        new_splits = {}
        for split_name, split_data in dataset.items():
            filtered_split = split_data.filter(lambda x: x["difficulty"] in selected_difficulties)
            if len(filtered_split) > 0:
                new_splits[split_name] = filtered_split
            else:
                logger.warning(f"Split {split_name} has no problems with difficulty {selected_difficulties}")
        
        if not new_splits:
            logger.error("No problems found after filtering")
            sys.exit(1)
            
        dataset = DatasetDict(new_splits)

    # Get problem set
    problem_set = get_problem_set(dataset)
    
    # Handle resume logic
    completed_problem_ids = set()
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Resume file does not exist: {args.resume}")
            sys.exit(1)
        
        try:
            with open(args.resume, "r") as f:
                existing_results = json.load(f)
            
            # Extract completed problem IDs from existing results
            for item in existing_results:
                problem_id = item.get("problem_id")
                if problem_id:
                    completed_problem_ids.add(problem_id)
                    # Restore completed problem states
                    if problem_id in problem_set:
                        problem_set[problem_id].text_response = item.get("text_response")
                        problem_set[problem_id].code = item.get("code")
                        problem_set[problem_id].judge_result = item.get("judge_result")
                        problem_set[problem_id].response_meta = item.get("response_meta")
            
            logger.info(f"Restored {len(completed_problem_ids)} completed problems from {args.resume}")
            result_filename = args.resume  # Continue writing to the same file
        except Exception as e:
            logger.error(f"Failed to read resume file: {e}")
            sys.exit(1)
    else:
        # Generate new result filename
        model_name = llm_instance.name
        splits_str = "_".join(sorted(dataset.keys())) if args.split else "all"
        difficulty_str = "_".join(sorted(args.difficulty)) if args.difficulty else "all"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"{args.output_dir}/{model_name}_{splits_str}_{difficulty_str}_{timestamp}.json"
    
    # Count pending problems
    pending_problems = [p for p in problem_set.values() if p.problem_id not in completed_problem_ids]
    
    logger.info(f"Results will be saved to: {result_filename}")
    logger.info(f"Total problems: {len(problem_set)}, Completed: {len(completed_problem_ids)}, Pending: {len(pending_problems)}")

    # Run benchmark
    with LightCPVerifierJudge(worker=args.worker) as judge:
        for problem in tqdm.tqdm(pending_problems, desc="Benchmarking"):
            # Skip completed problems
            if problem.problem_id in completed_problem_ids:
                continue
            
            # 1. Generate solution
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing problem: {problem.problem_id} - {problem.problem_title}")
                logger.info(f"Difficulty: {problem.difficulty}")
                logger.info(f"{'='*60}")
                
                response, meta = llm_instance.generate_solution(problem.problem_statement)
                problem.text_response = response
                problem.code = extract_longest_cpp_code(response)
                problem.response_meta = meta
                
            except Exception as e:
                logger.error(f"Failed to generate solution for {problem.problem_id}: {e}")
                problem.text_response = f"Error: {e}"
                import traceback
                traceback.print_exc()
                continue

            # 2. Submit for evaluation
            if problem.code:
                try:
                    problem.submission_id = judge.submit(
                        problem.problem_id, 
                        SupportedLanguage.CPP, 
                        problem.code
                    )
                    
                    # Wait for results
                    while True:
                        res = judge.get_result(problem.submission_id)
                        if res != "Judging":
                            problem.judge_result = res
                            break
                        time.sleep(0.5)
                    
                    logger.info(f"Evaluation result: {problem.judge_result}")
                    
                except ProblemNotFoundError:
                    logger.warning(f"Problem {problem.problem_id} does not exist in the evaluation dataset")
                    problem.judge_result = "Problem Not Found"
                except Exception as e:
                    logger.error(f"Failed to submit problem {problem.problem_id}: {e}")
                    problem.judge_result = "Submission Error"
            else:
                problem.judge_result = "No Code Extracted"
                logger.warning(f"Failed to extract code from response")

            # 3. Save results immediately
            results = []
            for p in problem_set.values():
                if p.text_response is not None:
                    results.append(BenchmarkResult(**p.model_dump()).model_dump())
            
            with open(result_filename, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # Print statistics
    print_stats(dataset, problem_set)
    
    logger.info(f"\n✅ Benchmark completed! Results saved to: {result_filename}")


if __name__ == "__main__":
    main()


