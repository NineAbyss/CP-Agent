from datasets import load_dataset, DatasetDict
import pydantic
import typing
import api_interface
import json
import tqdm
import time
import logging
import os
from datetime import datetime
from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code
import argparse
import sys

# change this to the number of workers you want to use in LightCPVerifier
# recommended to be <= number of CPU physical cores
worker = 8


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkResult(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    text_response: str
    code: str | None
    judge_result: str
    response_meta: typing.Any

class ProblemTestState(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    problem_statement: str
    text_response: str | None = None
    code: str | None = None
    submission_id: int | None = None
    judge_result: str = "Judging"
    response_meta: typing.Any = None

def get_problem_set(dataset: DatasetDict) -> dict[str, ProblemTestState]:
    problem_set = {}
    for split in dataset.values():
        for row in split:
            if row["problem_id"] not in problem_set:
                problem_set[row["problem_id"]] = ProblemTestState(**row)
    return problem_set

def print_stats(dataset: DatasetDict, problem_set: dict[str, ProblemTestState]):
    print("=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)

    split_difficulty_stats = {}

    for split_name, split in dataset.items():
        split_difficulty_stats[split_name] = {}
        
        for row in split:
            problem_id = row["problem_id"]
            difficulty = row.get("difficulty", "unknown")

            if problem_id in problem_set:
                judge_result = problem_set[problem_id].judge_result
            else:
                judge_result = "Not Tested"

            if difficulty not in split_difficulty_stats[split_name]:
                split_difficulty_stats[split_name][difficulty] = {
                    "total": 0, 
                    "accepted": 0, 
                    "judge_results": {}
                }
            
            split_difficulty_stats[split_name][difficulty]["total"] += 1
            if judge_result == "Accepted":
                split_difficulty_stats[split_name][difficulty]["accepted"] += 1

            if judge_result not in split_difficulty_stats[split_name][difficulty]["judge_results"]:
                split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result] = []
            split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result].append(problem_id)

    for split_name in split_difficulty_stats:
        print(f"\n[SPLIT: {split_name.upper()}]")
        print("-" * 60)
        
        total_problems_in_split = 0
        total_accepted_in_split = 0
        
        for difficulty, stats in sorted(split_difficulty_stats[split_name].items()):
            total = stats["total"]
            accepted = stats["accepted"]
            accuracy = (accepted / total * 100) if total > 0 else 0.0
            
            print(f"\n{difficulty.upper()} Difficulty: {accepted}/{total} ({accuracy:.1f}%)")

            for judge_result, problem_ids in sorted(stats["judge_results"].items()):
                count = len(problem_ids)
                percentage = (count / total * 100) if total > 0 else 0.0
                print(f"  {judge_result:20s}: {count:3d} ({percentage:5.1f}%) - {', '.join(sorted(problem_ids))}")
            
            total_problems_in_split += total
            total_accepted_in_split += accepted

        overall_accuracy = (total_accepted_in_split / total_problems_in_split * 100) if total_problems_in_split > 0 else 0.0
        print(f"\nOVERALL for {split_name}: {total_accepted_in_split}/{total_problems_in_split} ({overall_accuracy:.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark on LiveCodeBench-Pro")
    parser.add_argument("--model", default="deepseek-reasoner", 
                        choices=["deepseek-chat", "deepseek-reasoner"],
                        help="DeepSeek model to use (deepseek-chat: 8192 tokens, deepseek-reasoner: 65536 tokens)")
    parser.add_argument("--split", nargs="+", help="Dataset splits to benchmark (e.g., quater_2024_10_12)")
    parser.add_argument("--difficulty", nargs="+", help="Difficulty levels to benchmark (e.g., easy medium hard)")
    args = parser.parse_args()

    # Create LLM instance based on model argument
    llm_instance = api_interface.DeepSeekLLM(model=args.model)
    logger.info(f"Using model: {llm_instance.name} (max_tokens: {llm_instance.max_tokens})")

    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")

    # Create results folder
    os.makedirs("results", exist_ok=True)

    # Filter by split
    if args.split:
        available_splits = list(dataset.keys())
        selected_splits = []
        for s in args.split:
            if s in available_splits:
                selected_splits.append(s)
            else:
                logger.warning(f"Split '{s}' not found. Available: {available_splits}")
        
        if not selected_splits:
            logger.error("No valid splits selected.")
            sys.exit(1)
            
        dataset = DatasetDict({k: dataset[k] for k in selected_splits})

    # Filter by difficulty
    if args.difficulty:
        valid_difficulties = {"easy", "medium", "hard"}
        selected_difficulties = set(args.difficulty)
        
        # Validate difficulties
        if not selected_difficulties.issubset(valid_difficulties):
             logger.warning(f"Unknown difficulties found: {selected_difficulties - valid_difficulties}. Valid: {valid_difficulties}")

        # Filter each split
        new_splits = {}
        for split_name, split_data in dataset.items():
            filtered_split = split_data.filter(lambda x: x["difficulty"] in selected_difficulties)
            if len(filtered_split) > 0:
                new_splits[split_name] = filtered_split
            else:
                logger.warning(f"No problems found with difficulty {selected_difficulties} in split {split_name}")
        
        if not new_splits:
            logger.error("No problems found with selected difficulties in selected splits.")
            sys.exit(1)
            
        dataset = DatasetDict(new_splits)

    problem_set = get_problem_set(dataset)

    # Generate result filename: {model}_{splits}_{difficulties}_{timestamp}.json
    model_name = llm_instance.name
    splits_str = "_".join(sorted(dataset.keys())) if args.split else "all"
    difficulty_str = "_".join(sorted(args.difficulty)) if args.difficulty else "all"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"results/{model_name}_{splits_str}_{difficulty_str}_{timestamp}.json"

    with LightCPVerifierJudge(worker=worker) as judge:
        # Process problems serially to save results immediately
        for problem in tqdm.tqdm(problem_set.values(), desc="Benchmarking"):
            # 1. Generate Solution
            try:
                response, meta = llm_instance.generate_solution(problem.problem_statement)
                problem.text_response = response
                problem.code = extract_longest_cpp_code(response)
                problem.response_meta = meta
            except Exception as e:
                logger.error(f"Error generating solution for {problem.problem_id}: {e}")
                problem.text_response = f"Error: {e}"
                continue

            # 2. Submit and Judge
            if problem.code:
                try:
                    problem.submission_id = judge.submit(problem.problem_id, SupportedLanguage.CPP, problem.code)
                    
                    # Wait for result immediately
                    while True:
                        res = judge.get_result(problem.submission_id)
                        if res != "Judging":
                            problem.judge_result = res
                            break
                        time.sleep(0.5)
                except ProblemNotFoundError:
                    logger.warning(f"Problem {problem.problem_id} not found in judge dataset.")
                    problem.judge_result = "Problem Not Found"
                except Exception as e:
                    logger.error(f"Error submitting problem {problem.problem_id}: {e}")
                    problem.judge_result = "Submission Error"
            else:
                problem.judge_result = "No Code Extracted"

            # 3. Save Results Immediately
            results = []
            for p in problem_set.values():
                # Only save problems that have been attempted (have a response)
                if p.text_response is not None:
                    results.append(BenchmarkResult(**p.model_dump()).model_dump())
            
            with open(result_filename, "w") as f:
                json.dump(results, f, indent=4)
    


    print_stats(dataset, problem_set)
