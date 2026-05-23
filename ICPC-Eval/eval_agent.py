import datasets
from AgentCodeGenerator import AgentCodeGenerator
from SubmitProblem_local import LocalCodeSubmitter
import json
import os
import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
import argparse
import re

import litellm
litellm.drop_params = True

litellm.callbacks = []
litellm._async_success_callback = []
litellm._async_failure_callback = []

import smolagents.models as _smolagents_models
_original_supports_stop = _smolagents_models.supports_stop_parameter
def _patched_supports_stop_parameter(model_id: str) -> bool:
    model_name = model_id.split("/")[-1].lower()
    if re.match(r"gpt-5\.\d+", model_name):
        return False
    return _original_supports_stop(model_id)
_smolagents_models.supports_stop_parameter = _patched_supports_stop_parameter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logger(config_name=None, timestamp=None):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("eval_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if config_name and timestamp:
        log_filename = f"eval_agent_{config_name}_{timestamp}.log"
    elif config_name:
        log_filename = f"eval_agent_{config_name}.log"
    elif timestamp:
        log_filename = f"eval_agent_{timestamp}.log"
    else:
        log_filename = "eval_agent.log"
    
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_filename),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, log_filename


def process_problem(
    code_generator,
    submitter,
    problem_id,
    case,
    submit_lock,
    result_lock,
    logger,
    evaluation_report,
    passed_count,
    results_file,
    max_attempts,
    max_generate,
    stream,
    sleep,
    reasoning_history,
):
    thread_name = threading.current_thread().name

    logger.info(
        f"Thread {thread_name} - {case['id']}.{case['title']}: Evaluating problem"
    )

    problem_result = {
        "id": case["id"],
        "title": case["title"],
        "url": case.get("qoj_url"),
        "type": case["type"],
        "problem_len": len(case["description"]) if case.get("description") else 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "attempts": [],
        "eval_status": "Running",
        "source": case.get("source"),
        "year": case.get("year"),
        "problem_label": case.get("problem_label"),
        "tags": case.get("tags"),
    }
    with result_lock:
        update_evaluation_report(
            evaluation_report,
            problem_result,
            logger,
            results_file,
        )

    correct_info = ""
    final_status = "Failed"
    passed_this_problem = 0

    for gen_idx in range(max_generate):
        if gen_idx > 0:
            logger.info(
                f"Thread {thread_name} - {case['id']}.{case['title']}: Regenerating solution (attempt {gen_idx+1}/{max_generate})"
            )
            correct_info = ""

        for attempt_idx in range(max_attempts):
            current_attempt_num = gen_idx * max_attempts + attempt_idx + 1
            logger.info(
                f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: "
                f"Attempting to solve problem (generate {gen_idx+1}/{max_generate}, attempt {attempt_idx+1}/{max_attempts})"
            )

            attempt = {
                "attempt_number": current_attempt_num,
                "sample_tests": {},
                "submission": None,
            }

            code = ""
            if attempt_idx == 0:
                start_time = time.time()
                code, output, output_len, code_len, token_count = (
                    code_generator.generate_code(
                        case,
                        sleep=sleep,
                        stream=stream,
                        reasoning_history=reasoning_history,
                    )
                )
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Thread {thread_name} - Generated code in {elapsed_time:.2f} seconds"
                )
            else:
                start_time = time.time()
                code, output, output_len, code_len, token_count = (
                    code_generator.correct_code(
                        correct_info,
                        period,
                        sleep=sleep,
                        stream=stream,
                        reasoning_history=reasoning_history,
                    )
                )
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Thread {thread_name} - Corrected code in {elapsed_time:.2f} seconds"
                )

            attempt["output"] = output
            attempt["output_len"] = output_len
            attempt["code_len"] = code_len
            attempt["token_count"] = token_count
            attempt["code"] = code
            attempt["generation"] = gen_idx + 1

            period = "Sample"

            if "examples" in case and case["examples"]:
                logger.info(
                    f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Running sample tests..."
                )
                result, correct_info_sample = code_generator.check_examples(code, case)
                attempt["sample_tests"] = {
                    "status": result,
                    "details": correct_info_sample,
                }

                if result == "Accepted":
                    period = "Submit"
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Sample tests passed"
                    )
                else:
                    period = "Sample"
                    correct_info = correct_info_sample
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Sample tests failed: {result}"
                    )
                    if result == "Compilation Error":
                        logger.error(f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: {correct_info_sample}")
            else:
                logger.info(
                    f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: No sample tests, proceeding to submission"
                )
                period = "Submit"

            if period == "Submit":
                with submit_lock:
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: Submitting code..."
                    )
                    submission_result = submit_code_with_retry(
                        submitter,
                        case,
                        code,
                        thread_name,
                        logger,
                        attempt_idx,
                    )

                attempt["submission"] = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "details": submission_result,
                }

                if (
                    submission_result["success"]
                    and submission_result["verdict"] == "Accepted"
                ):
                    with result_lock:
                        passed_this_problem = 1
                    final_status = "Accepted"
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: ✅ ACCEPTED"
                    )
                else:
                    correct_info = submission_result["verdict"]
                    logger.info(
                        f"Thread {thread_name} - {case['id']}.{case['title']}.{current_attempt_num}: ❌ {submission_result['verdict']}"
                    )

            problem_result["attempts"].append(attempt)
            problem_result["final_status"] = final_status
            problem_result["eval_status"] = "Running"

            if final_status == "Accepted":
                problem_result["eval_status"] = "Completed"
                token_stats = code_generator.get_token_stats()
                problem_result["total_prompt_tokens"] = token_stats.get("total_prompt_tokens", 0)
                problem_result["total_completion_tokens"] = token_stats.get("total_completion_tokens", 0)
                with result_lock:
                    update_evaluation_report(
                        evaluation_report,
                        problem_result,
                        logger,
                        results_file,
                    )
                code_generator.end_problem(final_status="Accepted")
                conversation_id = getattr(code_generator, 'conversation_id', None)
                tracker_stats = getattr(code_generator, 'last_tracker_stats', {})
                return problem_result, passed_this_problem, conversation_id, tracker_stats

            with result_lock:
                update_evaluation_report(
                    evaluation_report,
                    problem_result,
                    logger,
                    results_file,
                )

            if period == "Submit" and final_status != "Accepted":
                pass
            elif period == "Sample" and result != "Accepted":
                pass
            else:
                break

        if final_status == "Accepted":
            break

    problem_result["eval_status"] = "Completed"
    token_stats = code_generator.get_token_stats()
    problem_result["total_prompt_tokens"] = token_stats.get("total_prompt_tokens", 0)
    problem_result["total_completion_tokens"] = token_stats.get("total_completion_tokens", 0)
    with result_lock:
        update_evaluation_report(
            evaluation_report,
            problem_result,
            logger,
            results_file,
        )
    code_generator.end_problem(final_status=final_status)
    conversation_id = getattr(code_generator, 'conversation_id', None)
    tracker_stats = getattr(code_generator, 'last_tracker_stats', {})
    return problem_result, passed_this_problem, conversation_id, tracker_stats


def submit_code_with_retry(submitter, case, code, thread_name, logger, i):
    submission_result = submitter.submit_code(
        problem_id=case["id"], code=code, problem_info=case, all_judge=True
    )

    logger.info(
        f"Thread {thread_name} - {case['id']}.{case['title']}.{i+1}: Submission result: {submission_result['verdict']}"
    )
    
    if submission_result['verdict'] == "Compilation Error" and 'message' in submission_result:
        logger.error(f"Thread {thread_name} - {case['id']}.{case['title']}.{i+1}: Compile error details:\n{submission_result['message']}")

    return submission_result


def update_evaluation_report(evaluation_report, problem_result, logger, results_file):
    existing_idx = -1
    for i, result in enumerate(evaluation_report["detailed_results"]):
        if result["id"] == problem_result["id"]:
            existing_idx = i
            break

    if existing_idx != -1:
        evaluation_report["detailed_results"][existing_idx] = problem_result
    else:
        insert_pos = 0
        while (
            insert_pos < len(evaluation_report["detailed_results"])
            and evaluation_report["detailed_results"][insert_pos]["id"]
            < problem_result["id"]
        ):
            insert_pos += 1
        evaluation_report["detailed_results"].insert(insert_pos, problem_result)

    running_problems = 0
    completed_problems = 0
    passed = 0
    for result in evaluation_report["detailed_results"]:
        if result["eval_status"] == "Running":
            running_problems += 1
        elif result["eval_status"] == "Completed":
            completed_problems += 1
            if result["final_status"] == "Accepted":
                passed += 1

    evaluation_report["summary"]["evaluated_problems"] = completed_problems
    evaluation_report["summary"]["running_problems"] = running_problems
    evaluation_report["summary"]["passed_problems"] = passed

    denominator = evaluation_report["summary"]["evaluated_problems"]
    if denominator > 0:
        evaluation_report["summary"]["success_rate"] = f"{passed/denominator*100:.2f}%"
    else:
        evaluation_report["summary"]["success_rate"] = "0.00%"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=4)

    logger.info(f"\nCurrent progress: Evaluated {completed_problems}, Passed {passed}, "
                f"Success rate {evaluation_report['summary']['success_rate']}")


def load_evaluation_report(total_problems_in_dataset, logger, results_file):
    passed = 0
    completed_results = []

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            try:
                evaluation_report = json.load(f)
                if not isinstance(evaluation_report.get("detailed_results"), list):
                    raise ValueError("Invalid report format")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load report: {e}, creating new report")
                evaluation_report = None

        if evaluation_report:
            for result in evaluation_report.get("detailed_results", []):
                if result.get("eval_status") == "Completed":
                    completed_results.append(result)
                    if result.get("final_status") == "Accepted":
                        passed += 1

            evaluation_report["detailed_results"] = completed_results
            evaluation_report["summary"]["total_problems"] = total_problems_in_dataset
            evaluation_report["summary"]["evaluated_problems"] = len(completed_results)
            evaluation_report["summary"]["passed_problems"] = passed
            evaluation_report["summary"]["running_problems"] = 0

            logger.info(f"Loaded existing report: Evaluated {len(completed_results)}, Passed {passed}")
        else:
            evaluation_report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_problems": total_problems_in_dataset,
                    "passed_problems": 0,
                    "evaluated_problems": 0,
                    "running_problems": 0,
                    "success_rate": "0.00%",
                },
                "detailed_results": [],
            }
    else:
        evaluation_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_problems": total_problems_in_dataset,
                "passed_problems": 0,
                "evaluated_problems": 0,
                "running_problems": 0,
                "success_rate": "0.00%",
            },
            "detailed_results": [],
        }
        logger.info(f"Creating new evaluation report")

    evaluation_report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=4)

    return evaluation_report, passed, completed_results


def evaluate_test_cases(
    num_threads,
    model,
    base_url,
    api_key,
    timeout,
    dataset_name,
    dataset_config_name,
    dataset_split,
    results_file,
    max_attempts,
    max_generate,
    stream,
    sleep,
    reasoning_history,
    auto_choose,
    problem_ids_to_run,
    skip_ids_list,
    tools_config,
    unit_test_cases_on,
    enable_attempt_compression=False,
    config_name=None,
    timestamp=None,
):
    logger, log_filename = setup_logger(config_name=config_name, timestamp=timestamp)
    logger.info(f"Log file: logs/{log_filename}")

    logger.info("=" * 60)
    logger.info("CP-Agent Agent ICPC-Eval Evaluation Started")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Threads: {num_threads}")
    logger.info(f"Max attempts: {max_attempts}")
    logger.info(f"Max regenerate: {max_generate}")
    logger.info(f"Tools config: {tools_config}")
    logger.info(f"Context compression: {'Enabled' if enable_attempt_compression else 'Disabled'}")

    submitter = LocalCodeSubmitter()

    try:
        logger.info(f"Loading dataset '{dataset_name}'...")
        loaded_dataset = datasets.load_dataset(
            dataset_name, name=dataset_config_name, split=dataset_split
        )
        all_problems_list = list(loaded_dataset)
        logger.info(f"Successfully loaded {len(all_problems_list)} problems")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 0

    total_problems_in_current_run = len(all_problems_list)

    evaluation_report, initial_passed_count, completed_results = load_evaluation_report(
        total_problems_in_current_run, logger, results_file
    )

    current_run_passed_count = 0

    submit_lock = threading.Lock()
    result_lock = threading.Lock()
    task_queue = queue.Queue()

    problems_for_this_run = []
    skip_ids_set = set(skip_ids_list) if skip_ids_list else set()
    if skip_ids_set:
        logger.info(f"Will skip problem IDs: {sorted(skip_ids_set)}")
    
    if auto_choose:
        already_evaluated_ids = {result["id"] for result in completed_results}
        for case in reversed(all_problems_list):
            if case["id"] not in already_evaluated_ids and case["id"] not in skip_ids_set:
                problems_for_this_run.append(case)
        logger.info(f"Auto-select mode: Will evaluate {len(problems_for_this_run)} new problems")
    else:
        if problem_ids_to_run:
            cases_by_id_map = {case["id"]: case for case in all_problems_list}
            for pid in problem_ids_to_run:
                if pid in skip_ids_set:
                    logger.info(f"Problem ID {pid} is in skip list, skipped")
                    continue
                if pid in cases_by_id_map:
                    problems_for_this_run.append(cases_by_id_map[pid])
                else:
                    logger.warning(f"Problem ID {pid} does not exist")
            logger.info(f"Manual select mode: Will evaluate {len(problems_for_this_run)} specified problems")
        else:
            logger.info("No problems specified, nothing to evaluate")

    for case in problems_for_this_run:
        task_queue.put(case)

    this_run_conversation_ids = []
    
    total_generated_cases = 0
    total_generated_cases_improved = 0

    if task_queue.empty():
        logger.info("No problems to evaluate")
    else:
        max_workers = min(num_threads, task_queue.qsize())
        logger.info(f"Starting {max_workers} worker threads")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_to_case_id = {}

            for _ in range(max_workers):
                if not task_queue.empty():
                    case_to_process = task_queue.get_nowait()
                    logger.info(f"Creating AgentCodeGenerator with model={model}")
                    code_generator = AgentCodeGenerator(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        timeout=timeout,
                        tools_config=tools_config,
                        unit_test_cases_on=unit_test_cases_on,
                        enable_attempt_compression=enable_attempt_compression,
                    )
                    future = executor.submit(
                        process_problem,
                        code_generator,
                        submitter,
                        case_to_process["id"],
                        case_to_process,
                        submit_lock,
                        result_lock,
                        logger,
                        evaluation_report,
                        None,
                        results_file,
                        max_attempts,
                        max_generate,
                        stream,
                        sleep,
                        reasoning_history,
                    )
                    futures_to_case_id[future] = case_to_process["id"]

            while futures_to_case_id:
                done, _ = concurrent.futures.wait(
                    futures_to_case_id.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    case_id_processed = futures_to_case_id.pop(future)
                    try:
                        _, problem_passed_status, conv_id, tracker_stats = future.result()
                        if problem_passed_status > 0:
                            current_run_passed_count += 1
                        if conv_id:
                            this_run_conversation_ids.append(conv_id)
                        if tracker_stats and tracker_stats.get("has_generated_cases"):
                            total_generated_cases += 1
                            if tracker_stats.get("generated_cases_improved"):
                                total_generated_cases_improved += 1
                    except Exception as e:
                        logger.error(f"Problem {case_id_processed} evaluation failed: {e}")

                    if not task_queue.empty():
                        case_to_process = task_queue.get_nowait()
                        code_generator = AgentCodeGenerator(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            timeout=timeout,
                            tools_config=tools_config,
                            unit_test_cases_on=unit_test_cases_on,
                            enable_attempt_compression=enable_attempt_compression,
                        )
                        new_future = executor.submit(
                            process_problem,
                            code_generator,
                            submitter,
                            case_to_process["id"],
                            case_to_process,
                            submit_lock,
                            result_lock,
                            logger,
                            evaluation_report,
                            None,
                            results_file,
                            max_attempts,
                            max_generate,
                            stream,
                            sleep,
                            reasoning_history,
                        )
                        futures_to_case_id[new_future] = case_to_process["id"]

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            final_report = json.load(f)
        summary = final_report.get("summary", {})
        logger.info(f"Total problems: {summary.get('total_problems', 'N/A')}")
        logger.info(f"Evaluated: {summary.get('evaluated_problems', 'N/A')}")
        logger.info(f"Passed: {summary.get('passed_problems', 'N/A')}")
        logger.info(f"Success rate: {summary.get('success_rate', 'N/A')}")

    if unit_test_cases_on:
        try:
            from agentflow import config as agent_config
            work_dir_base = agent_config.WORK_DIR
            if not os.path.isabs(work_dir_base):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                work_dir_base = os.path.join(project_root, work_dir_base)
            
            logger.info(f"Collected conversation_ids count: {len(this_run_conversation_ids)}")
            if this_run_conversation_ids:
                logger.info(f"First 5 conversation_ids: {this_run_conversation_ids[:5]}")
            
            all_pass_rates = []
            all_total_inputs = 0
            all_consensus_reached = 0
            
            for session_name in this_run_conversation_ids:
                stats_path = os.path.join(work_dir_base, session_name, "unit_test_data", "majority_vote_stats.json")
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path, 'r', encoding='utf-8') as f:
                            stats = json.load(f)
                        pass_rate = stats.get('pass_rate', 0.0)
                        total_inputs = stats.get('total_inputs', 0)
                        consensus_reached = stats.get('consensus_reached', 0)
                        
                        all_pass_rates.append(pass_rate)
                        all_total_inputs += total_inputs
                        all_consensus_reached += consensus_reached
                    except Exception as e:
                        logger.warning(f"Failed to read {stats_path}: {e}")
            
            if all_pass_rates:
                avg_pass_rate = sum(all_pass_rates) / len(all_pass_rates)
                overall_pass_rate = all_consensus_reached / all_total_inputs if all_total_inputs > 0 else 0.0
                
                logger.info("\n" + "-" * 40)
                logger.info("Test case generation statistics for this run")
                logger.info("-" * 40)
                total_problems = len(this_run_conversation_ids)
                problems_with_testcases = len(all_pass_rates)
                generation_rate = problems_with_testcases / total_problems if total_problems > 0 else 0.0
                logger.info(f"Problems in this run: {total_problems}")
                logger.info(f"Problems with generated test cases: {problems_with_testcases}")
                logger.info(f"Test case generation success rate: {generation_rate:.2%} ({problems_with_testcases}/{total_problems})")
                logger.info(f"Total test inputs: {all_total_inputs}")
                logger.info(f"Consensus reached: {all_consensus_reached}")
                logger.info(f"Average pass rate: {avg_pass_rate:.2%}")
                logger.info(f"Overall pass rate: {overall_pass_rate:.2%} ({all_consensus_reached}/{all_total_inputs})")
                
                if os.path.exists(results_file):
                    with open(results_file, "r", encoding="utf-8") as f:
                        final_report = json.load(f)
                    final_report["testcase_generation_summary"] = {
                        "total_problems": total_problems,
                        "problems_with_testcases": problems_with_testcases,
                        "generation_success_rate": round(generation_rate, 4),
                        "total_inputs": all_total_inputs,
                        "consensus_reached": all_consensus_reached,
                        "average_pass_rate": round(avg_pass_rate, 4),
                        "overall_pass_rate": round(overall_pass_rate, 4)
                    }
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(final_report, f, ensure_ascii=False, indent=4)
            else:
                logger.info("\nNo test case generation statistics found")
                logger.info("   Possible reasons: All brute-force solutions failed, or test case generation incomplete")
        except Exception as e:
            logger.warning(f"Failed to summarize test case statistics: {e}")
    
    summary_msg = "\n" + "=" * 60 + "\n"
    summary_msg += "Generated test case improvement statistics summary (CppValidationTool actual usage count)\n"
    summary_msg += "=" * 60 + "\n"
    
    if total_generated_cases > 0:
        improvement_rate = (total_generated_cases_improved / total_generated_cases) * 100
        summary_msg += f"Total generated case runs: {total_generated_cases}\n"
        summary_msg += f"Improved cases (failed -> success): {total_generated_cases_improved}\n"
        summary_msg += f"Improvement rate: {improvement_rate:.2f}%\n"
    else:
        summary_msg += "No generated test cases or related statistics in this run.\n"
        
    summary_msg += "=" * 60 + "\n"
    
    logger.info(summary_msg)
    
    if os.path.exists(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                final_report = json.load(f)
            final_report["generated_cases_improvement_summary"] = {
                "total_generated_cases": total_generated_cases,
                "total_generated_cases_improved": total_generated_cases_improved,
                "improvement_rate": round(improvement_rate, 4) if total_generated_cases > 0 else 0.0
            }
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(final_report, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save generated test case improvement statistics: {e}")

    return current_run_passed_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CP-Agent on ICPC-Eval benchmark"
    )
    
    parser.add_argument("--num_threads", type=int, default=1, help="Number of parallel threads")
    parser.add_argument("--model", type=str, default=None, help="Model name (defaults to config)")
    parser.add_argument("--base_url", type=str, default=None, help="API base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--timeout", type=int, default=10000, help="API timeout in seconds")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RUC-AIBOX/ICPC-Eval",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset configuration name (optional)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split (e.g., test, train)",
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default="eval_results_agent.json",
        help="Evaluation results output file",
    )

    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum correction attempts per generation (refine@5)",
    )
    parser.add_argument(
        "--max_generate",
        type=int,
        default=1,
        help="Maximum regeneration count per problem",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming output (not supported by Agent, will be ignored)",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Wait time after code generation in seconds",
    )
    parser.add_argument(
        "--reasoning_history",
        action="store_true",
        help="Use reasoning history",
    )

    parser.add_argument(
        "--auto_choose",
        action="store_true",
        help="Automatically select unevaluated problems",
    )
    parser.add_argument(
        "--problem_ids",
        type=int,
        nargs="*",
        help="List of problem IDs to evaluate",
    )
    parser.add_argument(
        "--skip_ids",
        type=int,
        nargs="*",
        help="List of problem IDs to skip",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Agent configuration file path (YAML)",
    )
    parser.add_argument(
        "--tools_config",
        type=str,
        default=None,
        help="Tools configuration (e.g., cpp_validation,cpp_interpreter,luogu_retrieval)",
    )
    parser.add_argument(
        "--unit_test_cases_on",
        action="store_true",
        help="Enable test case generation feature",
    )
    parser.add_argument(
        "--enable_attempt_compression",
        action="store_true",
        help="Enable context compression (compress history attempts to reduce token usage)",
    )
    
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Resume from a previous results file (automatically enables auto_choose mode)",
    )

    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    resume_mode = False
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Resume file does not exist: {args.resume}")
            return
        
        resume_mode = True
        args.results_file = args.resume
        args.auto_choose = True
        
        resume_filename = os.path.basename(args.resume)
        if resume_filename.startswith("eval_results_") and resume_filename.endswith(".json"):
            middle_part = resume_filename[len("eval_results_"):-len(".json")]
            parts = middle_part.rsplit("_", 2)
            if len(parts) >= 3:
                try:
                    datetime.strptime(parts[-2] + "_" + parts[-1], "%Y%m%d_%H%M%S")
                    extracted_config_name = "_".join(parts[:-2])
                    if not args.config and extracted_config_name:
                        possible_config_paths = [
                            f"./agentflow/configs/ds-chat/{extracted_config_name}.yaml",
                            f"./agentflow/configs/{extracted_config_name}.yaml",
                        ]
                        for config_path in possible_config_paths:
                            if os.path.exists(config_path):
                                args.config = config_path
                                print(f"Resume mode: Auto-detected config file: {config_path}")
                                break
                except ValueError:
                    pass
        
        print(f"Resume mode enabled")
        print(f"Resume file: {args.resume}")
    
    config_name = None
    if args.config:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    if args.config:
        from agentflow import config
        print(f"Using config file: {args.config}")
        print(f"Config name: {config_name}")
        
        env_vars_to_clear = ['MODEL_ID', 'API_BASE', 'DEEPSEEK_API_KEY']
        cleared_vars = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                cleared_vars[var] = os.environ[var]
                del os.environ[var]
        
        if cleared_vars:
            print(f"Cleared environment variables that may override config: {list(cleared_vars.keys())}")
        
        config.reload_config(args.config)
        
        print(f"Config loaded:")
        print(f"   MODEL_ID: {config.MODEL_ID}")
        print(f"   API_BASE: {config.API_BASE}")
    
    from agentflow import config as agent_config
    
    model = args.model or agent_config.MODEL_ID
    base_url = args.base_url or agent_config.API_BASE
    
    unit_test_cases_on = args.unit_test_cases_on or getattr(agent_config, 'UNIT_TEST_CASES_ON', False)
    print(f"Test case generation: {'Enabled' if unit_test_cases_on else 'Disabled'} (CLI: {args.unit_test_cases_on}, Config: {getattr(agent_config, 'UNIT_TEST_CASES_ON', False)})")
    
    enable_attempt_compression = args.enable_attempt_compression or getattr(agent_config, 'ENABLE_ATTEMPT_COMPRESSION', False)
    print(f"Context compression: {'Enabled' if enable_attempt_compression else 'Disabled'} (CLI: {args.enable_attempt_compression}, Config: {getattr(agent_config, 'ENABLE_ATTEMPT_COMPRESSION', False)})")
    
    api_key = args.api_key or agent_config.API_KEY
    
    if not api_key:
        print("Error: Please provide --api_key or set API_KEY in config file")
        return
    
    results_dir = "./ICPC-Eval/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if not resume_mode:
        if args.results_file == "eval_results_agent.json":
            if config_name:
                results_filename = f"eval_results_{config_name}_{timestamp}.json"
            else:
                model_name_sanitized = model.replace("/", "_")
                results_filename = f"eval_results_agent_{model_name_sanitized}_{timestamp}.json"
            args.results_file = os.path.join(results_dir, results_filename)
        else:
            base_name = os.path.splitext(os.path.basename(args.results_file))[0]
            results_filename = f"{base_name}_{timestamp}.json"
            args.results_file = os.path.join(results_dir, results_filename)
    
    if config_name:
        log_filename_preview = f"eval_agent_{config_name}_{timestamp}.log"
    else:
        log_filename_preview = f"eval_agent_{timestamp}.log"
    
    if resume_mode:
        print(f"Resume results will be saved to: {args.results_file}")
    else:
        print(f"Results will be saved to: {args.results_file}")
    print(f"Log will be saved to: logs/{log_filename_preview}")

    evaluate_test_cases(
        num_threads=args.num_threads,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=args.timeout,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        results_file=args.results_file,
        max_attempts=args.max_attempts,
        max_generate=args.max_generate,
        stream=args.stream,
        sleep=args.sleep,
        reasoning_history=args.reasoning_history,
        auto_choose=args.auto_choose,
        problem_ids_to_run=args.problem_ids,
        skip_ids_list=args.skip_ids,
        tools_config=args.tools_config,
        unit_test_cases_on=unit_test_cases_on,
        enable_attempt_compression=enable_attempt_compression,
        config_name=config_name,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
