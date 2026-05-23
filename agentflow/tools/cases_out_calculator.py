from __future__ import annotations

from smolagents import Tool
import subprocess
import os
import re
from typing import Dict, List, Optional
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, OrderedDict
from agentflow.tools.session_manager import get_spj_code

CASES_OUT_MAX_WORKERS = 8

class Cases_Out_Calculator(Tool):
    """
    Uses a precompiled brute-force C++ solution (bf_solution) and runs it on
    previously generated inputs in cases_in/. Produces golden_outputs.json
    and final_cases.json (filtered passing cases based on simple error/timeout
    heuristics). Returns a status message including pass rate.
    """

    name = "cases_out_calculator"
    description = """
The 'Cases_Out_Calculator' tool runs an existing, precompiled brute-force executable (bf_solution),
tests it on all inputs under the current session’s cases_in/ directory, and records the outputs as
golden_outputs.json. It also filters the passing test cases into final_cases.json, and converts them
into per-file input_<n>/output_<n> format.

Prerequisite: first run 'bf_solution_generator' to generate and compile bf_solution.cpp into the
bf_solution executable.

Outputs:
- A short string message containing the number of processed test cases and the artifact storage location.

    """
    inputs = {
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_session_id: Optional[str] = None
        self._output_dir: Optional[str] = None

    def _set_output_dir(self, conversation_id: str):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
            print("global_dir is None")
        if global_dir:
            unit_test_data_dir = os.path.join(global_dir, "unit_test_data")
        else:
            unit_test_data_dir = os.path.join(project_root, "tools", "cpp", "unit_test_data", conversation_id)
        os.makedirs(unit_test_data_dir, exist_ok=True)
        self._output_dir = unit_test_data_dir
        print(f"Set output directory to: {unit_test_data_dir}")
        return unit_test_data_dir

    def _set_session_id(self, session_id: str):
        self._current_session_id = session_id

    def _get_session_id(self) -> Optional[str]:
        return self._current_session_id

    def _generate_conversation_id(self, problem_text: str) -> str:
        timestamp = str(int(time.time() * 1000))
        problem_hash = re.sub(r"[^0-9a-f]", "", str(hash(problem_text)))[:8]
        return f"auto_{timestamp}_{problem_hash}"

    # Removed: generation and compilation responsibilities. This tool now assumes
    # a precompiled brute-force executable exists in the session output directory.

    def _run_solution_over_testcases(self, cases_in_dir: str, bf_solution_exec_path: str):
        """
        Run one solution executable over all inputs in cases_in/ and return
        a list of items in input index order:
        [
            {"input_path": str, "input": str, "output": str},
            ...
        ]
        """
        input_files = glob.glob(os.path.join(cases_in_dir, 'cases_in', '*_stdout.txt'))
        input_files = sorted(input_files)
        print(f"Extract input files from {cases_in_dir}: {input_files}")
        print(f"Number of files to read: {len(input_files)}")

        # Ensure the brute-force executable is runnable once up-front
        try:
            if not os.access(bf_solution_exec_path, os.X_OK):
                os.chmod(bf_solution_exec_path, 0o755)
        except Exception as e:
            print(f"Warning: could not chmod exec file '{bf_solution_exec_path}': {e}")

        def run_single_case(index: int, input_file: str):
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = f.read()
            except Exception as read_err:
                return index, {
                    "input_path": input_file,
                    "input": "",
                    "output": f"EXCEPTION: Failed to read input file '{input_file}': {read_err}"
                }

            try:
                import resource

                def limit_resources():
                    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024))
                    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))

                proc = subprocess.run(
                    [bf_solution_exec_path],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=90,  # seconds
                    preexec_fn=limit_resources, 
                    start_new_session=True,
                )
                output = proc.stdout
                if proc.returncode != 0:
                    error_parts = []
                    error_parts.append(f"Exit code: {proc.returncode}")
                    if proc.stderr and proc.stderr.strip():
                        error_parts.append(f"Stderr: {proc.stderr.strip()}")
                    if proc.stdout and proc.stdout.strip():
                        stdout_content = proc.stdout.strip()
                        if any(keyword in stdout_content.lower() for keyword in ['error', 'exception', 'fault', 'segmentation', 'abort', 'terminate']):
                            error_parts.append(f"Stdout: {stdout_content}")
                    if proc.returncode == -11:
                        error_parts.append("Error type: Segmentation fault (SIGSEGV)")
                    elif proc.returncode == -6:
                        error_parts.append("Error type: Abort signal (SIGABRT)")
                    elif proc.returncode == -9:
                        error_parts.append("Error type: Process killed (SIGKILL)")
                    elif proc.returncode == 1:
                        error_parts.append("Error type: General error (exit code 1)")
                    elif proc.returncode == 2:
                        error_parts.append("Error type: Misuse of shell builtins (exit code 2)")
                    elif proc.returncode == 126:
                        error_parts.append("Error type: Command invoked cannot execute (exit code 126)")
                    elif proc.returncode == 127:
                        error_parts.append("Error type: Command not found (exit code 127)")
                    error_msg = " | ".join(error_parts) if error_parts else f"Process exited with code {proc.returncode} (no additional error information available)"
                    try:
                        error_msg = str(error_msg)
                    except UnicodeDecodeError:
                        error_msg = f"Unicode decode error in error message (return code: {proc.returncode})"
                    return index, {
                        "input_path": input_file,
                        "input": input_data,
                        "output": f"ERROR: {error_msg}"
                    }
                else:
                    return index, {
                        "input_path": input_file,
                        "input": input_data,
                        "output": output
                    }
            except subprocess.TimeoutExpired as e:
                return index, {
                    "input_path": input_file,
                    "input": input_data,
                    "output": f"TIMEOUT: Process timed out after {e.timeout} seconds"
                }
            except Exception as e:
                return index, {
                    "input_path": input_file,
                    "input": input_data,
                    "output": f"EXCEPTION: {str(e)}"
                }

        if not input_files:
            return []

        # Determine parallelism level
        max_workers_env = os.getenv("CASES_OUT_MAX_WORKERS")
        try:
            parsed = int(max_workers_env) if max_workers_env is not None else None
        except ValueError:
            parsed = None
        cpu_count = os.cpu_count() or 4
        max_workers = parsed if parsed and parsed > 0 else min(cpu_count, len(input_files))
        if max_workers <= 0:
            max_workers = 1

        results: List[Optional[Dict[str, str]]] = [None] * len(input_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(run_single_case, idx, input_file): idx
                for idx, input_file in enumerate(input_files)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    index, item = future.result()
                except Exception as e:
                    index, item = idx, {"input_path": input_files[idx], "input": "", "output": f"EXCEPTION: {str(e)}"}
                results[index] = item

        ordered = [item for item in results if item is not None]
        return ordered

    def _discover_multiple_solutions(self) -> OrderedDict:
        """
        Discover multiple bf_solution_{i}.cpp and executables bf_solution_{i} under output_dir.
        Also optionally include bf_solution (no index) as solution_0 if present.
        Returns OrderedDict keyed by 'solution_i' with dict {path_to_cpp_file, path_to_exec_file}.
        """
        assert self._output_dir, "Output dir not set"
        dirp = self._output_dir
        cpp_files = glob.glob(os.path.join(dirp, 'bf_solution_*.cpp'))
        exec_candidates = set()
        solutions = {}

        # Indexed solutions
        for cpp in cpp_files:
            base = os.path.basename(cpp)
            m = re.match(r"bf_solution_(\d+)\.cpp$", base)
            if not m:
                continue
            idx = int(m.group(1))
            exec_path = os.path.join(dirp, f"bf_solution_{idx}")
            if os.path.exists(exec_path):
                solutions[idx] = {
                    "path_to_cpp_file": cpp,
                    "path_to_exec_file": exec_path
                }
                exec_candidates.add(exec_path)

        # Unindexed fallback
        exec_fallback = os.path.join(dirp, "bf_solution")
        if os.path.exists(exec_fallback):
            cpp_fallback = os.path.join(dirp, "bf_solution.cpp")
            # Choose index 0 if not already used
            idx0 = 0
            while idx0 in solutions:
                idx0 += 1
            solutions[idx0] = {
                "path_to_cpp_file": cpp_fallback if os.path.exists(cpp_fallback) else "",
                "path_to_exec_file": exec_fallback
            }

        # Build OrderedDict with stable ordering by index
        ordered = OrderedDict()
        for idx in sorted(solutions.keys()):
            ordered[f"solution_{idx}"] = solutions[idx]
        return ordered

    def _init_solution_exec_details(self, details_path: str, question: str, solution_set: OrderedDict):
        details = {
            "question": question,
            "solution_set": {
                key: {
                    "path_to_cpp_file": value.get("path_to_cpp_file", ""),
                    "path_to_exec_file": value.get("path_to_exec_file", "")
                } for key, value in solution_set.items()
            },
            "generated i/o": {}
        }
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    def _update_solution_exec_details(self, details_path: str, solution_key: str, items: List[Dict[str, str]]):
        """
        Merge results of one solution into details file under 'generated i/o'.
        items is a list with fields: input_path, input, output in stable order.
        """
        try:
            with open(details_path, 'r', encoding='utf-8') as f:
                details = json.load(f)
        except Exception:
            details = {"question": "", "solution_set": {}, "generated i/o": {}}

        gen = details.setdefault("generated i/o", {})
        for i, item in enumerate(items, start=1):
            input_key = f"input_{i}"
            entry = gen.setdefault(input_key, {"input_path": item.get("input_path", ""), "output_set": {}})
            # Always record the first seen input_path
            if not entry.get("input_path"):
                entry["input_path"] = item.get("input_path", "")
            # Save output keyed by "output_from_<solution_key>"
            entry.setdefault("output_set", {})[f"output_from_{solution_key}"] = str(item.get("output", ""))

        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

        return len(items)

    def _compile_spj(self, spj_code: str) -> Optional[str]:
        """Compile SPJ code and return path to executable"""
        import tempfile
        import subprocess
        
        try:
            # Create a temporary directory for the SPJ
            temp_dir = tempfile.mkdtemp(prefix="spj_")
            spj_src = os.path.join(temp_dir, "spj.cpp")
            spj_exe = os.path.join(temp_dir, "spj.exe") if os.name == 'nt' else os.path.join(temp_dir, "spj")
            
            with open(spj_src, "w", encoding="utf-8") as f:
                f.write(spj_code)
            
            # Compile SPJ
            compile_cmd = ["g++", "-std=c++17", "-O2", spj_src, "-o", spj_exe]
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return spj_exe
            else:
                print(f"SPJ Compilation failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error compiling SPJ: {e}")
            return None

    def _run_spj(self, spj_executable: str, expected_output: str, actual_output: str) -> bool:
        """Run SPJ and return True if Accepted, False otherwise"""
        import tempfile
        import subprocess
        
        try:
            # Create temporary files for outputs
            with tempfile.NamedTemporaryFile(mode='w', suffix='.expected', delete=False) as f_exp:
                f_exp.write(expected_output)
                exp_path = f_exp.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.actual', delete=False) as f_act:
                f_act.write(actual_output)
                act_path = f_act.name
            
            try:
                # Run SPJ: spj_exe expected_output actual_output
                result = subprocess.run([spj_executable, exp_path, act_path], capture_output=True, text=True, timeout=10)
                
                # Check output for "Accepted" or return code 0
                is_accepted = "Accepted" in result.stdout or result.returncode == 0
                return is_accepted
            finally:
                # Cleanup temp files
                if os.path.exists(exp_path): os.remove(exp_path)
                if os.path.exists(act_path): os.remove(act_path)
        except Exception as e:
            print(f"Error running SPJ: {e}")
            return False

    def _compute_majority_final_cases(self, details_path: str, final_cases_path: str) -> int:
        """
        Read solution_exec_details.json and compute final_cases.

        Behavior:
        - Collect absolutely-correct cases from work_dir/test_data:
          - single pair: test_data/input & test_data/output  -> stored in 'basic_cases' with input_key "basic_input"
          - multiple pairs: test_data/input_{i} & test_data/output_{i} -> stored in 'basic_cases' with input_key "basic_input_{i}"
        - Then compute majority vote per input_n from 'generated i/o'. If chosen output contains 'error' (case-insensitive),
          discard that (input, output).
        - final output JSON at final_cases_path contains fields:
          {
            "question": "...",
            "solution_set": {...},
            "basic_cases": [...],
            "generated_cases": [...]
          }
        Returns number of generated_cases (majority-derived).
        """
        try:
            with open(details_path, 'r', encoding='utf-8') as f:
                details = json.load(f)
        except Exception as e:
            print(f"Error: cannot read {details_path}: {e}")
            with open(final_cases_path, 'w', encoding='utf-8') as f:
                json.dump({"question": "", "solution_set": {}, "basic_cases": [], "generated_cases": []}, f, indent=2, ensure_ascii=False)
            return 0

        basic_cases: List[Dict[str, str]] = []
        generated_cases: List[Dict[str, str]] = []

        # Collect absolutely-correct cases from work_dir/test_data
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None

        if global_dir:
            test_data_dir = os.path.join(global_dir, "test_data")
            # single pair: input / output -> key "basic_input"
            single_in = os.path.join(test_data_dir, "input")
            single_out = os.path.join(test_data_dir, "output")
            if os.path.exists(single_in) and os.path.exists(single_out):
                try:
                    with open(single_in, 'r', encoding='utf-8') as fi:
                        input_content = fi.read()
                except Exception:
                    input_content = ""
                try:
                    with open(single_out, 'r', encoding='utf-8') as fo:
                        output_content = fo.read()
                except Exception:
                    output_content = ""
                basic_cases.append({
                    "input_key": "basic_input",
                    "input_path": single_in,
                    "input": input_content,
                    "output": output_content
                })

            # multiple pairs: input_{i} / output_{i} -> keys "basic_input_{i}"
            if os.path.isdir(test_data_dir):
                indices = []
                try:
                    for name in os.listdir(test_data_dir):
                        m = re.fullmatch(r"input_(\d+)", name)
                        if not m:
                            continue
                        idx = int(m.group(1))
                        in_path = os.path.join(test_data_dir, f"input_{idx}")
                        out_path = os.path.join(test_data_dir, f"output_{idx}")
                        if os.path.exists(in_path) and os.path.exists(out_path):
                            indices.append(idx)
                except Exception:
                    indices = []
                for idx in sorted(indices):
                    in_path = os.path.join(test_data_dir, f"input_{idx}")
                    out_path = os.path.join(test_data_dir, f"output_{idx}")
                    try:
                        with open(in_path, 'r', encoding='utf-8') as fi:
                            input_content = fi.read()
                    except Exception:
                        input_content = ""
                    try:
                        with open(out_path, 'r', encoding='utf-8') as fo:
                            output_content = fo.read()
                    except Exception:
                        output_content = ""
                    basic_cases.append({
                        "input_key": f"basic_input_{idx}",
                        "input_path": in_path,
                        "input": input_content,
                        "output": output_content
                    })

        # Check for SPJ
        spj_code = get_spj_code()
        spj_executable = None
        if spj_code:
            print("Debug: SPJ code found for majority vote, compiling...")
            spj_executable = self._compile_spj(spj_code)

        # Majority vote from 'generated i/o'
        gen = details.get("generated i/o", {})
        for input_key in sorted(gen.keys(), key=lambda k: (k.startswith("input_"), int(k.split("_")[1]) if re.match(r"input_\d+$", k) else 10**9)):
            entry = gen[input_key]
            input_path = entry.get("input_path", "")
            output_set = entry.get("output_set", {})
            
            outputs = []
            for sol_key_full, status in output_set.items():
                if status == "SUCCESS":
                    sol_key = sol_key_full.replace("output_from_", "")
                    # input_key is "input_N"
                    m = re.match(r"input_(\d+)", input_key)
                    if m:
                        idx = m.group(1)
                        out_file = os.path.join(self._output_dir, "test_cases_output", sol_key, f"output_{idx}.txt")
                        if os.path.exists(out_file):
                            try:
                                with open(out_file, 'r', encoding='utf-8') as f:
                                    outputs.append(f.read())
                            except Exception as e:
                                print(f"Warning: failed to read {out_file}: {e}")
                elif status and "error" not in str(status).lower() and "timeout" not in str(status).lower():
                    outputs.append(str(status))
            
            if not outputs:
                continue
                
            total_successful = len(outputs)
            threshold = total_successful // 2 + 1
            # If multiple solutions were attempted, require at least 2 to agree for reliability
            if len(output_set) > 1:
                threshold = max(2, threshold)
            
            chosen_value = None
            
            # Try exact match first
            if not outputs:
                continue
            counter = Counter(outputs)
            value, count = counter.most_common(1)[0]
            if count >= threshold:
                chosen_value = value
            elif spj_executable:
                # Try SPJ-based majority
                unique_outputs = list(set(outputs))
                for candidate in unique_outputs:
                    spj_count = 0
                    for other in outputs:
                        if candidate == other or self._run_spj(spj_executable, candidate, other):
                            spj_count += 1
                    if spj_count >= threshold:
                        chosen_value = candidate
                        break
            
            if chosen_value is not None:
                value = chosen_value
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        input_content = f.read()
                except Exception:
                    input_content = ""
                generated_cases.append({
                    "input_key": input_key,
                    "input_path": input_path,
                    "input": input_content,
                    "output": value
                })

        out_obj = {
            "question": details.get("question", ""),
            "solution_set": details.get("solution_set", {}),
            "basic_cases": basic_cases,
            "generated_cases": generated_cases
        }
        try:
            with open(final_cases_path, 'w', encoding='utf-8') as f:
                json.dump(out_obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save final cases to {final_cases_path}: {e}")

        # Cleanup SPJ
        if spj_executable:
            try:
                import shutil
                shutil.rmtree(os.path.dirname(spj_executable))
            except: pass

        # Calculate statistics
        total_inputs = len(gen)
        consensus_reached = len(generated_cases)
        pass_rate = (consensus_reached / total_inputs) if total_inputs > 0 else 0.0
        
        # Get question ID from details or path
        question_id = "unknown"
        if "question" in details:
            # Try to extract from question text or just use "unknown"
            pass
        
        # Format stats message
        stats_msg = (
            f"[MajorityVoteStats] Problem: {self._current_session_id} | "
            f"Total Inputs: {total_inputs} | "
            f"Consensus Reached: {consensus_reached} | "
            f"Pass Rate: {pass_rate:.2%}"
        )
        print(stats_msg)
        
        # Write to majority_vote_stats.json
        stats_data = {
            "session_id": self._current_session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_inputs": total_inputs,
            "consensus_reached": consensus_reached,
            "pass_rate": pass_rate,
            "details": {
                "input_count": total_inputs,
                "consensus_count": consensus_reached
            }
        }
        try:
            stats_path = os.path.join(self._output_dir, 'majority_vote_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save majority vote stats: {e}")

        # Append to main.log (skip for LiveCodeBench-Pro which uses its own results directory)
        if os.getenv("LCB_PRO_MODE"):
            print(f"[LCB-Pro] Skipping main.log append (using results directory instead)")
        else:
            try:
                # Find project root
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                logs_dir = os.path.join(project_root, "logs")
                
                if os.path.exists(logs_dir):
                    # Find most recent subdirectory in logs
                    subdirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
                    if subdirs:
                        latest_subdir = max(subdirs, key=os.path.getmtime)
                        main_log_path = os.path.join(latest_subdir, "main.log")
                        
                        if os.path.exists(main_log_path):
                            # Append to main.log
                            with open(main_log_path, "a", encoding="utf-8") as f:
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - {stats_msg}\n")
                            print(f"Successfully appended stats to {main_log_path}")
                        else:
                            print(f"main.log not found in {latest_subdir}")
                    else:
                        print(f"No subdirectories found in {logs_dir}")
                else:
                    print(f"Logs directory not found at {logs_dir}")
            except Exception as e:
                print(f"Warning: Could not append to main.log: {e}")

        return len(generated_cases)

    # DEPRECATED: _generate_outputs_for_testcases is superseded by _run_solution_over_testcases in multi-solution mode.
    def _generate_outputs_for_testcases(self, cases_in_dir: str, bf_solution_exec_path: str, output_json_path: str, question: str, bf_solution: str):
        # Backward compatibility: keep old behavior if needed elsewhere.
        items = self._run_solution_over_testcases(cases_in_dir, bf_solution_exec_path)
        result_json = {
            "question": question,
            "bf_solution": bf_solution,
            "compiled_file": bf_solution_exec_path,
            "golden i/o": [
                {"input": it["input"], "output": it["output"]} for it in items
            ]
        }
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

    def _convert_golden_outputs_to_test_data(self, golden_outputs_json_path: str, conversation_id: str) -> str:
        """
        Convert either:
        - legacy golden_outputs.json with "golden i/o", or
        - new final_cases.json with "generated_cases"
        into formatted_io_cases/input_n and formatted_io_cases/output_n files.
        """
        try:
            with open(golden_outputs_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return f"Error reading golden outputs: {e}"

        items: List[Dict[str, str]] = []
        if isinstance(data, dict) and "generated_cases" in data:
            for item in data.get("generated_cases", []):
                if not isinstance(item, dict):
                    continue
                inp = str(item.get("input", ""))
                out = str(item.get("output", ""))
                items.append({"input": inp, "output": out})
        elif isinstance(data, dict):
            raw_items = data.get("golden i/o", [])
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                inp = str(item.get("input", ""))
                out = str(item.get("output", ""))
                items.append({"input": inp, "output": out})

        if not items:
            return "No <input, output> pairs found in outputs"

        # Prepare destination directory aligned with unit tests
        try:
            from agentflow.tools.session_manager import get_global_work_dir
            global_dir = get_global_work_dir()
        except Exception:
            global_dir = None
        if global_dir:
            test_data_dir = os.path.join(global_dir, "unit_test_data")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            test_data_dir = os.path.join(project_root, "tools", "cpp", "unit_test_data", conversation_id)
        os.makedirs(test_data_dir, exist_ok=True)

        # Remove any previous merged files if present (best-effort, ignore errors)
        for legacy_name in ("input", "output"):
            legacy_path = os.path.join(test_data_dir, legacy_name)
            try:
                if os.path.exists(legacy_path):
                    os.remove(legacy_path)
            except Exception:
                pass

        formatted_io_cases_dir = os.path.join(test_data_dir, "formatted_io_cases")
        try:
            os.makedirs(formatted_io_cases_dir, exist_ok=True)
        except Exception as e:
            return f"Error creating formatted_io_cases directory: {e}"

        written = 0
        for index, pair in enumerate(items, start=1):
            input_filename = os.path.join(formatted_io_cases_dir, f"input_{index}")
            output_filename = os.path.join(formatted_io_cases_dir, f"output_{index}")
            try:
                with open(input_filename, 'w', encoding='utf-8') as f_in:
                    f_in.write(pair.get("input", ""))
                with open(output_filename, 'w', encoding='utf-8') as f_out:
                    f_out.write(pair.get("output", ""))
                written += 1
            except Exception as e:
                return f"Error writing converted test data for case {index}: {e}"

        return f"Successfully converted {written} pairs to {formatted_io_cases_dir} as input_<n>, output_<n>"

    # Removed: LLM-based brute-force code generation. Use bf_solution_generator.py instead.

    def forward(self) -> str:
        try:
            from agentflow.tools.session_manager import get_current_session_id
            conversation_id = get_current_session_id()
            if not conversation_id:
                raise ValueError("Error: No active conversation session. Please use run_coding_agent_with_session().")
            self._set_session_id(conversation_id)
            self._set_output_dir(conversation_id)
        except ImportError:
            raise ImportError("Error: Cannot access conversation session management.")
        except Exception as e:
            return f"Error: {e}"

        try:
            original_question_path = os.path.join(self._output_dir, "original_question.txt")
            print(f"original_question_path: {original_question_path}")
            with open(original_question_path, "r", encoding="utf-8") as f:
                question = f.read()
        except Exception as e:
            return f"Error: Cannot read original question: {e}"

        try:
            # Discover multiple solutions
            solutions = self._discover_multiple_solutions()
            if not solutions:
                return (
                    "Error: No precompiled brute-force executables found. "
                    "Please run bf_solution_generator to produce bf_solution_{index} executables."
                )

            # Initialize details file
            details_path = os.path.join(self._output_dir, 'solution_exec_details.json')
            self._init_solution_exec_details(details_path, question, solutions)

            # Ensure per-solution output directory root
            per_solution_root = os.path.join(self._output_dir, "test_cases_output")
            os.makedirs(per_solution_root, exist_ok=True)

            # Process each solution in parallel
            def process_solution(sol_key, meta_data):
                exec_path_inner = meta_data.get("path_to_exec_file", "")
                if not exec_path_inner or not os.path.exists(exec_path_inner):
                    print(f"Warning: exec for {sol_key} not found, skipping")
                    return sol_key, None
                
                # Run one solution across all inputs
                items_inner = self._run_solution_over_testcases(self._output_dir, exec_path_inner)

                # Save per-solution outputs to files for debugging
                dst_dir = os.path.join(per_solution_root, sol_key)
                os.makedirs(dst_dir, exist_ok=True)
                for i, it in enumerate(items_inner, start=1):
                    out_path = os.path.join(dst_dir, f"output_{i}.txt")
                    output_content = str(it.get("output", ""))
                    try:
                        with open(out_path, 'w', encoding='utf-8') as f:
                            f.write(output_content)
                    except Exception as e:
                        print(f"Warning: failed to write {out_path}: {e}")
                    
                    # Clear output to save memory and avoid large JSON, but keep status
                    if output_content.startswith("ERROR:") or output_content.startswith("TIMEOUT:") or output_content.startswith("EXCEPTION:"):
                        it["output"] = output_content
                    else:
                        it["output"] = "SUCCESS"
                
                return sol_key, items_inner

            # Determine max workers for solutions
            sol_max_workers = min(len(solutions), os.cpu_count() or 4)
            
            with ThreadPoolExecutor(max_workers=sol_max_workers) as executor:
                future_to_sol = {
                    executor.submit(process_solution, sol_key, meta): sol_key
                    for sol_key, meta in solutions.items()
                }
                for future in as_completed(future_to_sol):
                    sol_key, items = future.result()
                    if items:
                        # Merge into solution_exec_details.json (serial merge to avoid race conditions)
                        self._update_solution_exec_details(details_path, sol_key, items)

            # Majority vote -> final_cases.json
            final_cases_path = os.path.join(self._output_dir, 'final_cases.json')
            num_cases = self._compute_majority_final_cases(details_path, final_cases_path)

            # Convert to formatted_io_cases
            try:
                convert_msg = self._convert_golden_outputs_to_test_data(final_cases_path, conversation_id)
                print(convert_msg)
            except Exception as e:
                print(f"Warning: Could not convert final cases to sample_extractor format: {e}")

            return f"Detailed Test Cases are available now. {num_cases} test cases are provided."
        except Exception as e:
            print(f"Error: {e}")
            return f"Failed: {str(e)}"
        
if __name__ == "__main__":
    # Ensure we can import session manager via package path when running as script
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from agentflow.tools.session_manager import (
        set_current_session_id
    )

    # Define a reproducible session id for testing
    conversation_id = "session_1761028552883_73100d5c"

    # Prepare a sample problem statement
    question = r"""
**Problem D. Destruction of the Dandelion Fields**

*Time limit per test:* $2$ seconds
*Memory limit per test:* $256$ megabytes

Farmer John has a lawnmower, initially turned **off**. He also has $ n $ fields, where the $ i $-th field contains $ a_i $ dandelions. He will visit **all** the fields **exactly once**, in any order he chooses.

The lawnmower behaves as follows:
- **Before visiting a field**, it checks whether the number of dandelions in that field is **odd** or **even**.
  - If the number is **odd**, the lawnmower **toggles its state**:
    - If it was **off**, it turns **on**.
    - If it was **on**, it turns **off**.
- **After possibly toggling**, if the lawnmower is **on**, it cuts **all** dandelions in that field.
- If the lawnmower is **off**, no dandelions are cut (FJ just visits the field).

Determine the **maximum total number of dandelions** Farmer John can cut by choosing an optimal visiting order.

---

### Input

The first line contains an integer $ t $ ($ 1 \leq t \leq 10^4 $) — the number of test cases.

Each test case is described as follows:
- The first line contains an integer $ n $ ($ 1 \leq n \leq 2 \cdot 10^5 $) — the number of fields.
- The second line contains $ n $ space-separated integers $ a_1, a_2, \dots, a_n $ ($ 1 \leq a_i \leq 10^9 $) — the number of dandelions in each field.

It is guaranteed that the **sum of $ n $** over all test cases does not exceed $ 2 \cdot 10^5 $.

---

### Output

For each test case, output a single integer: the **maximum number of dandelions** FJ can cut when visiting the fields in an optimal order.

---

### Examples

**Input:**
```
3
3
2 4 6
4
4 2 1 6
4
1000000000 999999999 1000000000 999999999
```

**Output:**
```
0
13
2999999999
```
"""

    # Initialize session and timeout
    set_current_session_id(conversation_id)
    from agentflow.tools.session_manager import set_global_work_dir, get_global_work_dir
    from agentflow import config
    work_dir_base = config.WORK_DIR
    if not os.path.isabs(work_dir_base):
        work_dir_base = os.path.join(project_root, work_dir_base)
    set_global_work_dir(os.path.join(work_dir_base, conversation_id))

    # Prepare a minimal cases_in directory with one input file
    global_dir = get_global_work_dir()
    unit_test_data_dir = os.path.join(global_dir, "unit_test_data")
    testcases_dir = os.path.join(unit_test_data_dir, "TestCases")
    os.makedirs(testcases_dir, exist_ok=True)

    # Run the calculator
    calculator = Cases_Out_Calculator()
    result_msg = calculator.forward()
    print(result_msg)

    # Verify artifacts
    golden_outputs_path = os.path.join(unit_test_data_dir, 'golden_outputs.json')
    final_cases_path = os.path.join(unit_test_data_dir, 'final_cases.json')
    converted_input_path = os.path.join(unit_test_data_dir, 'input')
    converted_output_path = os.path.join(unit_test_data_dir, 'output')

    print(f"golden_outputs.json exists: {os.path.exists(golden_outputs_path)} -> {golden_outputs_path}")
    print(f"final_cases.json exists: {os.path.exists(final_cases_path)} -> {final_cases_path}")
    print(f"converted 'input' exists: {os.path.exists(converted_input_path)} -> {converted_input_path}")
    print(f"converted 'output' exists: {os.path.exists(converted_output_path)} -> {converted_output_path}")

    # Print a short summary from golden_outputs if present
    if os.path.exists(golden_outputs_path):
        try:
            with open(golden_outputs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data.get("golden i/o", []) if isinstance(data, dict) else []
            print(f"golden_outputs items: {len(items)}")
        except Exception as e:
            print(f"Failed to read golden_outputs.json: {e}")
