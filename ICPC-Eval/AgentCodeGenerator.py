
import os
import sys
import re
import tempfile
import shutil
import subprocess
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.main_agent import create_coding_agent, generate_conversation_id
from agentflow.tools.session_manager import (
    set_current_session_id, 
    set_global_work_dir,
    get_global_work_dir,
    set_run_timeout,
    get_global_work_dir,
    set_spj_code,
    reset_token_usage,
    get_token_usage,
)
from agentflow import config as agent_config


_otel_disabled = os.environ.get("OTEL_SDK_DISABLED", "").lower() in ("true", "1", "yes")
if _otel_disabled:
    _tracer = None
    _trace_available = False
    otel_context = None
    Context = None
else:
    try:
        from opentelemetry import trace
        from opentelemetry import context as otel_context
        from opentelemetry.context import Context
        _tracer = trace.get_tracer(__name__)
        _trace_available = True
    except ImportError:
        _tracer = None
        _trace_available = False
        otel_context = None
        Context = None


class AgentCodeGenerator:

    
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        timeout=3600,
        init_prompt=None,  
        tools_config=None,
        unit_test_cases_on=None,
        enable_attempt_compression=None,  
    ):

        self.model = model
        self.api_base = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.tools_config = tools_config
        self.unit_test_cases_on = unit_test_cases_on
        self.enable_attempt_compression = enable_attempt_compression
        
        self.agent = None
        self.history = []
        self.current_problem = None
        
        self.last_tracker_stats = {
            "has_generated_cases": False,
            "generated_cases_improved": False
        }
        
        self._root_span = None
        self._span_context_manager = None
        self._context_token = None
        
        self._attempt_manager = None
        


    def _cleanup_span(self):

        if self._span_context_manager:
            try:
                self._span_context_manager.__exit__(None, None, None)
            except Exception:
                pass
            self._span_context_manager = None
        
        if self._root_span:
            try:
                self._root_span.end()
            except Exception:
                pass
            self._root_span = None
        
        if self._context_token is not None:
            try:
                otel_context.detach(self._context_token)
            except Exception:
                pass
            self._context_token = None


        if not self.agent:
            return
        
        try:
            from agentflow.tools.attempt_logger import AttemptBasedTrajectoryManager
            from agentflow.tools.attempt_agent_wrapper import wrap_agent_with_attempt_tracking
            
            work_dir = getattr(self, '_global_work_dir', None)
            if not work_dir:
                work_dir = get_global_work_dir()
            
            if not work_dir:
                return
            
            self._attempt_manager = AttemptBasedTrajectoryManager(
                model_id=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                work_dir=work_dir,
                keep_full_attempts=1,  
                enable_summary=True    
            )
            
            wrap_agent_with_attempt_tracking(self.agent, self._attempt_manager)
            
            
        except Exception as e:
            print(f"⚠️ Attempt compression initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self._attempt_manager = None

    def end_problem(self, final_status="Unknown"):

        self._collect_tracker_stats()
        
        if self._root_span:
            try:
                self._root_span.set_attribute("problem.final_status", final_status)
                self._root_span.set_attribute("problem.total_attempts", getattr(self, 'refine_count', 0) + 1)
            except Exception:
                pass
        
        self._cleanup_span()
        print(f"📊  {final_status})")

    def _format_problem_prompt(self, problem_info):

        prompt = f"""Problem: {problem_info['title']}
Time limit: {problem_info['time_limit_ms']}ms
Memory limit: {problem_info['memory_limit_mb']}MB

[Description]
{problem_info['description']}

[Input]
{problem_info['input']}

[Output]
{problem_info['output']}

"""
        examples = problem_info.get('examples', [])
        for idx, example in enumerate(examples):
            if isinstance(example, (list, tuple)) and len(example) >= 2:
                prompt += f"[Sample Input {idx + 1}]\n{example[0]}\n"
                prompt += f"[Sample Output {idx + 1}]\n{example[1]}\n"
        
        note = problem_info.get('note', '')
        if note:
            prompt += f"\n[Note]\n{note}\n"
        

        return prompt

    def _fix_broken_char_literals(self, code: str) -> str:

        if not code:
            return code
        
        original_code = code

        code = re.sub(r"'\s*\n\s*'", r"'\\n'", code)
        code = re.sub(r"'\t'", r"'\\t'", code)
        code = re.sub(r"'\r'", r"'\\r'", code)
        
        def fix_broken_string(match):
            content = match.group(1)
            if '\n' in content or '\t' in content or '\r' in content:
                fixed = content.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
                return f'"{fixed}"'
            return match.group(0)
        
        code = re.sub(r'"([^"]*?)"', fix_broken_string, code)
        
        def fix_broken_comment(match):
            comment_start = match.group(1)
            broken_content = match.group(2)
            cpp_indicators = [';', '{', '}', '#include', 'int ', 'void ', 'return', 'for', 'while', 'if', 'else']
            is_likely_code = any(ind in broken_content for ind in cpp_indicators)
            if not is_likely_code and broken_content.strip():
                return f"{comment_start} {broken_content.strip()}"
            return match.group(0)
        
        code = re.sub(r'(//[^\n]*)\n([^\n]*?)(?=\n|$)', fix_broken_comment, code)
        
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            if re.search(r"<<\s*'\\[nt]$", line):
                line = line + "';"
            fixed_lines.append(line)
        code = '\n'.join(fixed_lines)
        

        return code

    def extract_code_block(self, output):

        if not output:
            return ""
        
        patterns = [
            r"```cpp\n(.*?)```",
            r"```c\+\+\n(.*?)```",
            r"```\n(.*?)```",
        ]
        
        code = ""
        for pattern in patterns:
            code_blocks = re.findall(pattern, output, re.DOTALL)
            if code_blocks:
                code = code_blocks[-1].strip()
                break
        
        if not code and "#include" in output:
            start = output.find("#include")
            end = max(output.rfind("}"), output.rfind("return 0;"))
            if end > start:
                code = output[start:end+1].strip()
        
        if code:
            code = self._fix_broken_char_literals(code)
        
        return code

    def _initialize_session(self, problem_info, question):

        conversation_id = generate_conversation_id(question)
        set_current_session_id(conversation_id)
        self.conversation_id = conversation_id
        print(f"📌 Session ID: {conversation_id}")
        
        work_dir_base = agent_config.WORK_DIR
        if not os.path.isabs(work_dir_base):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            work_dir_base = os.path.join(project_root, work_dir_base)
        global_work_dir = os.path.join(work_dir_base, conversation_id)
        os.makedirs(global_work_dir, exist_ok=True)
        set_global_work_dir(global_work_dir)
        self._global_work_dir = global_work_dir 
        print(f"📁 work dir: {global_work_dir}")
        
    
        time_limit_ms = problem_info.get("time_limit_ms", 2000)
        timeout_seconds = max(1, (time_limit_ms + 500) // 1000)  
        set_run_timeout(timeout_seconds)
        
        if problem_info.get("type") == "spj" and problem_info.get("spj_code"):
            set_spj_code(problem_info["spj_code"])
        
        preprocess_list = agent_config.PREPROCESS_TOOLS_ENABLED or []
        print(f"🔧  {', '.join(preprocess_list) if preprocess_list else ''}")
        
        if "info_extractor" in preprocess_list:
            try:
                from agentflow.tools.problem_info_extractor import ProblemInfoExtractorTool
                

                extractor = ProblemInfoExtractorTool(
                    model_id=self.model, 
                    api_base=self.api_base, 
                    api_key=self.api_key
                )
                extractor.set_session_id(conversation_id)
                
                extract_result = extractor.forward(question, conversation_id=conversation_id)
                print(extract_result)
                print("=" * 30)
            except Exception as e:
                print(f"⚠️ : {e}")
                print("...")
        
       
    def generate_code(
        self, problem_info, sleep=0, stream=False, reasoning_history=True
    ):

        self.current_problem = problem_info
        self.history = []
        self.refine_count = 0
        self.agent = None  
        reset_token_usage()
        
        self._cleanup_span()
        
        question = self._format_problem_prompt(problem_info)
        problem_id = problem_info.get('id', 'unknown')
        
        print(f"\n{'='*60}")
        print(f"AgentCodeGenerator:{problem_id} - {problem_info.get('title')}")
        print(f"🔧  model={self.model}, api_base={self.api_base}")
        print(f"{'='*60}")

        if _trace_available and _tracer and otel_context and Context:
            try:
          
                self._context_token = otel_context.attach(Context())
            except Exception as e:
                print(f"⚠️  {e}")
                self._context_token = None
            
            if self._context_token is not None:
                try:
                    span_name = f"problem:{problem_id}"
                    self._root_span = _tracer.start_span(span_name)
                    self._root_span.set_attribute("problem.id", str(problem_id))
                    self._root_span.set_attribute("problem.title", problem_info.get('title', ''))
                    self._root_span.set_attribute("model.id", self.model or '')
                    self._root_span.set_attribute("unit_test_cases_on", self.unit_test_cases_on or False)
                    
                    self._span_context_manager = trace.use_span(self._root_span, end_on_exit=False)
                    self._span_context_manager.__enter__()
                    print(f"📊  {span_name}")
                except Exception as e:
                    if self._root_span:
                        try:
                            self._root_span.end()
                        except Exception:
                            pass
                    self._root_span = None
                    self._span_context_manager = None
        
        try:
            self._initialize_session(problem_info, question)
            
            simplified_question = question
            try:
                global_work_dir = get_global_work_dir()
                if global_work_dir:
                    simple_draft_path = os.path.join(global_work_dir, "simple_draft.txt")
                    if os.path.exists(simple_draft_path):
                        with open(simple_draft_path, "r", encoding="utf-8") as f:
                            simplified_question = f.read()
                        print(f"📄  Using simplified question from {simple_draft_path}")
            except Exception as e:
                print(f"⚠️  Failed to read simplified question: {e}")
            
            self.agent = create_coding_agent(
                model_id=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                tools_config=self.tools_config,
                unit_test_cases_on=self.unit_test_cases_on
            )
            
            if self.enable_attempt_compression:
                self._initialize_attempt_compression()
            
            result = self.agent.run(simplified_question)
            result = str(result)
            
            self._collect_tracker_stats()
            
            code = self.extract_code_block(result)
            
            if not code:
                code = self._try_read_code_from_workdir()
            
            self.history = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result}
            ]
            
            output_len = len(result) if result else 0
            code_len = len(code) if code else 0
            
            time.sleep(sleep)
            
            return (code, result, output_len, code_len, 0)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self._root_span:
                self._root_span.record_exception(e)
            self._cleanup_span()
            return ("", "", 0, 0, 0)
        


    def _collect_tracker_stats(self):

        self.last_tracker_stats = {
            "has_generated_cases": False,
            "generated_cases_improved": False
        }
        
        if not self.agent:
            return
            
        try:
            if hasattr(self.agent, '_validation_tracker') and self.agent._validation_tracker:
                tracker = self.agent._validation_tracker
                
                self.last_tracker_stats["generated_cases_improved"] = tracker.check_generated_cases_improvement()
                
                for record in tracker.records:
                    if record.generated_stats and record.generated_stats.get("has_generated_cases"):
                        self.last_tracker_stats["has_generated_cases"] = True
                        break
                        
                if self.last_tracker_stats["has_generated_cases"]:
                    improved = "Yes" if self.last_tracker_stats["generated_cases_improved"] else "No"
                
                self._collect_experience_from_tracker(tracker)
                
        except Exception as e:
            print(f"⚠️ Failed: {e}")
    
    def _collect_experience_from_tracker(self, tracker):

        try:
            from agentflow import config
            long_term_exp_enabled = config.LONG_TERM_EXP_ENABLED
            if long_term_exp_enabled is None:
                if self.tools_config and "long_term_exp" in self.tools_config:
                    long_term_exp_enabled = True
                else:
                    long_term_exp_enabled = False
            
            if not long_term_exp_enabled:
                return
        except Exception:
            return
        
        if not tracker.check_success_after_failure():
            return
        

        try:
            context = tracker.extract_improvement_context(self.agent)
            
            from agentflow.tools.success_analyzer import SuccessAnalyzer
            
            analyzer = SuccessAnalyzer(
                model_id=self.model,
                api_base=self.api_base,
                api_key=self.api_key
            )
            
            success_data = analyzer.analyze(context)
            
            if success_data:
                from agentflow.tools.memory_module import get_memory_module
                from agentflow import config as agent_config
                
                failure_info = context.get('failure', {})
                success_info = context.get('success', {})
                code_context = success_info.get('code', '') or failure_info.get('code', '')
                
                memory = get_memory_module(
                    storage_path=agent_config.MEMORY_STORE_PATH,
                    model_id=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key
                )
                exp_id, is_new = memory.add_experience(
                    fix_summary=success_data.get('fix_summary', ''),
                    original_error=success_data.get('original_error', ''),
                    key_insight=success_data.get('key_insight', ''),
                    code_context=code_context,
                    metadata={
                        'timestamp': success_data.get('timestamp'),
                        'root_cause': success_data.get('root_cause', '')
                    },
                    error_context=success_data.get('error_context', ''),
                    error_cause=success_data.get('error_cause', ''),
                    fix_method=success_data.get('fix_method', ''),
                    fix_result=success_data.get('fix_result', '')
                )
              
                
        except Exception as e:
            import traceback
            traceback.print_exc()
        

    def get_token_stats(self) -> dict:

        stats = get_token_usage()
        return stats

    def _try_read_code_from_workdir(self):

        try:
            from agentflow.tools.session_manager import get_global_work_dir
            work_dir = get_global_work_dir()
            
            if work_dir and os.path.exists(work_dir):
                possible_files = [
                    "solution.cpp",
                    "code.cpp",
                    "main.cpp",
                ]
                for filename in possible_files:
                    filepath = os.path.join(work_dir, filename)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            code = f.read()
                            return self._fix_broken_char_literals(code)
        except Exception:
            pass
        return ""

    def correct_code(
        self, correct_info, period, sleep=0, stream=False, reasoning_history=True
    ):

        if not self.current_problem:
            return ("", "", 0, 0, 0)
        

        suggestion_map = {
        "Runtime Error": "This means your program crashed or failed to run correctly. Check for issues such as division by zero, out-of-bounds array access, or null-pointer dereferences.",
        "Time Limit Exceeded": "Your program is taking too long to run. Analyze its time complexity and consider using a more efficient algorithm or data structure. Also optimize loop structure and input/output (I/O) operations.",
        "Time Limit Exceeded (Wall-clock timeout)": "Your program is taking too long to run. Optimize loop structure and input/output (I/O) operations; for example, replace newlines with endl in the output, or comment out any untying/unbinding code.",
        "Memory Limit Exceeded": "Your program is using too much memory. Check for overly large data structures, memory leaks, or inefficient recursion that causes excessive stack/memory overhead, and optimize memory usage accordingly.",
        "Compilation Error": "Your program has syntax errors or is missing required header files (#include). Carefully read the compiler error messages and fix the issues as indicated.",
        "Wrong Answer": "Your output is incorrect. Debug and verify your logic, test with edge cases/extreme inputs, and make sure you understand the constraints and the required output format. Please pay special attention to the test cases that caused the error.",
        "Compile error": "Your program has syntax errors or is missing required header files (#include). Carefully read the compiler error messages and fix the issues as indicated.",
        }

        suggestion = suggestion_map.get(correct_info)
        if not suggestion:

            sorted_keys = sorted(suggestion_map.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if key in correct_info:
                    suggestion = suggestion_map[key]
                    break
        
        if not suggestion:
            suggestion = "An unknown error occurred. Please review your code logic and try to identify the issue."
        
        last_code = ""
        if self.history:
            for msg in reversed(self.history):
                if msg["role"] == "assistant":
                    last_code = self.extract_code_block(msg["content"])
                    if last_code:
                        break
        
# Build the correction prompt (including the previous code and error information)
        correction_prompt = f"""During the '{period}' testing phase, the previously submitted code resulted in a '{correct_info}' error.
{suggestion}
Based on the error message and your code, analyze the possible cause of this error and provide a C++ solution for this problem.
You must still answer the question following the instructions given at the very beginning, and you may still use all the provided tools.
Finally, you must provide the complete, runnable corrected code in the final_answer tool.
"""

        self.refine_count = getattr(self, 'refine_count', 0) + 1
        
        print(f"\n{'='*60}")
        print(f"AgentCodeGenerator: (Attempt #{self.refine_count + 1})")
        print(f"📋 : {correct_info}")
        print(f"📋 : {period}")
        print(f"{'='*60}")
        
        try:
            if self.agent is not None:
 

                result = self.agent.run(correction_prompt, reset=False)
                result = str(result)
            else:

                full_prompt = self._format_problem_prompt(self.current_problem)
                full_prompt += f"\n\n---\n{correction_prompt}"
                
                self.agent = create_coding_agent(
                    model_id=self.model,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    tools_config=self.tools_config,
                    unit_test_cases_on=self.unit_test_cases_on
                )
                
                if self.enable_attempt_compression:
                    self._initialize_attempt_compression()
                result = self.agent.run(full_prompt)
                result = str(result)
            
            code = self.extract_code_block(result)
            
            if not code:
                code = self._try_read_code_from_workdir()
            
            self.history.append({"role": "user", "content": correction_prompt})
            self.history.append({"role": "assistant", "content": result})
            
            output_len = len(result) if result else 0
            code_len = len(code) if code else 0
            
            time.sleep(sleep)
            
            return (code, result, output_len, code_len, 0)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ("", "", 0, 0, 0)

    def check_examples(self, code, problem_info):

        examples_to_check = problem_info.get("examples", [])
        if not examples_to_check:
            return "Accepted", "No examples provided to check."

        tmp_dir = ""
        try:
            tmp_dir = tempfile.mkdtemp(prefix="code_check_")

            gpp_compiler = "g++"
            cxx_flags_list = ["-std=c++17", "-O2", "-pipe", "-static", "-s"]

            src_file = os.path.join(tmp_dir, "example_main.cpp")
            exe_file = os.path.join(tmp_dir, "example_main_exe")
            if os.name == "nt":
                exe_file += ".exe"

            with open(src_file, "w", encoding="utf-8") as f:
                f.write(code)

            compile_cmd = [gpp_compiler, *cxx_flags_list, src_file, "-o", exe_file]
            compile_res = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=20
            )

            if compile_res.returncode != 0:
                err_msg = (
                    compile_res.stderr.strip()
                    if compile_res.stderr.strip()
                    else compile_res.stdout.strip()
                )
                return (
                    "Compilation Error",
                    f"Compile error:\n{err_msg if err_msg else 'Unknown compilation error'}",
                )

            problem_type = problem_info.get("type", "traditional")
            spj_exe_file = None
            if problem_type == "spj":
                spj_code = problem_info.get("spj_code")
                if not spj_code:
                    return "System Error", "SPJ problem but no spj_code provided."
                
                spj_src_file = os.path.join(tmp_dir, "spj_checker.cpp")
                spj_exe_file = os.path.join(tmp_dir, "spj_checker_exe")
                if os.name == "nt":
                    spj_exe_file += ".exe"
                
                with open(spj_src_file, "w", encoding="utf-8") as f:
                    f.write(spj_code)
                
                spj_compile_cmd = [gpp_compiler, *cxx_flags_list, spj_src_file, "-o", spj_exe_file]
                spj_compile_res = subprocess.run(
                    spj_compile_cmd, capture_output=True, text=True, timeout=20
                )
                
                if spj_compile_res.returncode != 0:
                    err_msg = (
                        spj_compile_res.stderr.strip()
                        if spj_compile_res.stderr.strip()
                        else spj_compile_res.stdout.strip()
                    )
                    return (
                        "System Error",
                        f"SPJ compile error:\n{err_msg if err_msg else 'Unknown SPJ compilation error'}",
                    )

            time_limit_s = problem_info.get("time_limit_ms", 1000) / 1000.0

            for idx, example_pair in enumerate(examples_to_check):
                if not (
                    isinstance(example_pair, (list, tuple))
                    and len(example_pair) == 2
                    and isinstance(example_pair[0], str)
                    and isinstance(example_pair[1], str)
                ):
                    continue

                input_str, expected_output_str = example_pair

                try:
                    proc_example = subprocess.Popen(
                        [exe_file],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    actual_stdout, actual_stderr = proc_example.communicate(
                        input=input_str, timeout=time_limit_s + 0.5
                    )

                    if proc_example.returncode != 0:
                        return (
                            "Runtime Error",
                            f"Runtime Error on example {idx + 1}. Stderr:\n{actual_stderr[:500]}",
                        )

                    def normalize_local(text):
                        if text is None:
                            return []
                        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                        processed = [line.rstrip() for line in lines]
                        while processed and processed[-1] == "":
                            processed.pop()
                        return processed

                    if problem_type == "spj" and spj_exe_file:
                        temp_expected_path = os.path.join(tmp_dir, f"example_{idx}_expected.txt")
                        temp_user_path = os.path.join(tmp_dir, f"example_{idx}_user.txt")
                        
                        with open(temp_expected_path, "w", encoding="utf-8") as f:
                            f.write(expected_output_str)
                        with open(temp_user_path, "w", encoding="utf-8") as f:
                            f.write(actual_stdout or "")
                        
                        spj_command = [
                            spj_exe_file,
                            temp_expected_path,  
                            temp_user_path,      
                        ]
                        
                        try:
                            spj_run_res = subprocess.run(
                                spj_command,
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            spj_output = spj_run_res.stdout.strip()
                            
                            if spj_run_res.returncode != 0 and not spj_output:
                                return (
                                    "System Error",
                                    f"SPJ execution failed on example {idx + 1}. stderr: {spj_run_res.stderr[:200]}",
                                )
                            elif spj_output:
                                if spj_output not in ["Accepted", "AC", "accepted", "ac"]:
                                    return (
                                        "Wrong Answer",
                                        f"Wrong Answer on example {idx + 1} (SPJ verdict: {spj_output}).\nActual:\n{actual_stdout[:500]}\nExpected:\n{expected_output_str[:500]}",
                                    )
                            elif spj_run_res.returncode != 0:
                                return (
                                    "Wrong Answer",
                                    f"Wrong Answer on example {idx + 1} (SPJ returned non-zero).\nActual:\n{actual_stdout[:500]}\nExpected:\n{expected_output_str[:500]}",
                                )
                            
                        except subprocess.TimeoutExpired:
                            return (
                                "System Error",
                                f"SPJ timed out on example {idx + 1}",
                            )
                        except Exception as spj_e:
                            return (
                                "System Error",
                                f"SPJ error on example {idx + 1}: {spj_e}",
                            )
                    else:
                        actual_lines = normalize_local(actual_stdout)
                        expected_lines = normalize_local(expected_output_str)

                        if actual_lines != expected_lines:
                            return (
                                "Wrong Answer",
                                f"Wrong Answer on example {idx + 1}.\nActual:\n{actual_stdout[:500]}\nExpected:\n{expected_output_str[:500]}",
                            )

                except subprocess.TimeoutExpired:
                    proc_example.kill()
                    return (
                        "Time Limit Exceeded",
                        f"Time Limit Exceeded on example {idx + 1} (>{time_limit_s}s)",
                    )
                except Exception as e_run:
                    return "Runtime Error", f"Error running example {idx + 1}: {e_run}"

            return "Accepted", "All examples passed!"

        except Exception as e_outer:
            return "System Error", f"Error during example check setup: {e_outer}"
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_problem = {
        "id": 1,
        "title": "A+B Problem",
        "type": "traditional",
        "time_limit_ms": 1000,
        "memory_limit_mb": 256,
        "description": "Read two integers a and b, print their sum.",
        "input": "Two integers a and b (-10^9 <= a, b <= 10^9).",
        "output": "The sum a+b.",
        "examples": [["1 2\n", "3\n"]],
        "note": "",
    }
    
    generator = AgentCodeGenerator()
    code, output, output_len, code_len, tokens = generator.generate_code(test_problem)
    

    print(code)
    
    if code:
        result, message = generator.check_examples(code, test_problem)


