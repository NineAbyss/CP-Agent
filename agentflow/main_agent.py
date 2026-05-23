import sys
import os
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from contextlib import nullcontext
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.tools.cpp_interpreter import CppInterpreterTool
from agentflow.tools.cases_out_calculator import Cases_Out_Calculator
from agentflow.tools.session_manager import (
    get_session_manager, 
    set_current_session_id, 
    get_current_session_id,
    generate_session_id,
    clear_current_session,
    set_global_work_dir,
    reset_token_usage,
    get_token_usage,
)

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from agentflow import config

EMBEDDING_API_URL = config.EMBEDDING_API_URL

PHOENIX_COLLECTOR_ENDPOINT = config.PHOENIX_COLLECTOR_ENDPOINT


if not os.getenv("_PHOENIX_INITIALIZED") and not os.getenv("ABLATION_CONFIG_NAME"):
    try:
        tracer_provider = register(
            project_name=config.PHOENIX_PROJECT_NAME,
            endpoint=f"{config.PHOENIX_COLLECTOR_ENDPOINT}/v1/traces",
            auto_instrument=True,
            batch=True  
        )
        instrumentor = SmolagentsInstrumentor()
        try:
            already = getattr(instrumentor, "is_instrument", False)
        except Exception:
            already = False
        if not already:
            instrumentor.instrument
        os.environ["_PHOENIX_INITIALIZED"] = "1"
        os.environ["_PHOENIX_PROJECT_NAME"] = config.PHOENIX_PROJECT_NAME
        print(f" {config.PHOENIX_PROJECT_NAME}）")
    except Exception as e:
        print(f"⚠️ {e}")
elif os.getenv("_PHOENIX_INITIALIZED"):
    print(f"🔄 Project: {os.getenv('_PHOENIX_PROJECT_NAME')})")
else:
    print(f"🔄 config: {os.getenv('ABLATION_CONFIG_NAME')})")


import litellm
litellm.drop_params = True


import smolagents.models as _smolagents_models
import re
_original_supports_stop = _smolagents_models.supports_stop_parameter
def _patched_supports_stop_parameter(model_id: str) -> bool:
    model_name = model_id.split("/")[-1].lower()
    if re.match(r"gpt-5\.\d+", model_name):
        return False
    return _original_supports_stop(model_id)
_smolagents_models.supports_stop_parameter = _patched_supports_stop_parameter

from smolagents import LiteLLMModel, ToolCallingAgent, OpenAIServerModel, CodeAgent
from agentflow.models import create_model

try:
    from smolagents.models import REMOVE_PARAMETER
except ImportError:
    class _ParameterRemove:
        pass
    REMOVE_PARAMETER = _ParameterRemove()

def setup_logging(log_file: str = None):

    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"main_agent_{timestamp}.log")
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger('agentflow')
    logger.setLevel(logging.INFO)
    
    class LogWriter:
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level
        
        def write(self, message):
            if message.strip():  
                self.logger.log(self.level, message.strip())
        
        def flush(self):
            pass
    
    sys.stdout = LogWriter(root_logger, logging.INFO)
    sys.stderr = LogWriter(root_logger, logging.ERROR)
    
    logger.info(f" {log_file}")
    return logger, log_file

def generate_conversation_id(question: str) -> str:

    return generate_session_id(question)

def create_coding_agent(model_id: str = None, api_base: str = None, api_key: str = None, tools_config: str = None, unit_test_cases_on: bool = None, preprocess_config: str = None):

    model_id = model_id or config.MODEL_ID
    api_base = api_base or config.API_BASE
    api_key = api_key or config.API_KEY
    
    if tools_config:
        tools_list = [t.strip() for t in tools_config.split(",")]
    else:
        tools_list = config.TOOLS_ENABLED
    
    if unit_test_cases_on is None:
        unit_test_cases_on = config.UNIT_TEST_CASES_ON
    

    long_term_exp_enabled = config.LONG_TERM_EXP_ENABLED
    
    if long_term_exp_enabled is None:
        if "long_term_exp" in tools_list:
            long_term_exp_enabled = True
        else:
            long_term_exp_enabled = False 

    elif "long_term_exp" in tools_list and not long_term_exp_enabled:
        print("Warning: 'long_term_exp' found in tools list but explicitly disabled in features. Long term exp will be DISABLED.")
    

    validation_tracker = None
    if "cpp_validation" in tools_list or "cpp_interpreter" in tools_list:
        from agentflow.tools.validation_tracker import ValidationTracker
        validation_tracker = ValidationTracker()
    
    tools = []
    

    

    cpp_validation = None  
    cpp_interpreter = None 
    if "cpp_validation" in tools_list:
        if unit_test_cases_on:
            from agentflow.tools.cpp_validation_w_cases_generator import CppValidationTool
            cpp_validation = CppValidationTool(
                sandbox_url=config.SANDBOX_URL,
                enable_error_analysis=long_term_exp_enabled,
                model_id=model_id,
                api_base=api_base,
                api_key=api_key
            )
            tools.append(cpp_validation)
            status = "with test case generation"
            if long_term_exp_enabled:
                status += " and error analysis"
            print(f"  ✓ CppValidationTool ({status})")
        else:
            from agentflow.tools.cpp_validation import CppValidationTool
            cpp_validation = CppValidationTool(
                sandbox_url=config.SANDBOX_URL,
                enable_error_analysis=long_term_exp_enabled,
                model_id=model_id,
                api_base=api_base,
                api_key=api_key
            )
            tools.append(cpp_validation)
            status = "basic"
            if long_term_exp_enabled:
                status += " with error analysis"
            print(f"  ✓ CppValidationTool ({status})")
    
    if "cpp_interpreter" in tools_list:
        cpp_interpreter = CppInterpreterTool(
            sandbox_url=config.SANDBOX_URL,
            enable_error_analysis=long_term_exp_enabled,
            model_id=model_id,
            api_base=api_base,
            api_key=api_key
        )
        tools.append(cpp_interpreter)
        status = "basic"
        if long_term_exp_enabled:
            status = "with error analysis"
        print(f"  ✓ CppInterpreterTool ({status})")
    
   



    general_experiences_content = ""
    memory_module = None
    if long_term_exp_enabled:
        try:
            from agentflow.tools.memory_module import get_memory_module, create_algorithm_experience_tool
            
            memory_module = get_memory_module(
                storage_path=config.MEMORY_STORE_PATH,  
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                enable_dedup=config.ENABLE_EXPERIENCE_DEDUP  
            )
            
            general_experiences_content = memory_module.get_general_experiences_text(limit=10)
            if general_experiences_content:
                print(f"✅  {len(memory_module.general_experiences)} general experiences")
            else:
                print("ℹ️ General experiences are empty")
            
            algo_exp_tool = create_algorithm_experience_tool(memory_module=memory_module)
            tools.append(algo_exp_tool)
            print("  ✓ AlgorithmExperienceRetrieverTool (added to tools list)")
            
            stats = memory_module.get_statistics()
            print(f"📊 Memory statistics: General={stats['general']['count']}, Algorithm={stats['algorithm']['count']}")
            
        except Exception as e:
            print(f"⚠️ Memory module initialization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("ℹ️ Memory module is disabled (long_term_exp_enabled=False)")

    agent_type = config.AGENT_TYPE.lower() if hasattr(config, 'AGENT_TYPE') else "code"
    

    extra_model_kwargs = {}
    if agent_type == "tool_calling":
        extra_model_kwargs = {
            "tools": REMOVE_PARAMETER,       
            "tool_choice": REMOVE_PARAMETER 
        }
    
    model = create_model(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        include_reasoning_in_context=config.INCLUDE_REASONING_IN_CONTEXT,
        temperature=config.MODEL_TEMPERATURE,
        max_tokens=config.MODEL_MAX_TOKENS,
        **extra_model_kwargs
    )

    if agent_type == "tool_calling":
        agent = ToolCallingAgent(
            tools=tools,  # List of tools available to the agent
            model=model,
            max_steps=config.AGENT_MAX_STEPS,  # Limit the number of reasoning steps
            verbosity_level=config.AGENT_VERBOSITY_LEVEL,
            add_base_tools=False,  
        )
    else:
        agent = CodeAgent(
            tools=tools,  # List of tools available to the agent
            model=model,
            max_steps=config.AGENT_MAX_STEPS,  # Limit the number of reasoning steps
            # planning_interval=3, # This is where you activate planning!
            verbosity_level=config.AGENT_VERBOSITY_LEVEL,
            add_base_tools=False, 
        )
    import yaml
    from jinja2 import Template
    from agentflow.config import AGENT_PROMPT_PATH
    
    if os.path.isabs(AGENT_PROMPT_PATH):
        prompt_file_path = AGENT_PROMPT_PATH
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_file_path = os.path.join(project_root, AGENT_PROMPT_PATH)
    
    print(f"📝  {prompt_file_path}")
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        agent.prompt_templates = yaml.safe_load(f)
    

    original_prompt = agent.prompt_templates["system_prompt"]
    
    import re
    pattern = r'\{%-?\s*if\s+long_term_experiences\s*%\}.*?\{%-?\s*endif\s*%\}'
    replacement = general_experiences_content 
    
    agent.prompt_templates["system_prompt"] = re.sub(
        pattern, 
        replacement, 
        original_prompt, 
        flags=re.DOTALL
    )
    
    if validation_tracker:
        if cpp_validation:
            original_validation_forward = cpp_validation.forward
            
            def wrapped_validation_forward(code):
                result = original_validation_forward(code)
                
                step_idx = len(agent.memory.steps) if hasattr(agent, 'memory') and agent.memory and hasattr(agent.memory, 'steps') else 0
                
                is_success = ("✅" in result or "PASSED" in result or "All tests passed" in result)
                
                validation_tracker.record_validation(
                    step_idx, 
                    code, 
                    result, 
                    is_success, 
                    tool_type="validation",
                    generated_stats=getattr(cpp_validation, 'last_run_stats', None)
                )
                
                return result
            
            cpp_validation.forward = wrapped_validation_forward
        
        if cpp_interpreter:
            original_interpreter_forward = cpp_interpreter.forward
            
            def wrapped_interpreter_forward(code):
                result = original_interpreter_forward(code)
                
                step_idx = len(agent.memory.steps) if hasattr(agent, 'memory') and agent.memory and hasattr(agent.memory, 'steps') else 0
                

                is_success = not ("Compile error" in result or "Compilation failed" in result or 
                                "Runtime error" in result or "RUNTIME ERROR" in result or
                                "Sandbox error" in result)
                
                validation_tracker.record_validation(step_idx, code, result, is_success, tool_type="interpreter")
                
                return result
            
            cpp_interpreter.forward = wrapped_interpreter_forward
    
    agent._validation_tracker = validation_tracker
    agent._model_config = (model_id, api_base, api_key)

    return agent

def serialize_raw_data(raw_data):

    if raw_data is None:
        return None
    
    try:
        json.dumps(raw_data)
        return raw_data
    except TypeError:
        if hasattr(raw_data, '__dict__'):
            try:
                return {
                    "type": str(type(raw_data).__name__),
                    "data": str(raw_data)
                }
            except:
                return str(raw_data)
        else:
            return str(raw_data)

def save_agent_conversation(agent, output_file="agent_conversation.json"):

    all_messages = []
    
    if not hasattr(agent, 'memory') or agent.memory is None:
        print("Warning: Agent memory is None or not available")
        return None
    
    if not hasattr(agent.memory, 'steps') or agent.memory.steps is None:
        print("Warning: Agent memory.steps is None or not available")
        return None
    
    for step_idx, step in enumerate(agent.memory.steps):
        if hasattr(step, "model_input_messages") and step.model_input_messages is not None:
            for msg in step.model_input_messages:
                all_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "raw": serialize_raw_data(msg.raw),
                    "step_index": step_idx,
                    "step_type": type(step).__name__ if hasattr(step, '__class__') else "Unknown",
                    "message_type": "input"
                })
        
        if hasattr(step, "model_output_message") and step.model_output_message is not None:
            msg = step.model_output_message
            all_messages.append({
                "role": msg.role,
                "content": msg.content,
                "raw": serialize_raw_data(msg.raw),
                "step_index": step_idx,
                "step_type": type(step).__name__ if hasattr(step, '__class__') else "Unknown",
                "message_type": "output"
            })
    
    all_messages.sort(key=lambda x: (x["step_index"], 0 if x["message_type"] == "input" else 1))
    
    conversation_data = []
    current_turn = None
    
    for msg in all_messages:
        if msg["message_type"] == "input":
            if current_turn is not None:
                conversation_data.append(current_turn)
            current_turn = {
                "input_role": msg["role"],
                "input_content": msg["content"],
                "input_raw": msg["raw"],
                "output_role": None,
                "output_content": None,
                "output_raw": None,
                "step_index": msg["step_index"],
                "step_type": msg["step_type"]
            }
        elif msg["message_type"] == "output" and current_turn is not None:
            current_turn.update({
                "output_role": msg["role"],
                "output_content": msg["content"],
                "output_raw": msg["raw"]
            })
    
    if current_turn is not None:
        conversation_data.append(current_turn)
    
    conversation_record = {
        "timestamp": datetime.now().isoformat(),
        "conversation": conversation_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversation_record, f, ensure_ascii=False, indent=2)
    
    print(f"Save to: {output_file}")
    return conversation_record


def run_coding_agent_with_session(question: str, model_id: str = None, api_base: str = None, api_key: str = None, tools_config: str = None, unit_test_cases_on: bool = None, preprocess_config: str = None, question_id: str = None) -> Tuple[str, Dict]:

    model_id = model_id or config.MODEL_ID
    api_base = api_base or config.API_BASE
    api_key = api_key or config.API_KEY
    
    if unit_test_cases_on is None:
        unit_test_cases_on = config.UNIT_TEST_CASES_ON
    
    long_term_exp_enabled = config.LONG_TERM_EXP_ENABLED
    if long_term_exp_enabled is None:
        if tools_config and "long_term_exp" in tools_config:
            long_term_exp_enabled = True
        else:
            long_term_exp_enabled = False
    
    if preprocess_config:
        preprocess_list = [t.strip() for t in preprocess_config.split(",")]
    else:
        preprocess_list = config.PREPROCESS_TOOLS_ENABLED
    
    
    conversation_id = generate_conversation_id(question)
    set_current_session_id(conversation_id)
    
    reset_token_usage()
    
    work_dir_base = config.WORK_DIR
    if not os.path.isabs(work_dir_base):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        work_dir_base = os.path.join(project_root, work_dir_base)
    global_work_dir = os.path.join(work_dir_base, conversation_id)
    os.makedirs(global_work_dir, exist_ok=True)
    set_global_work_dir(global_work_dir)
    
    unit_test_data_dir = os.path.join(global_work_dir, "unit_test_data")
    os.makedirs(unit_test_data_dir, exist_ok=True)
    original_question_path = os.path.join(unit_test_data_dir, "original_question.txt")
    try:
        with open(original_question_path, 'w', encoding='utf-8') as f:
            f.write(question)
    except Exception as e:
        print(f"⚠️ Save Failed: {e}")
    
    print(f"Session ID: {conversation_id}")
    

    try:
        from opentelemetry import trace
        from opentelemetry.trace import use_span
        from opentelemetry.context import Context
        tracer = trace.get_tracer(__name__)
        span_name = f"problem:{question_id}" if question_id else "CP-Agent.full_pipeline"
        empty_context = Context()
        root_span = tracer.start_span(span_name, context=empty_context)
        root_span.set_attribute("session.id", conversation_id)
        root_span.set_attribute("problem.question", question)
        root_span.set_attribute("model.id", model_id)
        root_span.set_attribute("tools_config", tools_config or "")
        root_span.set_attribute("unit_test_cases_on", unit_test_cases_on)
        root_span.set_attribute("preprocess_config", preprocess_config)
        if question_id:
            root_span.set_attribute("problem.question_id", question_id)
        trace_available = True
    except (ImportError, Exception) as e:
        root_span = None
        trace_available = False
        print(f"⚠️  Phoenix trace Failed: {e}")
    
    with use_span(root_span, end_on_exit=True) if root_span else nullcontext():
        try:

            
            if "info_extractor" in preprocess_list:
                from agentflow.tools.problem_info_extractor import ProblemInfoExtractorTool
            

            
            if root_span and trace_available:
                with tracer.start_as_current_span("step.problem_info_extraction") as step_span:
                        step_span.set_attribute("step.name", "problem_info_extraction")
                        step_span.set_attribute("step.input.question", question)
                        
                        try:
                            extractor = ProblemInfoExtractorTool(model_id=model_id, api_base=api_base, api_key=api_key)
                            extractor.set_session_id(conversation_id)
                            
                            extract_result = extractor.forward(question, conversation_id=conversation_id)
                            step_span.set_attribute("step.output.result", extract_result)
                            print(extract_result)
                            print("=" * 30)
                        except Exception as e:
                            step_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                            step_span.record_exception(e)
                            print(f"Information extraction failed: {e}")
                            print("Continue running agent...")
            else:
                try:
                    
                    extractor = ProblemInfoExtractorTool(model_id=model_id, api_base=api_base, api_key=api_key)
                    extractor.set_session_id(conversation_id)
                    
                    extract_result = extractor.forward(question, conversation_id=conversation_id)
                    print(extract_result)
                    print("=" * 30)
                except Exception as e:
                    print(f"Extraction Failed: {e}")
                    print("Continue running agent...")

            simplified_question = question 


            

            if root_span and trace_available:
                with tracer.start_as_current_span("step.agent_solving") as step_span:
                        step_span.set_attribute("step.name", "agent_solving")
                        step_span.set_attribute("step.input.question", simplified_question)
                        
                        agent = get_coding_agent(model_id=model_id, api_base=api_base, api_key=api_key, tools_config=tools_config, unit_test_cases_on=unit_test_cases_on, preprocess_config=preprocess_config)
                        
                        result = agent.run(simplified_question)
                        
                        step_span.set_attribute("step.output.result", str(result))
            else:
                agent = get_coding_agent(model_id=model_id, api_base=api_base, api_key=api_key, tools_config=tools_config, unit_test_cases_on=unit_test_cases_on, preprocess_config=preprocess_config)

                
                result = agent.run(simplified_question)
            
            tracker_stats = {
                "generated_cases_improved": False,
                "has_generated_cases": False
            }
            
            if hasattr(agent, '_validation_tracker') and agent._validation_tracker:
                tracker = agent._validation_tracker
                
                tracker_stats["generated_cases_improved"] = tracker.check_generated_cases_improvement()
                
                for record in tracker.records:
                    if record.generated_stats and record.generated_stats.get("has_generated_cases"):
                        tracker_stats["has_generated_cases"] = True
                        break
                
                if tracker.check_success_after_failure() and long_term_exp_enabled:

                    
                    try:
                        context = tracker.extract_improvement_context(agent)
                        
                        from agentflow.tools.success_analyzer import SuccessAnalyzer
                        from agentflow.tools.memory_module import get_memory_module
                        
                        model_config = agent._model_config if hasattr(agent, '_model_config') else (model_id, api_base, api_key)
                        analyzer = SuccessAnalyzer(*model_config)
                        
                        success_data = analyzer.analyze(context)
                        
                        if success_data:
                            try:
                                failure_info = context.get('failure', {})
                                success_info = context.get('success', {})
                                code_context = success_info.get('code', '') or failure_info.get('code', '')
                                
                                memory = get_memory_module(
                                    storage_path=config.MEMORY_STORE_PATH,
                                    model_id=model_config[0],
                                    api_base=model_config[1],
                                    api_key=model_config[2],
                                    enable_dedup=config.ENABLE_EXPERIENCE_DEDUP
                                )
                                exp_id, is_new = memory.add_experience(
                                    fix_summary=success_data.get('fix_summary', ''),
                                    original_error=success_data.get('original_error', ''),
                                    key_insight=success_data.get('key_insight', ''),
                                    code_context=code_context,
                                    metadata={
                                        'session_id': conversation_id if 'conversation_id' in locals() else None,
                                        'timestamp': success_data.get('timestamp'),
                                        'root_cause': success_data.get('root_cause', '')
                                    },
                                    error_context=success_data.get('error_context', ''),
                                    error_cause=success_data.get('error_cause', ''),
                                    fix_method=success_data.get('fix_method', ''),
                                    fix_result=success_data.get('fix_result', '')
                                )
                                if is_new:
                                    print(f"✅ New Experience (ID: {exp_id})")
                                else:
                                    print(f"ℹ️ Experience already exists, incremented usage count (ID: {exp_id})")
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                        else:
                            print("⚠ ")
                            
                    except Exception as e:
                        print(f"⚠ : {e}")
                        import traceback
                        traceback.print_exc()
                    
                    print("=" * 60 + "\n")
            
            token_stats = get_token_usage()
            tracker_stats["total_prompt_tokens"] = token_stats.get("total_prompt_tokens", 0)
            tracker_stats["total_completion_tokens"] = token_stats.get("total_completion_tokens", 0)
            tracker_stats["total_tokens"] = token_stats.get("total_tokens", 0)
            
            tracker_stats["step_count"] = len(agent.memory.steps) if hasattr(agent, 'memory') and agent.memory else 0
            
            if long_term_exp_enabled:
                from agentflow.tools.memory_module import get_memory_module
                memory = get_memory_module()
                mem_stats = memory.get_statistics()
                tracker_stats["general_exp_count"] = mem_stats['general']['count']
                tracker_stats["algo_exp_count"] = mem_stats['algorithm']['count']
                tracker_stats["memory_size"] = tracker_stats["general_exp_count"] + tracker_stats["algo_exp_count"]
            else:
                tracker_stats["general_exp_count"] = 0
                tracker_stats["algo_exp_count"] = 0
                tracker_stats["memory_size"] = 0
            
            print(f"📊 Statistics: Steps {tracker_stats['step_count']}, Memory(G/A) {tracker_stats['general_exp_count']}/{tracker_stats['algo_exp_count']}, Token input {tracker_stats['total_prompt_tokens']:,}, Token output {tracker_stats['total_completion_tokens']:,}")
            
            return str(result), tracker_stats
        except Exception as e:
            if root_span and trace_available:
                root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                root_span.record_exception(e)
            raise

_coding_agent = None
_agent_config = None

def get_coding_agent(model_id: str = None, api_base: str = None, api_key: str = None, tools_config: str = None, unit_test_cases_on: bool = None, preprocess_config: str = None):
    global _coding_agent, _agent_config
    
    current_config = (model_id, api_base, api_key, tools_config, unit_test_cases_on, preprocess_config)
    
    if _coding_agent is None or _agent_config != current_config:
        _coding_agent = create_coding_agent(model_id=model_id, api_base=api_base, api_key=api_key, tools_config=tools_config, unit_test_cases_on=unit_test_cases_on, preprocess_config=preprocess_config)
        _agent_config = current_config
    
    return _coding_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute CP-Agent Agent')
    parser.add_argument(
        '--config', 
        '-c',
        type=str,
        default=None,
        help='Example agentflow/configs/o4mini_reflection_only.yaml'
    )
    args = parser.parse_args()
    
    if args.config:
        print(f"📋  {args.config}")
        saved_env_vars = {}
        env_vars_to_clear = ['MODEL_ID', 'API_BASE', 'DEEPSEEK_API_KEY', 'API_KEY']
        for var in env_vars_to_clear:
            if var in os.environ:
                saved_env_vars[var] = os.environ[var]
                del os.environ[var]
        
        config.reload_config(args.config)
        

        print(f"   MODEL_ID: {config.MODEL_ID}")
        print(f"   API_BASE: {config.API_BASE}")
        if saved_env_vars:
            print(f"   ⚠️: {list(saved_env_vars.keys())}")
    else:
        config_path = os.getenv("CONFIG_PATH")
        if config_path:
            config.reload_config(config_path)
    
    logger, log_file = setup_logging()
    

    question = r"""
# Problem: Build the Lexicographically Smallest String

## Constraints
- **Time limit per test:** 1 second  
- **Memory limit per test:** 256 megabytes  

## Statement
You are given an array `a` of `n` strings `a1, a2, ..., an`, each consisting of lowercase English letters, and an initially empty string `s`.

For the `i`-th step (`1 ≤ i ≤ n`), you must do **one** of the following:
1. Add `ai` to the **beginning** of `s`, or  
2. Add `ai` to the **end** of `s`.

### Example (single step)
If before the `i`-th step `s = "aba"` and `ai = "bba"`, then after the step:
- `s` becomes `"ababba"` (add to end), or
- `s` becomes `"bbaaba"` (add to beginning)

## Goal
Find the **lexicographically smallest** string `s` that can be obtained after `n` steps.

## Lexicographical Order Definition
Given two strings `a` and `b` of the same length, `a` is lexicographically smaller than `b` if:
- At the first position where they differ, `a` has a letter that appears **earlier** in the alphabet than the corresponding letter in `b`.

---

## Input
Each test contains multiple test cases.

- The first line contains an integer `t` (`1 ≤ t ≤ 500`) — number of test cases.
- For each test case:
  - The first line contains an integer `n` (`1 ≤ n ≤ 1000`) — size of array `a`.
  - The next line contains `n` strings `a1, a2, ..., an` (`1 ≤ |ai| ≤ 4000`), each consisting of lowercase English letters.

### Guarantees
- The sum of `n` over all test cases does not exceed `1000`.
- The total length of all strings over all test cases does not exceed `4000`.

---

## Output
For each test case, print the lexicographically minimum string `s` you can obtain after all `n` steps.

---

## Example

### Input
3
4
amir rima amin nima
1
codeforces
3
a ab abc

### Output
aminamirrimanima
codeforces
aababc
    """


    MODEL_ID = config.MODEL_ID
    API_BASE = config.API_BASE
    API_KEY = config.API_KEY
    
    print(f"   MODEL_ID: {MODEL_ID}")
    print(f"   API_BASE: {API_BASE}")
    print(f"   API_KEY: {API_KEY[:20]}..." if API_KEY else "   API_KEY: (not set)")

    agent_output = run_coding_agent_with_session(
        question,
        model_id=MODEL_ID,
        api_base=API_BASE,
        api_key=API_KEY
    )

    agent = get_coding_agent(model_id=MODEL_ID, api_base=API_BASE, api_key=API_KEY)
    conversation_record = save_agent_conversation(agent, "agent_conversation.json")
    
    if conversation_record and 'conversation' in conversation_record:
        print(f"Saved {len(conversation_record['conversation'])} conversation records")
    else:
        print("Conversation record save failed")

    print("\nFinal answer:")
    print(agent_output)
    print(f"\n📝 Log Dir: {log_file}")
    logger.info("Finished")