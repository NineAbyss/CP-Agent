
import time
import hashlib
import threading
from typing import Optional


class SessionManager:

    
    def __init__(self):
        self._local = threading.local()
    
    def _get_local_state(self):
        if not hasattr(self._local, 'session_id'):
            self._local.session_id = None
        if not hasattr(self._local, 'session_data'):
            self._local.session_data = {}
        if not hasattr(self._local, 'work_dir'):
            self._local.work_dir = None
        return self._local
    
    def set_session_id(self, session_id: str):
        state = self._get_local_state()
        state.session_id = session_id
        if session_id not in state.session_data:
            state.session_data[session_id] = {}
    
    def get_session_id(self) -> Optional[str]:
        state = self._get_local_state()
        return state.session_id
    
    def generate_session_id(self, content: str = "") -> str:
        timestamp = str(int(time.time() * 1000))
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8] if content else "default"
        return f"session_{timestamp}_{content_hash}"
    
    def clear_session(self):
        state = self._get_local_state()
        state.session_id = None
    
    def set_session_data(self, key: str, value):
        state = self._get_local_state()
        if state.session_id:
            if state.session_id not in state.session_data:
                state.session_data[state.session_id] = {}
            state.session_data[state.session_id][key] = value
    
    def get_session_data(self, key: str, default=None):
        state = self._get_local_state()
        if state.session_id:
            if state.session_id not in state.session_data:
                state.session_data[state.session_id] = {}
            return state.session_data[state.session_id].get(key, default)
        return default

    def set_global_work_dir(self, path: str):
        state = self._get_local_state()
        state.work_dir = path

    def get_global_work_dir(self) -> Optional[str]:
        state = self._get_local_state()
        return state.work_dir


_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    return _session_manager


def set_current_session_id(session_id: str):
    _session_manager.set_session_id(session_id)


def get_current_session_id() -> Optional[str]:
    return _session_manager.get_session_id()


def generate_session_id(content: str = "") -> str:
    return _session_manager.generate_session_id(content)


def clear_current_session():
    _session_manager.clear_session()


def set_run_timeout(timeout_seconds: int):
    _session_manager.set_session_data("run_timeout", timeout_seconds)


def get_run_timeout(default: int = 30) -> int:
    timeout = _session_manager.get_session_data("run_timeout", default)
    return timeout if isinstance(timeout, int) else default


def parse_time_limit(time_limit_str: str) -> int:

    import re
    
    if not time_limit_str:
        return 30  
    

    time_str = time_limit_str.lower().strip()
    
    
    numbers = re.findall(r'\d+(?:\.\d+)?', time_str)
    if not numbers:
        return 30 
    
    time_value = float(numbers[0])
    
    if 'ms' in time_str or 'millisecond' in time_str:
        return max(1, int(time_value / 1000))  
    elif 'minute' in time_str:
        return int(time_value * 60) 
    else:
        return max(1, int(time_value)) 


def set_global_work_dir(path: str):
    _session_manager.set_global_work_dir(path)


def get_global_work_dir() -> Optional[str]:
    return _session_manager.get_global_work_dir()


def set_memory_limit(memory_limit_mb: int):
    _session_manager.set_session_data("memory_limit", memory_limit_mb)


def get_memory_limit(default: int = 256) -> int:
    limit = _session_manager.get_session_data("memory_limit", default)
    return limit if isinstance(limit, int) else default


def set_spj_code(spj_code: str):
    _session_manager.set_session_data("spj_code", spj_code)


def get_spj_code() -> Optional[str]:
    return _session_manager.get_session_data("spj_code")


def parse_memory_limit(memory_limit_str: str) -> int:

    import re
    
    if not memory_limit_str:
        return 256      
    
    mem_str = memory_limit_str.lower().strip()
    
    numbers = re.findall(r'\d+(?:\.\d+)?', mem_str)
    if not numbers:
        return 256 
    
    mem_value = float(numbers[0])
    
    if 'gb' in mem_str or 'gigabyte' in mem_str:
        return int(mem_value * 1024)  
    elif 'kb' in mem_str or 'kilobyte' in mem_str:
        return max(1, int(mem_value / 1024))
    else:
        return max(1, int(mem_value))



def add_token_usage(prompt_tokens: int, completion_tokens: int, source: str = "unknown"):

    stats = _session_manager.get_session_data("token_stats")
    if stats is None:
        stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "call_count": 0,
            "by_source": {}
        }
    
    stats["total_prompt_tokens"] += prompt_tokens
    stats["total_completion_tokens"] += completion_tokens
    stats["call_count"] += 1
    
    if source not in stats["by_source"]:
        stats["by_source"][source] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "call_count": 0
        }
    stats["by_source"][source]["prompt_tokens"] += prompt_tokens
    stats["by_source"][source]["completion_tokens"] += completion_tokens
    stats["by_source"][source]["call_count"] += 1
    
    _session_manager.set_session_data("token_stats", stats)
    
    total = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
    print(f"[TokenStats]  +{prompt_tokens}+{completion_tokens} ({source}) | Accumulated: {total:,} tokens")


def get_token_usage() -> dict:


    stats = _session_manager.get_session_data("token_stats")
    if stats is None:
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "by_source": {}
        }
    
    return {
        "total_prompt_tokens": stats["total_prompt_tokens"],
        "total_completion_tokens": stats["total_completion_tokens"],
        "total_tokens": stats["total_prompt_tokens"] + stats["total_completion_tokens"],
        "call_count": stats["call_count"],
        "by_source": stats["by_source"]
    }


def reset_token_usage():
    _session_manager.set_session_data("token_stats", {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "call_count": 0,
        "by_source": {}
    })
