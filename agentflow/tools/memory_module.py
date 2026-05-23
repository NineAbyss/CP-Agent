import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from smolagents import Tool
from agentflow.tools.tracked_completion import tracked_completion

ALGORITHM_TAXONOMY = {
    "Basic Algorithms": ["Simulation", "Enumeration", "Recursion", "Greedy", "Divide and Conquer", "Binary Search", "Sorting", "Binary Lifting", "Construction", "Prefix Sum and Difference Array"],
    "Dynamic Programming": ["Memoized Search", "Knapsack DP", "Interval DP", "DP on DAG", "Tree DP", "Bitmask DP", "Digit DP", "Counting DP", "Probability DP", "DP Optimization"],
    "Strings": ["String Matching (KMP/Boyer-Moore)", "String Hashing", "Trie", "Aho–Corasick Automaton", "Suffix Array (SA)", "Suffix Automaton (SAM)", "Manacher", "Palindromic Tree"],
    "Mathematics": ["Number Theory", "Combinatorics", "Linear Algebra", "Game Theory", "Probability and Expectation", "Polynomials", "Fast Exponentiation", "Primes", "Divisors", "Modular Arithmetic"],
    "Data Structures": ["Stack", "Queue", "Linked List", "Hash Table", "Disjoint Set Union (DSU)", "Binary Indexed Tree (Fenwick)", "Segment Tree", "Sqrt Decomposition", "Balanced BST", "Tree of Trees", "Persistent Data Structures"],
    "Graph Theory": ["Graph Traversal (DFS/BFS)", "Shortest Path", "Minimum Spanning Tree", "Topological Sorting", "Bipartite Graphs", "Network Flow", "Strongly Connected Components (SCC)", "Biconnected Components", "Lowest Common Ancestor (LCA)", "Tree Problems"],
    "Computational Geometry": ["Vector Operations", "Convex Hull", "Half-Plane Intersection", "Rotating Calipers", "Sweep Line"],
    "Miscellaneous": ["Two Pointers", "Discretization", "Randomization", "Search (Pruning/Heuristics/A*)"]
}

SECONDARY_TO_PRIMARY = {}
for primary, secondaries in ALGORITHM_TAXONOMY.items():
    for secondary in secondaries:
        SECONDARY_TO_PRIMARY[secondary] = primary


class MemoryModule:

    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        model_id: str = None,
        api_base: str = None,
        api_key: str = None,
        enable_dedup: bool = True
    ):

        self.enable_dedup = enable_dedup
        if storage_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.storage_path = os.path.join(project_root, "memory_store.json")
        else:
            self.storage_path = storage_path
        
        self.model_id = model_id or os.getenv("MODEL_ID", "deepseek-chat")
        self.api_base = api_base or os.getenv("API_BASE", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY")
        
        self.general_experiences: List[Dict] = []
        self.algorithm_experiences: Dict[str, Dict[str, List[Dict]]] = {}
        
        for primary, secondaries in ALGORITHM_TAXONOMY.items():
            self.algorithm_experiences[primary] = {}
            for secondary in secondaries:
                self.algorithm_experiences[primary][secondary] = []
        
        self._load()
        

    
    def _count_algorithm_experiences(self) -> int:
        count = 0
        for primary_dict in self.algorithm_experiences.values():
            for exp_list in primary_dict.values():
                count += len(exp_list)
        return count
    
    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.general_experiences = data.get('general_experiences', [])
                    
                    saved_algo = data.get('algorithm_experiences', {})
                    for primary in ALGORITHM_TAXONOMY.keys():
                        if primary in saved_algo:
                            for secondary in ALGORITHM_TAXONOMY[primary]:
                                if secondary in saved_algo[primary]:
                                    self.algorithm_experiences[primary][secondary] = saved_algo[primary][secondary]
                    
                    print(f"[MemoryModule]")
            except Exception as e:
                print(f"[MemoryModule] Failed: {e}")
        else:
            print(f"[MemoryModule]")
    
    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else '.', exist_ok=True)
            
            data = {
                'meta': {
                    'version': '2.0',
                    'last_updated': datetime.now().isoformat(),
                    'general_count': len(self.general_experiences),
                    'algorithm_count': self._count_algorithm_experiences()
                },
                'general_experiences': self.general_experiences,
                'algorithm_experiences': self.algorithm_experiences
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"[MemoryModule] Saving Failed: {e}")
    
    def _generate_experience_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _classify_experience(self, fix_summary: str, original_error: str, code_context: str = "") -> Tuple[str, Optional[str], Optional[str]]:

        taxonomy_text = "\n".join([
            f"- {primary}: [{', '.join(secondaries)}]"
            for primary, secondaries in ALGORITHM_TAXONOMY.items()
        ])
        
        prompt = f"""You are a code experience classification expert. Please analyze the following repair experience and determine which category it belongs to.

## Repair Experience
- Error Description: {original_error}
- Repair Method: {fix_summary}
{f'- Code Context: {code_context[:500]}' if code_context else ''}

## Classification Rules

### 1. General (General Errors)
Applicable to general programming errors that are not specific to any algorithm, such as:
- Syntax errors (missing semicolons, mismatched parentheses, etc.)
- Typo
- Header file/library import problems
- Variable not declared/not initialized
- Type conversion errors
- Input/output format problems
- Memory management problems (not algorithm-related)

### 2. Algorithm (Algorithm Errors)
Applicable to errors that are specific to a particular algorithm, you need to select the corresponding algorithm tag:
{taxonomy_text}

## Format Output
Return JSON:
```json
{{
    "category": "general" or "algorithm",
    "primary_tag": "Primary tag (only required for algorithm type)",
    "secondary_tag": "Secondary tag (only required for algorithm type)",
    "reason": "Reason for classification (short)"
}}
```

Note:
- If the category is general, primary_tag and secondary_tag should be null
- If the category is algorithm, you must select from the above tags (you can choose the closest one)
"""
        
        try:
            response = tracked_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a professional code experience classification expert."},
                    {"role": "user", "content": prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.2,
                max_tokens=300,
                source="memory_classify",
            )
            
            content = response.choices[0].message.content
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                category = result.get("category", "general")
                primary_tag = result.get("primary_tag")
                secondary_tag = result.get("secondary_tag")
                
                if category == "algorithm":
                    if primary_tag not in ALGORITHM_TAXONOMY:
                        print(f"[MemoryModule] Invalid primary tag '{primary_tag}', downgrade to general")
                        return ("general", None, None)
                    if secondary_tag not in ALGORITHM_TAXONOMY.get(primary_tag, []):
                        # Try to find the closest secondary tag
                        print(f"[MemoryModule] Invalid secondary tag '{secondary_tag}', use the first one")
                        secondary_tag = ALGORITHM_TAXONOMY[primary_tag][0]
                
                print(f"[MemoryModule] Classification result: {category}, {primary_tag}, {secondary_tag}")
                return (category, primary_tag, secondary_tag)
            
        except Exception as e:
            print(f"[MemoryModule] Classification failed: {e}, default to general")
        
        return ("general", None, None)
    
    def _check_duplicate(self, new_exp: Dict, existing_exps: List[Dict]) -> Optional[Dict]:

        if not existing_exps:
            return None
        
        existing_summaries = "\n".join([
            f"{i+1}. {exp.get('fix_summary', '')}"
            for i, exp in enumerate(existing_exps[:20])  # At most 20 comparisons
        ])
        
        prompt = f"""You are a code experience deduplication expert. Determine if the new experience is duplicated or highly similar to existing experiences.

## New Experience
{new_exp.get('fix_summary', '')}

## Existing Experiences List
{existing_summaries}

## Judgment Criteria
- If the new experience describes the same problem and solution as an existing experience (even if the wording is different), it is considered duplicated
- If the new experience is a special case or generalization of an existing experience, it is also considered duplicated
- If the new experience involves different problems or different solutions, it is considered not duplicated

## Output Format
Return JSON:
```json
{{
    "is_duplicate": true or false,
    "duplicate_index": The index of the duplicated experience (1-based, if not duplicated fill null),
    "reason": "Reason for judgment"
}}
```
"""
        
        try:
            response = tracked_completion(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a professional code experience deduplication expert."},
                    {"role": "user", "content": prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.2,
                max_tokens=200,
                source="memory_dedup",
            )
            
            content = response.choices[0].message.content
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                if result.get("is_duplicate") and result.get("duplicate_index"):
                    idx = result["duplicate_index"] - 1
                    if 0 <= idx < len(existing_exps):
                        print(f"[MemoryModule] Detected duplicate experience: {result.get('reason', '')}")
                        return existing_exps[idx]
        
        except Exception as e:
            print(f"[MemoryModule]: {e}")
        
        return None
    
    def add_experience(
        self,
        fix_summary: str,
        original_error: str,
        key_insight: Optional[str] = None,
        code_context: str = "",
        metadata: Optional[Dict] = None,
        error_context: Optional[str] = None,
        error_cause: Optional[str] = None,
        fix_method: Optional[str] = None,
        fix_result: Optional[str] = None
    ) -> Tuple[str, bool]:

        category, primary_tag, secondary_tag = self._classify_experience(
            fix_summary, original_error, code_context
        )
        
        exp_id = self._generate_experience_id(f"{fix_summary}_{original_error}")
        new_exp = {
            'experience_id': exp_id,
            'category': category,
            'fix_summary': fix_summary,
            'original_error': original_error,
            'key_insight': key_insight,
            'usage_count': 0, 
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if error_context:
            new_exp['error_context'] = error_context
        if error_cause:
            new_exp['error_cause'] = error_cause
        if fix_method:
            new_exp['fix_method'] = fix_method
        if fix_result:
            new_exp['fix_result'] = fix_result
        
        if category == "algorithm":
            new_exp['primary_tag'] = primary_tag
            new_exp['secondary_tag'] = secondary_tag
        
        if self.enable_dedup:
            if category == "general":
                existing_exps = self.general_experiences
            else:
                existing_exps = self.algorithm_experiences.get(primary_tag, {}).get(secondary_tag, [])
            
            duplicate = self._check_duplicate(new_exp, existing_exps)
            
            if duplicate:

                duplicate['last_used'] = datetime.now().isoformat()
                self._save()
                print(f"[MemoryModule] Experience already exists, skip adding (search count: {duplicate.get('usage_count', 0)})")
                return (duplicate['experience_id'], False)
        else:
            print(f"[MemoryModule] Deduplication is disabled, add experience directly")
        
        if category == "general":
            self.general_experiences.append(new_exp)
        else:
            self.algorithm_experiences[primary_tag][secondary_tag].append(new_exp)
        
        self._save()
        print(f"[MemoryModule]  [{category}]: {fix_summary[:50]}...")
        return (exp_id, True)
    
    def get_general_experiences_text(self, limit: int = 10) -> str:

        if not self.general_experiences:
            return ""
        
        sorted_exps = sorted(
            self.general_experiences,
            key=lambda x: x.get('usage_count', 0),
            reverse=True
        )[:limit]
        
        lines = [
            "[General Programming Experience]",
            "The following are the general programming experiences accumulated over time, please pay attention when coding:",
            ""
        ]
        
        for idx, exp in enumerate(sorted_exps, 1):
            lines.append(f"{idx}. {exp['fix_summary']}")
            if exp.get('key_insight'):
                lines.append(f"{exp['key_insight']}")
        
        return "\n".join(lines)
    
    def get_algorithm_experiences(
        self,
        primary_tag: str,
        secondary_tag: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:

        results = []
        
        if primary_tag not in self.algorithm_experiences:
            return results
        
        if secondary_tag:
            exps = self.algorithm_experiences[primary_tag].get(secondary_tag, [])
            sorted_exps = sorted(exps, key=lambda x: x.get('usage_count', 0), reverse=True)
            results.extend(sorted_exps[:limit])
        else:
            for sec_tag, exps in self.algorithm_experiences[primary_tag].items():
                sorted_exps = sorted(exps, key=lambda x: x.get('usage_count', 0), reverse=True)
                results.extend(sorted_exps[:limit])
        
        return results
    
    def get_algorithm_experiences_text(
        self,
        primary_tag: str,
        secondary_tag: Optional[str] = None,
        limit: int = 5
    ) -> str:

        exps = self.get_algorithm_experiences(primary_tag, secondary_tag, limit)
        
        if not exps:
            return ""
        
        tag_display = f"{primary_tag}" + (f" - {secondary_tag}" if secondary_tag else "")
        
        lines = [
            f"[Algorithm Experience for {tag_display}]",
            f"The following are the experiences accumulated when solving {tag_display} related problems:",
            ""
        ]
        
        for idx, exp in enumerate(exps, 1):
            error_context = exp.get('error_context')
            error_cause = exp.get('error_cause')
            key_insight = exp.get('key_insight')
            
            if error_context and error_cause and key_insight:
                lines.append(f"{idx}. ⚠️ When {error_context}, you should avoid: {error_cause}")
                lines.append(f"   💡 You can learn from: {key_insight}")
                if exp.get('fix_method'):
                    lines.append(f"   🔧 Fix method: {exp['fix_method']}")
            else:
                lines.append(f"{idx}. {exp.get('fix_summary', '未知经验')}")
                if key_insight:
                    lines.append(f"   💡 {key_insight}")
            
            lines.append("") 
        
        return "\n".join(lines)
    
    def increment_usage(self, experience_id: str):

        for exp in self.general_experiences:
            if exp.get('experience_id') == experience_id:
                exp['usage_count'] = exp.get('usage_count', 0) + 1
                exp['last_used'] = datetime.now().isoformat()
                self._save()
                return
        
        for primary_dict in self.algorithm_experiences.values():
            for exp_list in primary_dict.values():
                for exp in exp_list:
                    if exp.get('experience_id') == experience_id:
                        exp['usage_count'] = exp.get('usage_count', 0) + 1
                        exp['last_used'] = datetime.now().isoformat()
                        self._save()
                        return
    
    def get_statistics(self) -> Dict:
        general_usage = sum(e.get('usage_count', 0) for e in self.general_experiences)
        
        algo_count = 0
        algo_usage = 0
        algo_by_primary = {}
        
        for primary, secondary_dict in self.algorithm_experiences.items():
            primary_count = 0
            for exps in secondary_dict.values():
                primary_count += len(exps)
                algo_usage += sum(e.get('usage_count', 0) for e in exps)
            algo_by_primary[primary] = primary_count
            algo_count += primary_count
        
        return {
            'general': {
                'count': len(self.general_experiences),
                'total_usage': general_usage
            },
            'algorithm': {
                'count': algo_count,
                'total_usage': algo_usage,
                'by_primary': algo_by_primary
            }
        }



class AlgorithmExperienceRetrieverTool(Tool):

    
    name = "retrieve_algorithm_experience"
    description = """
Retrieve historical fix experiences for a specific algorithm type.

When you need to solve an algorithm-related problem, you can use this tool to obtain past experiences for that algorithm category.
These experiences come from previously solved similar problems and may help with the current problem.

Algorithm taxonomy:
- Basic Algorithms: Simulation, Enumeration, Recursion, Greedy, Divide and Conquer, Binary Search, Sorting, Binary Lifting, Construction, Prefix Sum and Difference Array
- Dynamic Programming: Memoized Search, Knapsack DP, Interval DP, DP on DAG, Tree DP, Bitmask DP, Digit DP, Counting DP, Probability DP, DP Optimization
- Strings: String Matching, String Hashing, Trie, Aho–Corasick Automaton, Suffix Array, Manacher, Palindromic Tree
- Mathematics: Number Theory, Combinatorics, Linear Algebra, Game Theory, Probability and Expectation, Fast Exponentiation, Primes, Divisors, Modular Arithmetic
- Data Structures: Stack, Queue, Linked List, Hash Table, Disjoint Set Union (DSU), Binary Indexed Tree (Fenwick), Segment Tree, Sqrt Decomposition, Balanced BST
- Graph Theory: Graph Traversal (DFS/BFS), Shortest Path, Minimum Spanning Tree, Topological Sorting, Bipartite Graphs, Network Flow, Strongly Connected Components (SCC), Lowest Common Ancestor (LCA)
- Computational Geometry: Vector Operations, Convex Hull, Half-Plane Intersection, Rotating Calipers, Sweep Line
- Miscellaneous: Two Pointers, Discretization, Randomization, Search (Pruning/Heuristics/A*)

    """
    
    inputs = {
        "primary_tag": {
            "type": "string",
            "description": "Primary algorithm tag, such as 'Dynamic Programming', 'Graph Theory', 'Data Structure' etc."
        },
        "secondary_tag": {
            "type": "string",
            "description": "Secondary algorithm tag (optional), such as 'Knapsack DP', 'Shortest Path' etc.",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, memory_module: MemoryModule = None, **kwargs):
        super().__init__(**kwargs)
        self._memory_module = memory_module
    
    def _get_memory_module(self) -> MemoryModule:
        if self._memory_module is None:
            self._memory_module = get_memory_module()
        return self._memory_module
    
    def forward(self, primary_tag: str, secondary_tag: Optional[str] = None) -> str:

        memory = self._get_memory_module()
        
        if primary_tag not in ALGORITHM_TAXONOMY:
            return f"Primary tag '{primary_tag}' not found. Available primary tags: {', '.join(ALGORITHM_TAXONOMY.keys())}"
        
        if secondary_tag and secondary_tag not in ALGORITHM_TAXONOMY.get(primary_tag, []):
            available = ', '.join(ALGORITHM_TAXONOMY[primary_tag])
            return f"Secondary tag '{secondary_tag}' not found. Available secondary tags for '{primary_tag}': {available}"
        
        result = memory.get_algorithm_experiences_text(primary_tag, secondary_tag, limit=5)
        
        if not result:
            tag_display = f"{primary_tag}" + (f" - {secondary_tag}" if secondary_tag else "")
            return f"No historical experiences related to [{tag_display}]."
        
        exps = memory.get_algorithm_experiences(primary_tag, secondary_tag, limit=5)
        for exp in exps:
            memory.increment_usage(exp['experience_id'])
        
        return result



_memory_module_instance: Optional[MemoryModule] = None


def get_memory_module(
    storage_path: Optional[str] = None,
    model_id: str = None,
    api_base: str = None,
    api_key: str = None,
    enable_dedup: bool = True
) -> MemoryModule:

    global _memory_module_instance
    
    if _memory_module_instance is None:
        _memory_module_instance = MemoryModule(
            storage_path=storage_path,
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            enable_dedup=enable_dedup
        )
    
    return _memory_module_instance


def clear_memory_module():
    global _memory_module_instance
    _memory_module_instance = None


def create_algorithm_experience_tool(
    memory_module: MemoryModule = None,
    model_id: str = None,
    api_base: str = None,
    api_key: str = None
) -> AlgorithmExperienceRetrieverTool:
    if memory_module is None:
        memory_module = get_memory_module(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key
        )
    return AlgorithmExperienceRetrieverTool(memory_module=memory_module)



