"""
基于Attempt的Trajectory管理器

核心思想：
1. 每次代码生成+执行 = 一个Attempt
2. Attempt结束后立即生成LLM摘要
3. 构建下一个Attempt的上下文时：
   - 旧Attempts: 只显示summary（1-2句话）
   - 最近N个Attempts: 显示完整信息（thinking + code + result）

优势：
- 实时压缩，避免事后批量处理
- 精准保留关键信息
- 大幅减少token使用（节省60-80%）
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from agentflow.tools.tracked_completion import tracked_completion
import json
import os
import re


class InterpreterVerificationLog:
    """
    CppInterpreterTool 验证记录
    
    与 AttemptLog 不同，验证记录没有"成功/失败"之分
    用于记录验证初步结论的过程
    
    包含：
    1. llm_output - LLM 的完整输出（思考过程 + 代码调用）
    2. 验证代码（code）
    3. 验证结果（result）- 工具返回的实际输出
    4. 摘要（summary）- LLM 总结的验证目的和结论
    """
    
    def __init__(self, verification_id: int):
        self.verification_id = verification_id
        self.llm_output: Optional[str] = None  # LLM 的完整输出
        self.code: Optional[str] = None
        self.result: Optional[str] = None  # 验证结果
        self.summary: Optional[str] = None
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            'verification_id': self.verification_id,
            'llm_output': self.llm_output,
            'code': self.code,
            'result': self.result,
            'summary': self.summary,
            'timestamp': self.timestamp
        }
    
    def get_compressed_view(self) -> str:
        """返回压缩视图"""
        return f"🔬 验证 {self.verification_id}: {self.summary or '无摘要'}"
    
    def get_full_view(self) -> str:
        """返回完整视图"""
        parts = [f"{'='*40}"]
        parts.append(f"🔬 验证 {self.verification_id}")
        parts.append(f"{'='*40}")
        
        if self.llm_output:
            llm_output = self.llm_output.strip()
            if len(llm_output) > 500:
                llm_output = llm_output[:500] + "\n... [输出截断]"
            parts.append(f"\n💭 LLM输出:\n{llm_output}")
        
        if self.code:
            code = self.code.strip()
            if len(code) > 500:
                code = code[:500] + "\n... [代码截断]"
            parts.append(f"\n💻 验证代码:\n```cpp\n{code}\n```")
        
        if self.result:
            result = self.result.strip()
            if len(result) > 300:
                result = result[:300] + "\n... [结果截断]"
            parts.append(f"\n📊 验证结果:\n{result}")
        
        return '\n'.join(parts)


class AttemptLog:
    """
    单次尝试（Attempt）的日志记录 - 用于 CppValidationTool
    
    一个Attempt包含：
    1. 思考过程（thinking）
    2. 生成的代码（code）
    3. 执行结果（execution_result）
    4. LLM生成的简洁摘要（summary）← 关键！
    """
    
    def __init__(self, attempt_id: int):
        self.attempt_id = attempt_id
        self.thinking: Optional[str] = None
        self.code: Optional[str] = None
        self.execution_result: Optional[str] = None
        self.is_success: Optional[bool] = None
        self.error_type: Optional[str] = None
        self.summary: Optional[str] = None  # LLM生成的摘要
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            'attempt_id': self.attempt_id,
            'thinking': self.thinking,
            'code': self.code,
            'execution_result': self.execution_result,
            'is_success': self.is_success,
            'error_type': self.error_type,
            'summary': self.summary,
            'timestamp': self.timestamp
        }
    
    def get_compressed_view(self) -> str:
        """
        返回压缩视图（只有摘要）
        
        格式: Attempt N: ✅/❌ 摘要内容
        """
        status = "成功尝试" if self.is_success else "错误尝试"
        return f"Attempt {self.attempt_id}: {status} {self.summary or '未生成摘要'}"
    
    def get_full_view(self) -> str:
        """
        返回完整视图
        
        包含：thinking + code + execution_result
        """
        parts = [f"{'='*60}"]
        parts.append(f"Attempt {self.attempt_id}")
        parts.append(f"{'='*60}")
        
        # 1. 思考过程
        if self.thinking:
            thinking = self.thinking.strip()

            parts.append(f"\n{thinking}")
        
        # 2. 代码
        if self.code:
            code = self.code.strip()
            if len(code) > 2000:
                # 代码太长，提取关键部分
                parts.append(f"\n📝 代码: [长度{len(code)}字符，显示关键部分]")
                parts.append("```cpp")
                parts.append(self._extract_key_code_parts(code))
                parts.append("```")
            else:
                parts.append(f"\n📝 代码:\n```cpp\n{code}\n```")
        
        # 3. 执行结果
        if self.execution_result:
            result = self.execution_result.strip()
            if len(result) > 1000:
                result = self._simplify_execution_result(result)
            
            # status_emoji = "✅" if self.is_success else "❌"
            status_text = "成功" if self.is_success else f"失败 ({self.error_type})"
            parts.append(f"\n执行结果: {status_text}")
            parts.append(result)
        
        return '\n'.join(parts)
    
    def _extract_key_code_parts(self, code: str) -> str:
        """提取代码的关键部分（头部+尾部）"""
        lines = code.split('\n')
        if len(lines) <= 60:
            return code
        
        # 保留前30行和后20行
        key_lines = lines
        return '\n'.join(key_lines)
    
    def _simplify_execution_result(self, result: str) -> str:
        """简化执行结果（保留关键信息）"""
        lines = result.split('\n')
        
        # 提取关键行（包含测试结果、错误信息）
        key_indicators = ['✅', '❌', 'error', 'Error', 'ERROR', 'passed', 'failed', 
                         'PASSED', 'FAILED', 'Test', 'Sample', '样例']
        key_lines = []
        
        for line in lines:
            if any(indicator in line for indicator in key_indicators):
                key_lines.append(line)
        
        if key_lines:
            # 如果提取到关键行，保留前20行
            summary = '\n'.join(key_lines)

            return summary
        
        # 如果没有关键行，保留前500字符
        return result


class AttemptBasedTrajectoryManager:
    """
    基于Attempt的Trajectory管理器
    
    核心功能：
    1. 记录每个Attempt的完整信息
    2. 每个Attempt结束后立即生成LLM摘要
    3. 构建上下文时自动使用压缩/完整视图
    4. 持久化到文件系统
    
    使用方式：
        manager = AttemptBasedTrajectoryManager(...)
        
        # 开始新attempt
        attempt = manager.start_new_attempt()
        manager.record_thinking("...")
        manager.record_code("...")
        manager.record_execution_result("...", is_success=True)
        manager.finalize_current_attempt()  # 自动生成摘要
        
        # 构建下一次的上下文
        context = manager.build_context_for_next_attempt()
    """
    
    def __init__(
        self, 
        model_id: str,
        api_base: str,
        api_key: str,
        work_dir: str,
        keep_full_attempts: int = 1,  # 保留最近N个attempt的完整信息
        enable_summary: bool = True   # 是否启用LLM摘要
    ):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        self.work_dir = work_dir
        self.keep_full_attempts = keep_full_attempts
        self.enable_summary = enable_summary
        
        self.attempts: List[AttemptLog] = []
        self.current_attempt: Optional[AttemptLog] = None
        self.amp_retrievals: List[Dict] = []  # 追踪AMP检索历史
        
        # 🔬 Interpreter 验证记录（与 attempts 分开管理）
        self.verifications: List[InterpreterVerificationLog] = []
        self.current_verification: Optional[InterpreterVerificationLog] = None
        
        # 创建日志目录
        self.log_dir = os.path.join(work_dir, "attempt_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"[AttemptManager] ✅ 初始化完成")
        print(f"[AttemptManager]    日志目录: {self.log_dir}")
        print(f"[AttemptManager]    保留完整信息: 最近{keep_full_attempts}个attempts")
        print(f"[AttemptManager]    LLM摘要: {'启用' if enable_summary else '禁用'}")
    
    def start_new_attempt(self) -> AttemptLog:
        """
        开始新的Attempt
        
        Returns:
            新创建的AttemptLog对象
        """
        attempt_id = len(self.attempts) + 1
        self.current_attempt = AttemptLog(attempt_id)
        
        print(f"\n{'='*60}")
        print(f"🚀 开始 Attempt {attempt_id}")
        print(f"{'='*60}")
        
        return self.current_attempt
    
    def record_thinking(self, thinking: str):
        """记录思考过程"""
        if self.current_attempt:
            self.current_attempt.thinking = thinking
            print(f"[Attempt {self.current_attempt.attempt_id}] 📝 已记录思考过程 ({len(thinking)}字符)")
    
    def record_code(self, code: str):
        """记录生成的代码"""
        if self.current_attempt:
            self.current_attempt.code = code
            print(f"[Attempt {self.current_attempt.attempt_id}] 💻 已记录代码 ({len(code)}字符)")
    
    def record_execution_result(self, result: str, is_success: bool, error_type: str = None):
        """
        记录执行结果
        
        Args:
            result: 执行结果文本
            is_success: 是否成功
            error_type: 错误类型（如果失败）
        """
        if self.current_attempt:
            self.current_attempt.execution_result = result
            self.current_attempt.is_success = is_success
            self.current_attempt.error_type = error_type
            
            status = "✅ 成功" if is_success else f"❌ 失败 ({error_type})"
            print(f"[Attempt {self.current_attempt.attempt_id}] 🔍 执行完成: {status}")
    
    def finalize_current_attempt(self):
        """
        完成当前Attempt
        
        关键步骤：
        1. 调用LLM生成简洁摘要
        2. 保存到历史记录
        3. 持久化到文件
        """
        if not self.current_attempt:
            print("[AttemptManager] ⚠️  没有正在进行的attempt")
            return
        
        attempt_id = self.current_attempt.attempt_id
        
        # 1. 生成摘要
        if self.enable_summary:
            print(f"[Attempt {attempt_id}] 🤖 正在生成摘要...")
            summary = self._generate_summary(self.current_attempt)
            self.current_attempt.summary = summary
            print(f"[Attempt {attempt_id}] 📋 摘要: {summary}")
        else:
            # 不使用LLM，生成简单摘要
            self.current_attempt.summary = self._generate_simple_summary(self.current_attempt)
        
        # 2. 保存到历史
        self.attempts.append(self.current_attempt)
        
        # 3. 持久化
        self._save_attempt_log(self.current_attempt)
        
        # 4. 重置当前attempt
        self.current_attempt = None
        
        print(f"[Attempt {attempt_id}] ✅ 已完成并保存")
        print(f"{'='*60}\n")
    
    # ========== Interpreter 验证记录方法 ==========
    
    def start_new_verification(self) -> InterpreterVerificationLog:
        """
        开始新的 Interpreter 验证记录
        
        Returns:
            新创建的 InterpreterVerificationLog 对象
        """
        verification_id = len(self.verifications) + 1
        self.current_verification = InterpreterVerificationLog(verification_id)
        
        print(f"\n{'='*40}")
        print(f"🔬 开始验证 {verification_id}")
        print(f"{'='*40}")
        
        return self.current_verification
    
    def record_llm_output(self, llm_output: str):
        """记录 LLM 的完整输出"""
        if self.current_verification:
            self.current_verification.llm_output = llm_output
            print(f"[验证 {self.current_verification.verification_id}] 💭 已记录LLM输出 ({len(llm_output)}字符)")
    
    def record_verification_code(self, code: str):
        """记录验证代码"""
        if self.current_verification:
            self.current_verification.code = code
            print(f"[验证 {self.current_verification.verification_id}] 💻 已记录验证代码 ({len(code)}字符)")
    
    def record_verification_result(self, result: str):
        """记录验证结果"""
        if self.current_verification:
            self.current_verification.result = result
            print(f"[验证 {self.current_verification.verification_id}] 📊 已记录验证结果")
    
    def finalize_current_verification(self):
        """
        完成当前验证记录
        
        生成摘要格式：验证[目的]，结果[结论]
        """
        if not self.current_verification:
            print("[AttemptManager] ⚠️  没有正在进行的验证")
            return
        
        verification_id = self.current_verification.verification_id
        
        # 生成验证摘要
        if self.enable_summary:
            print(f"[验证 {verification_id}] 🤖 正在生成验证摘要...")
            summary = self._generate_verification_summary(self.current_verification)
            self.current_verification.summary = summary
            print(f"[验证 {verification_id}] 📋 摘要: {summary}")
        else:
            self.current_verification.summary = self._generate_simple_verification_summary(self.current_verification)
        
        # 保存到历史
        self.verifications.append(self.current_verification)
        
        # 重置当前验证
        self.current_verification = None
        
        print(f"[验证 {verification_id}] ✅ 验证记录已保存")
        print(f"{'='*40}\n")
    
    def _generate_verification_summary(self, verification: InterpreterVerificationLog) -> str:
        """
        使用 LLM 生成验证摘要
        
        输入完整的上下文（LLM输出 + 代码 + 工具返回结果），让 LLM 总结验证目的和结论
        """
        prompt = f"""请分析这次代码验证过程，总结验证目的和结论。

## Agent 的思考和代码调用：
{verification.llm_output[:1500] if verification.llm_output else '无'}

## 验证代码：
```cpp
{verification.code[:1000] if verification.code else '无代码'}
```

## 工具返回的执行结果：
{verification.result[:500] if verification.result else '无结果'}

## 要求：
- 分析 Agent 想要验证什么（验证目的）
- 根据执行结果判断得出了什么结论
- 用一句话总结，格式："验证[目的]，结果[结论]"
- 不超过50字
- 客观描述，不要说"成功"或"失败"
"""
        
        try:
            response = tracked_completion(
                model=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个代码分析专家。请简洁地总结验证过程和结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                source="verification_summary",
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[AttemptManager] ⚠️  验证摘要生成失败: {e}")
            return self._generate_simple_verification_summary(verification)
    
    def _generate_simple_verification_summary(self, verification: InterpreterVerificationLog) -> str:
        """规则生成简单验证摘要（LLM调用失败时的降级方案）"""
        result_preview = verification.result[:80] if verification.result else "无输出"
        return f"验证代码片段，输出: {result_preview}"
    
    def get_verifications_summary(self) -> str:
        """获取所有验证记录的摘要"""
        if not self.verifications:
            return ""
        
        lines = ["### 验证记录"]
        for v in self.verifications:
            lines.append(v.get_compressed_view())
        
        return "\n".join(lines)
    
    def get_mixed_history_summary(self, keep_full_recent: int = 1) -> str:
        """
        获取混合的 attempts 和 verifications 历史摘要
        
        按时间顺序（attempt_id/verification_id）排列，排除最近 keep_full_recent 个记录
        和第一个记录（第一个记录完整保留，不出现在摘要中）
        
        Args:
            keep_full_recent: 保留多少个最近的记录不被压缩
            
        Returns:
            格式化的历史摘要字符串
        """
        # 合并所有记录
        all_records = []
        
        for attempt in self.attempts:
            all_records.append({
                'type': 'attempt',
                'id': attempt.attempt_id,
                'record': attempt,
                'timestamp': attempt.timestamp
            })
        
        for verification in self.verifications:
            all_records.append({
                'type': 'verification', 
                'id': verification.verification_id,
                'record': verification,
                'timestamp': verification.timestamp
            })
        
        if not all_records:
            return ""
        
        # 按时间戳排序
        all_records.sort(key=lambda x: x['timestamp'])
        
        total_records = len(all_records)
        if total_records <= keep_full_recent:
            # 记录太少，不需要摘要
            return ""
        
        # 跳过第一个记录（完整保留）和最后 keep_full_recent 个记录
        # 只对中间的记录生成摘要
        start_idx = 1  # 跳过第一个
        end_idx = total_records - keep_full_recent
        
        if start_idx >= end_idx:
            # 没有需要压缩的记录
            return ""
        
        lines = []
        for i in range(start_idx, end_idx):
            item = all_records[i]
            record = item['record']
            lines.append(record.get_compressed_view())
        
        return "\n".join(lines)
    
    # ========== 原有方法继续 ==========
    
    def _generate_summary(self, attempt: AttemptLog) -> str:
        """
        使用LLM生成简洁摘要
        
        摘要要求：
        - 成功: 描述核心算法/方法（1句话）
        - 失败: 描述错误原因（1-2句话）
        - 长度控制在50字以内
        """
        prompt = self._build_summary_prompt(attempt)
        
        try:
            response = tracked_completion(
                model=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个代码审查专家。请总结代码尝试的核心内容。并根据报错内容，描述具体的错误原因，以供编程竞赛选手反思。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                source="attempt_summary",
            )
            
            summary = response.choices[0].message.content.strip()
            
          
            
            return summary
            
        except Exception as e:
            print(f"[AttemptManager] ⚠️  LLM摘要生成失败: {e}")
            # 降级到规则方法
            return self._generate_simple_summary(attempt)
    
    def _build_summary_prompt(self, attempt: AttemptLog) -> str:
        """构建摘要提示词"""
        parts = []
        
        # 1. 思路摘要
        if attempt.thinking:
            thinking_snippet = attempt.thinking  
            parts.append(f"思路: {thinking_snippet}")
        
        # 2. 代码关键信息
        if attempt.code:
            code_info = self._extract_code_keywords(attempt.code)
            parts.append(f"实现: {code_info}")
        
        # 3. 结果
        status = "成功" if attempt.is_success else f"失败 - {attempt.error_type}"
        parts.append(f"结果: {status}")
        
        if not attempt.is_success and attempt.execution_result:
            error_snippet = attempt.execution_result
            parts.append(f"错误信息: {error_snippet}")
        
        prompt = f"""请总结这次尝试：

{chr(10).join(parts)}

要求：
- 如果成功：说明用了什么算法/数据结构
- 如果失败：说明遇到什么错误
- 不要重复代码细节
- 不遗漏重要信息的前提下，尽量简洁、准确"""
        
        return prompt
    
    def _extract_code_keywords(self, code: str) -> str:
        """提取代码关键词（算法、数据结构）"""
        keywords = []
        code_lower = code.lower()
        
        # 算法模式
        if 'sort' in code_lower:
            keywords.append('排序')
        if 'dp' in code_lower or 'memo' in code_lower:
            keywords.append('动态规划')
        if 'dfs' in code_lower:
            keywords.append('DFS')
        if 'bfs' in code_lower:
            keywords.append('BFS')
        if 'binary' in code_lower and 'search' in code_lower:
            keywords.append('二分查找')
        if 'greedy' in code_lower or '贪心' in code:
            keywords.append('贪心')
        
        # 数据结构
        if 'vector' in code_lower:
            keywords.append('数组')
        if 'map' in code_lower or 'unordered_map' in code_lower:
            keywords.append('哈希表')
        if 'set' in code_lower:
            keywords.append('集合')
        if 'priority_queue' in code_lower:
            keywords.append('优先队列')
        if 'stack' in code_lower:
            keywords.append('栈')
        if 'queue' in code_lower:
            keywords.append('队列')
        
        return ', '.join(keywords[:4]) if keywords else '基础实现'
    
    def _generate_simple_summary(self, attempt: AttemptLog) -> str:
        """
        规则生成简单摘要（不使用LLM的降级方案）
        """
        if attempt.is_success:
            keywords = self._extract_code_keywords(attempt.code or "")
            return f"成功实现 ({keywords})"
        else:
            error_type = attempt.error_type or "未知错误"
            return f"{error_type}"
    
    def _save_attempt_log(self, attempt: AttemptLog):
        """持久化attempt日志"""
        try:
            # 1. 保存单个attempt
            log_file = os.path.join(self.log_dir, f"attempt_{attempt.attempt_id}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(attempt.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 2. 更新汇总文件
            summary_file = os.path.join(self.log_dir, "all_attempts.json")
            all_attempts = [a.to_dict() for a in self.attempts]
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_attempts, f, ensure_ascii=False, indent=2)
            
            # 3. 生成可读的markdown摘要
            self._save_markdown_summary()
                
        except Exception as e:
            print(f"[AttemptManager] ⚠️  保存日志失败: {e}")
    
    def _save_markdown_summary(self):
        """生成可读的markdown摘要文件"""
        try:
            md_file = os.path.join(self.log_dir, "attempts_summary.md")
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write("# Attempts Summary\n\n")
                f.write(f"总尝试次数: {len(self.attempts)}\n\n")
                
                for attempt in self.attempts:
                    f.write(f"## Attempt {attempt.attempt_id}\n\n")
                    f.write(f"- 状态: {'✅ 成功' if attempt.is_success else f'❌ 失败 ({attempt.error_type})'}\n")
                    f.write(f"- 摘要: {attempt.summary}\n")
                    f.write(f"- 时间: {attempt.timestamp}\n\n")
                    
                    if not attempt.is_success and attempt.execution_result:
                        error_line = attempt.execution_result.split('\n')[0]
                        f.write(f"- 错误: {error_line}\n\n")
        except Exception as e:
            print(f"[AttemptManager] ⚠️  保存markdown摘要失败: {e}")
    
    def build_context_for_next_attempt(self) -> str:
        """
        构建下一个Attempt的上下文
        
        策略：
        - 验证记录（interpreter）: 显示摘要
        - 早期attempts（validation）: 只显示摘要（1行）
        - 最近N个attempts: 显示完整信息
        
        Returns:
            格式化的上下文字符串
        """
        if not self.attempts and not self.verifications:
            return ""
        
        parts = []
        
        # 0. 验证记录摘要（来自 cpp_interpreter）
        if self.verifications:
            parts.append("### 验证记录（初步结论验证）\n")
            for v in self.verifications:
                parts.append(v.get_compressed_view())
            parts.append("")  # 空行
        
        if not self.attempts:
            return '\n'.join(parts)
        
        # 分组
        old_attempts = self.attempts[:-self.keep_full_attempts] if len(self.attempts) > self.keep_full_attempts else []
        recent_attempts = self.attempts[-self.keep_full_attempts:]
        
        # 1. 早期attempts的压缩视图
        if old_attempts:
            parts.append("### 之前的提交尝试总结\n")
            for attempt in old_attempts:
                parts.append(attempt.get_compressed_view())
            parts.append("")  # 空行
        
        # 2. 最近attempts的完整视图
        if recent_attempts:
            parts.append("### 最近提交尝试\n")
            for attempt in recent_attempts:
                parts.append(attempt.get_full_view())
                parts.append("")  # 空行
        
        # 3. 指导性提示
        last_attempt = self.attempts[-1]
        next_id = len(self.attempts) + 1
        
        if not last_attempt.is_success:
            parts.append(f"**→ 请修复 Attempt {last_attempt.attempt_id} 中的问题，生成 Attempt {next_id}**")
        else:
            parts.append(f"**→ Attempt {last_attempt.attempt_id} 已成功！如需优化可继续改进。**")
        
        return '\n'.join(parts)
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            {
                'total_attempts': 总尝试次数,
                'successful_attempts': 成功次数,
                'failed_attempts': 失败次数,
                'error_distribution': 错误类型分布,
                'average_code_length': 平均代码长度
            }
        """
        total = len(self.attempts)
        success = sum(1 for a in self.attempts if a.is_success)
        failed = total - success
        
        # 错误类型分布
        error_types = {}
        for a in self.attempts:
            if not a.is_success and a.error_type:
                error_types[a.error_type] = error_types.get(a.error_type, 0) + 1
        
        # 平均代码长度
        code_lengths = [len(a.code) for a in self.attempts if a.code]
        avg_code_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0
        
        return {
            'total_attempts': total,
            'successful_attempts': success,
            'failed_attempts': failed,
            'success_rate': f"{success/total*100:.1f}%" if total > 0 else "0%",
            'error_distribution': error_types,
            'average_code_length': int(avg_code_length)
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print("📊 Attempt统计信息")
        print(f"{'='*60}")
        print(f"  总尝试次数: {stats['total_attempts']}")
        print(f"  成功: {stats['successful_attempts']}, 失败: {stats['failed_attempts']}")
        print(f"  成功率: {stats['success_rate']}")
        print(f"  平均代码长度: {stats['average_code_length']} 字符")
        
        if stats['error_distribution']:
            print(f"\n  错误类型分布:")
            for error_type, count in stats['error_distribution'].items():
                print(f"    - {error_type}: {count}次")
        
        print(f"{'='*60}\n")

    def record_amp_retrieval(self, query: str, result: str, current_problem: str = ""):
        """记录一次AMP检索，并提取已有的题解分析总结"""
        # 1. 首先尝试从结果中提取已有的【题解 X 分析总结】
        is_useful = False
        summary = self._extract_amp_summary(result)
        
        if summary:
            print(f"[AttemptManager] ✅ 从 AMP 结果中提取到已有的题解分析总结")
            is_useful = True  # 有总结说明检索成功
        elif self.enable_summary:
            # 如果没有已有总结，回退到 LLM 生成
            print(f"[AttemptManager] 🤖 未找到已有总结，使用 LLM 生成摘要...")
            summary, is_useful = self._generate_amp_summary(query, result, current_problem)
        else:
            summary = "检索完成"
        
        retrieval_record = {
            'id': len(self.amp_retrievals) + 1,
            'query': query,
            'result': result,
            'summary': summary,
            'is_useful': is_useful,
            'timestamp': datetime.now().isoformat()
        }
        self.amp_retrievals.append(retrieval_record)
        print(f"[AttemptManager] 📚 已记录 AMP 检索 (ID: {retrieval_record['id']}) - 有用: {is_useful}")
        summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
        print(f"[AttemptManager] 📋 AMP 摘要预览: {summary_preview}")

    def _extract_amp_summary(self, result: str) -> str:
        """
        从 amp_retrieval 的 lookup 结果中提取已有的题解分析总结
        
        支持的格式：
        - 单题解：【例题分析总结】
        - 多题解：【题解 1 分析总结】、【题解 2 分析总结】、【题解 3 分析总结】
        
        Returns:
            提取到的所有总结内容（用分隔线连接），如果没有找到则返回空字符串
        """
        import re
        
        summaries = []
        
        # 1. 尝试提取多题解格式：【题解 X 分析总结】
        # 匹配 【题解 X 分析总结】 后面直到下一个 【题解 或 文件结尾
        multi_pattern = r'【题解\s*(\d+)\s*分析总结】\s*([\s\S]*?)(?=【题解\s*\d+\s*分析总结】|【题解\s*\d+/\d+】|$)'
        multi_matches = re.findall(multi_pattern, result)
        
        if multi_matches:
            for idx, content in multi_matches:
                # 清理内容：去掉末尾的分隔线和空白
                content = content.strip()
                # 去掉末尾的 "---" 分隔线
                content = re.sub(r'\n*-{3,}\s*$', '', content).strip()
                if content:
                    summaries.append(f"【题解 {idx} 分析总结】\n{content}")
        
        # 2. 如果没有找到多题解格式，尝试单题解格式：【例题分析总结】
        if not summaries:
            single_pattern = r'【例题分析总结】\s*([\s\S]*?)(?=【题解|$)'
            single_match = re.search(single_pattern, result)
            if single_match:
                content = single_match.group(1).strip()
                content = re.sub(r'\n*-{3,}\s*$', '', content).strip()
                if content:
                    summaries.append(f"【例题分析总结】\n{content}")
        
        if summaries:
            return "\n\n---\n\n".join(summaries)
        
        return ""

    def _generate_amp_summary(self, query: str, result: str, current_problem: str) -> Tuple[str, bool]:
        """使用LLM生成AMP检索摘要并评估有用性"""
        try:
            prompt = f"""请分析这次算法检索的结果，并评估其对解决当前问题的有用性。

【当前问题】
{current_problem if current_problem else "未知 (请基于检索结果质量判断)"}

【检索查询】
{query}

【检索结果】
{result}

请按以下格式输出（不要输出其他废话）：
有用性评估: [有用/无用] (判断标准：检索到的例题或思路是否与当前问题高度相关且提供了具体指导)
摘要: [你的摘要] (格式："我查看了一道[类型]的例题/模式，它的核心思路是[...]，对我做当前题目的作用是[...]")

注意：
- 如果检索结果为空或报错，直接标记为"无用"。
- 摘要控制在100字以内。
"""
            response = tracked_completion(
                model=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个算法竞赛专家。请评估检索结果的价值并进行总结。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                source="amp_retrieval_summary",
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析结果
            is_useful = False
            summary = content
            
            lines = content.split('\n')
            for line in lines:
                if line.startswith("有用性评估:"):
                    eval_str = line.replace("有用性评估:", "").strip()
                    if "有用" in eval_str and "无用" not in eval_str:
                        is_useful = True
                elif line.startswith("摘要:"):
                    summary = line.replace("摘要:", "").strip()
            
            # 如果没找到标准格式，尝试回退
            if "摘要:" not in content and len(lines) > 1:
                # 假设最后一部分是摘要
                pass
                
            return summary, is_useful
            
        except Exception as e:
            print(f"[AttemptManager] ⚠️  AMP摘要生成失败: {e}")
            return "检索完成 (摘要生成失败)", False

    def get_amp_retrieval_summary(self) -> str:
        """生成AMP检索历史摘要"""
        if not self.amp_retrievals:
            return ""
        
        # 只总结除了最近一次之外的检索
        if len(self.amp_retrievals) <= 1:
            return ""
            
        old_retrievals = self.amp_retrievals[:-1]
        
        lines = []
        lines.append("### 📚 早期 AMP 检索记录")
        
        for ret in old_retrievals:
            summary = ret.get('summary', '无摘要')
            lines.append(f"{ret['id']}. {summary}")
            
        return "\n".join(lines)











