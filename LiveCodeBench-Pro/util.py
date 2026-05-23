import re

def _fix_broken_char_literals(code: str) -> str:
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
    
    if code != original_code:
        print("Warning: Detected broken character literals, auto-fixed")
    
    return code


def extract_longest_cpp_code(text):
    fenced_pattern = r"(?m)^```cpp\s*\n(.*?)\n```"
    fenced_blocks = re.findall(fenced_pattern, text, flags=re.DOTALL)
    if fenced_blocks:
        for block in reversed(fenced_blocks):
            if "#include" in block:
                return _fix_broken_char_literals(block.strip())

    cleaned_text = text

    main_matches = list(re.finditer(r"int\s+main\s*\(", cleaned_text))
    if main_matches:
        for main in reversed(main_matches):
            main_start_pos = main.start()
            main_end_pos = main.end()

            brace_start = cleaned_text.find("{", main_end_pos)
            if brace_start == -1:
                continue

            brace_count = 0
            idx = brace_start
            text_len = len(cleaned_text)
            while idx < text_len:
                ch = cleaned_text[idx]
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        idx += 1
                        break
                idx += 1
            func_end = idx

            lines = cleaned_text.splitlines()
            line_start_indices = []
            curr_idx = 0
            for line in lines:
                line_start_indices.append(curr_idx)
                curr_idx += len(line) + 1

            main_line_index = None
            for i, start in enumerate(line_start_indices):
                if start <= main_start_pos < (start + len(lines[i]) + 1):
                    main_line_index = i
                    break
            if main_line_index is None:
                main_line_index = 0

            include_line_index = None
            for i in range(main_line_index, -1, -1):
                if re.match(r"^\s*#include", lines[i]):
                    include_line_index = i
                else:
                    if include_line_index is not None:
                        break

            candidate_start = (
                line_start_indices[include_line_index]
                if include_line_index is not None
                else line_start_indices[main_line_index]
            )

            candidate_code = cleaned_text[candidate_start:func_end].strip()
            if "#include" in candidate_code:
                return _fix_broken_char_literals(candidate_code)

    return None
