"""Common utility functions shared across language handlers"""
import re
import base64
from io import BytesIO
from PIL import Image
from typing import Optional


def remove_ansi_escape(text: str) -> str:
    """
    Remove ANSI escape sequences from the given text.

    Args:
        text (str): Input text containing ANSI escape sequences.

    Returns:
        str: Text with ANSI escape sequences removed.
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def remove_bytes_literal(text: str, min_length: int = 100) -> str:
    """
    Remove long byte literals from error messages.

    Args:
        text (str): Input text containing byte literals.
        min_length (int): Minimum length of byte literals to remove.

    Returns:
        str: Text with long byte literals replaced by "b'<bytes>'".
    """
    pattern = r"b'[^']{" + str(min_length) + r",}'"
    return re.sub(pattern, "b'<bytes>'", text)


def shrink_png_b64_to_under_kb(
    b64_png: str, 
    max_kb: int = 200, 
    step: float = 0.9, 
    min_side: int = 64
) -> str:
    """
    Shrink a PNG base64 string to under a specified size in KB by resizing.

    Args:
        b64_png (str): Base64 encoded PNG (without data URI prefix).
        max_kb (int): Maximum size in KB.
        step (float): Scale factor for each iteration of resizing.
        min_side (int): Minimum dimension to preserve during resizing.

    Returns:
        str: Resized PNG as a base64 string.
    """
    try:
        raw = base64.b64decode(b64_png)
        max_bytes = max_kb * 1024
        
        if len(raw) <= max_bytes:
            return b64_png

        with Image.open(BytesIO(raw)) as img:
            img = img.convert("RGBA") if img.mode in ("LA", "RGBA", "P") else img.convert("RGB")
            
            while True:
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                current_size = len(buf.getvalue())
                
                if current_size <= max_bytes:
                    return base64.b64encode(buf.getvalue()).decode("utf-8")
                
                w, h = img.size
                if min(w, h) <= min_side:
                    return base64.b64encode(buf.getvalue()).decode("utf-8")
                
                new_w = int(w * step)
                new_h = int(h * step)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except Exception:
        return b64_png


def deduplicate_lines(
    text: str, 
    max_repeat: int = 5, 
    min_length: int = 10
) -> str:
    """
    Remove excessive line repetitions from the given text.

    Args:
        text (str): Input text to process.
        max_repeat (int): Maximum allowed repetitions of a line.
        min_length (int): Minimum length of a line to check for duplication.

    Returns:
        str: Text with excessive line repetitions removed.
    """
    from collections import defaultdict
    
    lines = text.splitlines()
    seen_count = defaultdict(int)
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Keep empty lines
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Keep short lines without checking
        if len(stripped) < min_length:
            cleaned_lines.append(line)
            continue

        # Count and filter long lines
        seen_count[stripped] += 1
        if seen_count[stripped] <= max_repeat:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def strip_thinking(text: str) -> str:
    THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
    out = THINK_BLOCK_RE.sub("", text)

    close_idx = out.lower().find("</think>")
    if close_idx != -1:
        before_close = out[:close_idx].lower()
        if "<think" not in before_close:
            out = out[close_idx + len("</think>") :]
    return out.lstrip("\n\r\t ")


_FENCE_LANG_TAGS = frozenset({
    "python", "py", "python3", "html", "javascript", "js", "typescript", "ts",
    "json", "xml", "css", "bash", "sh", "shell", "zsh", "sql", "yaml", "yml",
    "ruby", "rb", "java", "c", "cpp", "csharp", "cs", "go", "rust", "r",
    "swift", "kotlin", "scala", "perl", "php", "lua", "text", "plain",
    "plaintext", "markdown", "md", "csv", "toml", "ini", "dockerfile",
    "makefile", "cmake", "matlab", "octave", "jsx", "tsx",
})


def _is_fence_lang_tag(s: str) -> bool:
    """Return True if s looks like a markdown code-fence language tag."""
    return s.lower() in _FENCE_LANG_TAGS


def extract_fenced_code(
    response: str, 
    language_tags: list[str],
    remove_stop_tokens: bool = True
) -> str:
    """
    Extract code from fenced code blocks with fallback logic.

    Args:
        response (str): Model response text containing code blocks.
        language_tags (list[str]): Ordered list of language tags to try (e.g., ['python', 'json']).
        remove_stop_tokens (bool): Whether to remove common stop tokens from the response.

    Returns:
        str: Extracted code block.
    """
    code = ""

    response = strip_thinking(response)
    
    # Try language-specific tags first
    for tag in language_tags:
        marker = f"```{tag}"
        if marker in response:
            first_block = response.split(marker, 1)[1]
            code = first_block.split("```", 1)[0]
            return code.strip()
    
    # Fallback to any fenced block -- strip the language tag on the first line
    if "```" in response:
        first_block = response.split("```", 1)[1]
        code = first_block.split("```", 1)[0]
        lines = code.split("\n", 1)
        if len(lines) > 1 and _is_fence_lang_tag(lines[0].strip()):
            code = lines[1]
        return code.strip()
    
    # No fenced block found
    if remove_stop_tokens:
        stop_tokens = [
            "<|eot_id|>", "<|endoftext|>", "<EOS>",
            "<|start_header_id|>", "<|end_header_id|>"
        ]
        for token in stop_tokens:
            response = response.replace(token, "")
    
    return response.strip()


def remove_absolute_paths(text: str) -> str:
    """
    Remove absolute file paths from error messages.

    Args:
        text (str): Input text containing file paths.

    Returns:
        str: Text with absolute file paths replaced by 'file.py'.
    """
    return re.sub(r'/[^\s:]+\.py', 'file.py', text)