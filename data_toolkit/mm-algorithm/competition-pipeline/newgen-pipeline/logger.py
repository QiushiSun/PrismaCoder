import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading


class APILogger:
    def __init__(self, log_dir: str = "api_logs"):
        """
        初始化日志记录器

        Args:
            log_dir: 日志存储目录，默认为当前目录下的 api_logs
        """
        self.log_dir = Path(log_dir)
        # self._lock = threading.Lock()  # 线程安全

    def log_call(
        self,
        model: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        user: str = "default",
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ):
        """
        记录一次 API 调用
        Args:
            model: 模型名称，如 "claude-3-opus"
            request_data: 请求数据（dict）
            response_data: 响应数据（dict）
            user: 用户标识，默认 "default"
            duration_ms: 调用耗时（毫秒）
            success: 是否成功
            error: 错误信息（如果有）
            metadata: 额外的元数据
        """
        timestamp = datetime.now()

        # 计算 token 和字符数
        prompt_chars = self._calculate_prompt_chars(request_data)
        completion_chars = self._calculate_completion_chars(response_data)

        # 从响应中提取 token 信息（仅使用服务商返回的精确值，不估算）
        token_usage = self._extract_token_usage(response_data)
        prompt_tokens = token_usage["prompt_tokens"]
        completion_tokens = token_usage["completion_tokens"]
        total_tokens = token_usage["total_tokens"]

        # 构建日志条目
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "model": model,
            "user": user,
            "api_key": api_key,
            "request": request_data,
            "response": response_data,
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success,
            "error": error
        }

        # 添加元数据
        if metadata:
            log_entry["metadata"] = metadata

        # 写入完整日志
        self._write_full_log(timestamp, model, log_entry)

        # 写入统计日志
        self._write_stats_log(timestamp, model, user, {
            "timestamp": timestamp.isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_ms": duration_ms,
            "success": success
        })

    def _write_full_log(self, timestamp: datetime, model: str, log_entry: Dict[str, Any]):
        """写入完整日志到文件"""
        # 构建日志路径: calls/{model}/{YYYY-MM}/{YYYY-MM-DD}.jsonl
        year_month = timestamp.strftime("%Y-%m")
        date = timestamp.strftime("%Y-%m-%d")

        log_path = self.log_dir / "calls" / model / year_month
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"{date}.jsonl"

        # 线程安全写入
        # with self._lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _write_stats_log(
        self,
        timestamp: datetime,
        model: str,
        user: str,
        stats_entry: Dict[str, Any]
    ):
        """写入统计日志"""
        # 构建统计日志路径: stats/{model}/{user}_{YYYY-MM}.jsonl
        year_month = timestamp.strftime("%Y-%m")

        stats_path = self.log_dir / "stats" / model
        stats_path.mkdir(parents=True, exist_ok=True)

        stats_file = stats_path / f"{user}_{year_month}.jsonl"

        # 线程安全写入
        # with self._lock:
        with open(stats_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats_entry, ensure_ascii=False) + "\n")

    def _calculate_prompt_chars(self, request_data: Dict[str, Any]) -> int:
        """计算请求中的提示字符数"""
        total_chars = 0

        # 支持 OpenAI 格式
        if "messages" in request_data:
            for message in request_data["messages"]:
                if "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        # 支持多模态内容
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                total_chars += len(item["text"])
        # 支持 prompt 字段
        elif "prompt" in request_data:
            total_chars = len(str(request_data["prompt"]))

        return total_chars

    def _calculate_completion_chars(self, response_data) -> int:
        """计算响应中的完成字符数"""
        total_chars = 0

        # OpenAI
        if "choices" in response_data:
            for choice in response_data["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    total_chars += len(str(choice["message"]["content"]))
                elif "text" in choice:
                    total_chars += len(str(choice["text"]))
        elif "content" in response_data:
            # Anthropic
            if isinstance(response_data['content'], list):
                content_list = response_data['content']
                for item in content_list:
                    if 'thinking' in item:
                        total_chars += len(item['thinking'])
                    elif 'text' in item:
                        total_chars += len(item['text'])
            else:
                total_chars = len(str(response_data["content"]))

        return total_chars

    def _extract_token_usage(self, response_data: Dict[str, Any]) -> Dict[str, int]:
        """
        从响应中提取 token 使用信息

        优先使用服务商返回的精确 token 数，如果没有则返回 0（不估算）
        """
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        if "usage" in response_data and response_data["usage"] is not None:
            usage = response_data["usage"]
            # OpenAI
            if 'prompt_tokens' in usage:
                token_usage["prompt_tokens"] = usage.get("prompt_tokens", 0)
                token_usage["completion_tokens"] = usage.get("completion_tokens", 0)
                token_usage["total_tokens"] = usage.get("total_tokens", 0)
            # Anthropic
            elif 'input_tokens' in usage:
                token_usage["prompt_tokens"] = usage.get("input_tokens", 0)
                token_usage["completion_tokens"] = usage.get("output_tokens", 0)
                token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]

        return token_usage


# 全局默认实例
# _default_logger = APILogger()

# def log_call(*args, **kwargs):
#     """使用默认日志记录器记录调用"""
#     return _default_logger.log_call(*args, **kwargs)


# def set_log_dir(log_dir: str):
#     """设置全局日志目录"""
#     global _default_logger
#     _default_logger = APILogger(log_dir)

# log_completion = log_call