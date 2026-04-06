import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass, fields
from pathlib import Path
from typing import Any

from vllm import LLM, RequestOutput, SamplingParams

def to_dict(obj):
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except TypeError:
            return {f.name: getattr(obj, f.name) for f in fields(obj)}

    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()

    if hasattr(obj, "__dict__"):
        return vars(obj)

    if hasattr(obj, "__slots__"):
        return {slot: getattr(obj, slot) for slot in getattr(obj, "__slots__", [])}

    return {"repr": repr(obj)}

def check_files_exist(folder_path: Path | str, filenames: list[str]) -> bool:
    folder_path = Path(folder_path)
    existing_files = [(folder_path / filename).exists() for filename in filenames]
    return all(existing_files)


def get_model_name_and_path(
    model_name_or_path: str | Path, model_name: str | None = None
) -> tuple[str, str | None]:
    tokenizer_files = [
        "tokenizer_config.json",
        # "special_tokens_map.json",
        "tokenizer.json",
    ]
    tok_files_exist = check_files_exist(model_name_or_path, tokenizer_files)
    tok_name_or_path = None
    if not Path(model_name_or_path).exists():
        model_name = str(model_name_or_path)
    else:
        config_path = Path(model_name_or_path) / "config.json"
        if config_path.exists:
            with open(config_path, "r") as file:
                config_data = json.load(file)
            if config_data.get("_name_or_path") is not None:
                model_name = config_data.get("_name_or_path")
        if (not tok_files_exist) and (model_name is None):
            raise AttributeError(
                "You have no tokenizer files in model folder.\n"
                "Please provide model name for tokenizer either in config.json file in the model folder\n"
                "or as model.model_name parameter"
            )
        elif tok_files_exist and model_name is None:
            model_name = str(model_name_or_path)
        else:
            tok_name_or_path = model_name

    return model_name, tok_name_or_path


class VllmEngine:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        vllm_args: dict = {},
        generation_args: dict = {},
    ):
        self.name, tokenizer_name = get_model_name_and_path(
            model_name_or_path=model_name
        )
        if tokenizer_name is not None:
            vllm_args["tokenizer"] = tokenizer_name
        vllm_args.update({"max_model_len": 32768, "tensor_parallel_size": 8})
        if "temperature" in add_args:
            generation_args.update({"temperature": add_args["temperature"]})
        else:
            generation_args.update({"temperature": 0.0})
        generation_args.update({"max_tokens": 16000, "ignore_eos": False})
        self.llm = LLM(model=model_name, **vllm_args)
        self.sampling_params = SamplingParams(**generation_args)
        self.system_prompt = system_prompt

    def _truncate_if_needed(self, text: str) -> str:
        tokenizer = self.llm.get_tokenizer()
        tokens = tokenizer.encode(text)
        max_len = self.llm.llm_engine.model_config.max_model_len
        if len(tokens) > max_len:
            print(f"[WARNING] prompt too long ({len(tokens)} > {max_len}), truncating...")
            max_tokens = int(max_len * 0.9)  # Use 90% of maximum allowed tokens
            tokens = tokens[-max_tokens:]
            return tokenizer.decode(tokens)
        return text

    def generate(
        self,
        input_texts: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        
        truncated_inputs = [self._truncate_if_needed(t) for t in input_texts]
        responses = self.llm.generate(
            prompts=truncated_inputs,
            sampling_params=self.sampling_params,
            # use_tqdm=False
        )
        outputs = [self.get_outputs(response) for response in responses]

        return self.batch_output(outputs)

    def format_input(self, message: str) -> str:
        # if "meta-llama" in self.name:
        system_mes = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
        user_mes = f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
        assist_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        model_input = system_mes + user_mes + assist_prompt
        # else:
        #     model_input = (self.system_prompt + "\n" + message).strip()

        return model_input

    def make_request(
        self,
        request: str | list[str],
    ) -> dict | None:
        if isinstance(request, str):
            requests = [request]
        else:
            requests = request
        requests = [self.format_input(request) for request in requests]
        response = self.generate(input_texts=requests)

        return {"response": response["text"]}

    @staticmethod
    def get_outputs(response: RequestOutput) -> dict[str, Any]:
        metainfo = to_dict(response.outputs[0])
        metainfo.pop("text", None)
        metainfo.pop("token_ids", None)
        metainfo["time_metrics"] = to_dict(response.metrics)
        output_dict = {
            "text": response.outputs[0].text,
            "tokens": list(response.outputs[0].token_ids),
            "metainfo": metainfo,
        }
        return output_dict

    @staticmethod
    def batch_output(outputs: list[dict[str, Any]]) -> dict[str, list[Any]]:
        batched_output = defaultdict(list)
        for d in outputs:
            for key, value in d.items():
                batched_output[key].append(value)
        return dict(batched_output)

    def format_self_debug_conversation(self, messages: list[dict]) -> str:
        """
        Format a multi-turn conversation for self-debug mode.

        Args:
            messages: List of message dicts, each containing a 'content' field and optional 'is_assistant' flag.

        Returns:
            str: Formatted conversation string for the model input.
        """
        system_mes = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>"
        conversation = ""
        for msg in messages:
            if msg.get("is_assistant", False):
                conversation += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            else:
                conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        conversation += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return system_mes + conversation
    
    def make_self_debug_request(self, messages: list[list[dict]]) -> dict:
        """
        Handle batch debug mode requests.

        Args:
            messages: List of conversations, each conversation is a list of message dicts.

        Returns:
            dict: {"response": list of model responses}
        """
        model_inputs = [
            self.format_self_debug_conversation(conversation) 
            for conversation in messages
        ]
        
        # Generate responses in batch
        response = self.generate(input_texts=model_inputs)
        
        return {"response": response["text"]}