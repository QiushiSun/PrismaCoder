import warnings

import tiktoken
from PIL import Image

from .base_engine import BaseOpenAIEngine, BaseOpenAIImageEngine

Image.MAX_IMAGE_PIXELS = None


class OpenAIEngine(BaseOpenAIImageEngine, BaseOpenAIEngine):
    def __init__(
        self,
        model_name,
        system_prompt: str = "You are helpful assistant",
        do_logprobs: bool = False,
        tokens_highlighted: list[str] = [],
        add_args: dict = {},
        wait_time: float = 60.0,
        attempts: int = 30,
    ) -> None:
        api_key_name = "OPENAI_API_KEY"
        super().__init__(
            model_name, system_prompt, add_args, wait_time, attempts, api_key_name
        )
        self.name = "openai/" + model_name
        self.model_url = "https://az.gptplus5.com/v1/chat/completions"
        if do_logprobs:
            self.construct_logit_args(tokens_highlighted)
        self.tokens_highlighted = tokens_highlighted

    def construct_logit_args(
        self, tokens_highlighted: list[str] = [], logit_bias_value: float = 30.0
    ) -> None:
        tokenizer = tiktoken.encoding_for_model(self.name)

        options_tok_ids = dict()
        logit_bias = dict()

        for opt in tokens_highlighted:
            tok_ids = tokenizer.encode(opt)
            if len(tok_ids) == 1:
                warnings.warn(
                    "Output shpuld be 1 token length. We'll take only the first token"
                )
            logit_bias[tok_ids[0]] = logit_bias_value
            options_tok_ids[opt] = tok_ids

        self.args.update({"logit_bias": logit_bias})
