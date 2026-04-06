from .openai_engine import OpenAIEngine
from .together_engine import TogetherEngine


def get_model_by_name(
    model_name: str,
    model_pars: dict = {},
    system_prompt: str | None = None,
    **kwargs,
):
    kwargs.update(
        {
            "add_args": model_pars,
            "model_name": model_name,
        }
    )
    if system_prompt is not None:
        kwargs.update({"system_prompt": system_prompt})

    if model_name.startswith("openai/"):
        kwargs.update({"model_name": model_name[len("openai/") :]})
        model = OpenAIEngine(**kwargs)
    elif model_name.startswith("together/"):
        kwargs.update({"model_name": model_name[len("together/") :]})
        model = TogetherEngine(**kwargs)
    else:
        # That import is here temporary to prevent import of cuda-libraries if they are not needed.
        from .vllm_engine import VllmEngine

        model = VllmEngine(**kwargs)

    return model
