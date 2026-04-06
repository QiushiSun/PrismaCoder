import base64
import io
import os.path
import random
import re
import time
from copy import deepcopy
from pathlib import Path

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class BaseOpenAIEngine:
    def __init__(
        self,
        model_name,
        system_prompt: str = "You are helpful assistant",
        add_args: dict = {},
        wait_time: float = 60.0,
        attempts: int = 30,
        api_key_name: str = "OPENAI_API_KEY",
    ) -> None:
        api_key = os.getenv(api_key_name)
        if api_key is None:
            # raise ValueError(f"Please provide {api_key_name} in env variables!")
            print(f"Please provide {api_key_name} in env variables!")
        self.headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
        self.name = model_name
        self.system_prompt = system_prompt
        self.wait_time = wait_time
        self.attempts = attempts
        self.args = add_args
        self.payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}],
            "max_tokens": 3000,
        }
        self.payload.update(add_args)
        self.model_url = None

    @staticmethod
    def add_images(images: list[str | Path], image_detail: str) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def augment_image(image_path_or_enc: str | Path) -> str:
        raise NotImplementedError

    @staticmethod
    def get_content(content: list[dict]) -> list[dict]:
        return content

    def ask(
        self,
        request: str,
        images: list = [],
        image_detail: str = "auto",
    ) -> dict | None:
        content = [{"type": "text", "text": request}]

        if len(images) > 0:
            image_content = self.add_images(images, image_detail)
            content.extend(image_content)

        payload = deepcopy(self.payload)
        payload["messages"].append(
            {"role": "user", "content": self.get_content(content)}
        )

        response = requests.post(self.model_url, headers=self.headers, json=payload)

        try:
            response = response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(e)
            print(f"GPT response is not json-parsable. Response:\n{response}")
            return None

        return response

    def make_request(
        self,
        request: str,
        images: list[str | Path] | None = None,
        image_detail: str = "auto",
    ) -> dict | None:
        if images is None:
            images = []

        error_counts = 0
        response = None
        while error_counts < self.attempts:
            response = self.ask(
                request=request,
                images=images,
                image_detail=image_detail,
            )

            if response is None:
                return None
            if "usage" in response:
                self.print_debug_info(f"Prompt tokens: {response['usage'].get('prompt_tokens')}, "
                    f"Completion tokens: {response['usage'].get('completion_tokens')}, "
                    f"Total: {response['usage'].get('total_tokens')}")
                
            if "error" not in response.keys():
                response["response"] = response["choices"][0]["message"]["content"]
                response["choices"][0]["message"]["content"] = "MOVED to response key"
                break
            else:
                error_counts += 1
                message = response["error"]["message"]
                seconds_to_wait = re.search(r"Please try again in (\d+)s\.", message)
                if seconds_to_wait is not None:
                    wait_time = 1.5 * int(seconds_to_wait.group(1))
                    print(f"Waiting {wait_time} s")
                    time.sleep(wait_time)
                elif (
                    message.startswith(
                        "Your input image may contain content that is not allowed by our safety system."
                    )
                    or "unsupported image" in message
                ):
                    print(message)
                    print("The image would be resized and sent one more time.")
                    images[0] = self.augment_image(images[0])
                    error_counts += 1
                else:
                    print(
                        f"Cannot parse retry time from error message. Will wait for {self.wait_time} seconds"
                    )
                    print(message)
                    response = None
                    time.sleep(self.wait_time)

        if "response" not in response:
            response = None

        return response

    def make_self_debug_request(self, messages: list[list[dict]]) -> dict:
        """Format requests specifically for self-debug mode using an API model.

        Args:
            messages: A list of conversation lists, where each conversation is a list of message dictionaries.

        Returns:
            dict: {"response": list of model responses}
        """
        responses = []
        for conversation in messages:
            formatted_messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            for msg in conversation:
                if msg.get("is_assistant", False):
                    formatted_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
                else:
                    formatted_messages.append({
                        "role": "user",
                        "content": msg["content"]
                    })

            payload = deepcopy(self.payload)
            payload["messages"] = formatted_messages
            
            response = requests.post(self.model_url, headers=self.headers, json=payload)
            try:
                response_json = response.json()
                
                if "usage" in response_json:
                    print(f"[DEBUG] Prompt tokens: {response_json['usage'].get('prompt_tokens')}, "
                          f"[DEBUG] Completion tokens: {response_json['usage'].get('completion_tokens')}, "
                          f"[DEBUG] Total: {response_json['usage'].get('total_tokens')}")
                if "error" not in response_json:
                    responses.append(response_json["choices"][0]["message"]["content"])
                else:
                    responses.append("")
            except:
                responses.append("")

        return {"response": responses}

class BaseOpenAIImageEngine:
    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def encode_images(images: list[str | Path]) -> list[str]:
        # If you pass not Path object or existing file path, it will be read as encoded image
        encoded_images = []
        for image in images:
            if isinstance(image, Path) or os.path.exists(image):
                image_encoded = BaseOpenAIImageEngine.encode_image(image)
            else:
                image_encoded = image

            encoded_images.append(image_encoded)

        return encoded_images

    @staticmethod
    def augment_image(image_path_or_enc: str | Path) -> str:
        if os.path.exists(image_path_or_enc):
            image_enc = BaseOpenAIImageEngine.encode_image(image_path_or_enc)
        else:
            image_enc = image_path_or_enc
        decoded_image = base64.b64decode(image_enc)
        image_bytes = io.BytesIO(decoded_image)
        image = Image.open(image_bytes)

        scale = (image.width * image.height / (1024 * 1024)) ** (0.5)
        if scale > 1:
            new_width = round(image.width / scale)
            new_height = round(image.height / scale)
            image.resize((new_width, new_height))

        new_width = round(image.width * (1 + 0.2 * (random.random() - 0.5)))
        new_height = round(image.height * (1 + 0.2 * random.random() - 0.5))

        new_resolution = (new_width, new_height)
        distorted_image = image.resize(new_resolution)

        output_bytes = io.BytesIO()
        distorted_image.save(output_bytes, format="PNG")
        output_bytes.seek(0)
        image_aug = base64.b64encode(output_bytes.read()).decode("utf-8")

        return image_aug

    def add_images(self, images: list[str], image_detail: str) -> list[dict]:
        encoded_images = self.encode_images(images)
        content = []
        for encoded_image in encoded_images:
            content_image = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}",
                    "detail": image_detail,
                },
            }
            content.append(content_image)

        return content
