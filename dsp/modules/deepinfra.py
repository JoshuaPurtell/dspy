import requests
import json
import time
import functools
from typing import Any, Dict, List, Optional
from dsp.modules.lm import LM
import loguru
logger = loguru.logger
from langsmith import traceable
import openai
import os
import requests



class DeepInfraApi(LM):
    def __init__(self, model_name: str, model_type: str = "text", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name)
        self.api_key = api_key
        self.model_name = model_name
        self.model_type = model_type
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.kwargs = {
            "model": self.model_name,
            "temperature": 0.01,
            "max_tokens": 1000,
            "top_p": 0.97,
            "n": 1,
            "stop": ["\n", "\n\n"],
            **kwargs,
        }
    def hit_api(self, messages, **kwargs):
        url = f"https://api.endpoints.anyscale.com/v1/chat/completions"
        body = {
        "model": self.model_name,
        "messages": messages,
        "temperature": self.kwargs["temperature"],
        "max_tokens": self.kwargs["max_tokens"],
        "top_p": self.kwargs["top_p"],
        }
        s = requests.Session()
        authorization_headers = {"Authorization": f"Bearer {self.api_key}"}
        with s.post(url, headers=authorization_headers, json=body) as response:
            api_response = response#.json()
        print("response", api_response.text, api_response.status_code)
        print("response", api_response.json())
        return api_response.json()
    
    def chat(self, message_list: List[Dict[str, str]], **kwargs) -> str:
        response = self.hit_api(message_list, **kwargs)
        print("response", response)
        return response["output"]["choices"][0]["text"]

    def basic_request(self, system_prompt: str, user_prompt: str, **kwargs):
        return self.chat([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        time.sleep(1)
        if "---" not in prompt:
            raise Exception("Invalid prompt")
        system_prompt = "---".join(prompt.split("---")[0:-1])
        user_prompt = prompt.split("---")[-1]
        response = self.basic_request(system_prompt, user_prompt, **kwargs)
        return response
