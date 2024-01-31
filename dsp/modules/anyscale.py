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



class AnyScaleApi(LM):
    def __init__(self, model_name: str, model_type: str = "text", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name)
        self.api_key = api_key
        self.model_name = model_name
        self.model_type = model_type
        self.base_url = "https://api.together.xyz/inference"
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
        """
        client = openai.OpenAI()
        #'meta-llama/Llama-2-13b-chat-hf
        openai.api_key = self.api_key#os.getenv("ANYSCALE_API_KEY")
        openai.api_base = "https://api.endpoints.anyscale.com/v1"       
        response = client.chat.completions.create(
            model = self.kwargs["model"], 
            messages = message_list,
            stream = True,
            temperature = self.kwargs["temperature"],
            max_tokens = self.kwargs["max_tokens"],
            top_p = self.kwargs["top_p"],
        )
        """
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
        """Retrieves completions from Perplexity models.

        Args:
            prompt (str): prompt to send to the Perplexity model
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        time.sleep(1)
        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}
        if "---" not in prompt:
            raise Exception("Invalid prompt")
        system_prompt = "---".join(prompt.split("---")[0:-1])
        user_prompt = prompt.split("---")[-1]
        response = self.basic_request(system_prompt, user_prompt, **kwargs)
        return response
