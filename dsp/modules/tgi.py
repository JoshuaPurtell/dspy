import requests
import json
import time
import functools
from typing import Any, Dict, List, Optional
from dsp.modules.lm import LM
import loguru
logger = loguru.logger
from langsmith import traceable

class TogetherApi(LM):
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
            "max_tokens": 1500,
            "top_p": 0.97,
            "n": 1,
            "stop": ["\n\n\n"],
            **kwargs,
        }
        """
        self.kwargs = {
            "model": self.model_name,
            "max_tokens": 150,
            "temperature": 0.0,
            "top_k": 60,
            "top_p": 0.6,
            "n":1,
            "repetition_penalty": 1.0,
            "stop": ['\n\n'],
            **kwargs,
        }#'<human>', 
        """

    def _send_completion_request_internal(self, prompt_text: str, **kwargs) -> Dict[str, Any]:
        payload = {
            **self.kwargs,
            "prompt": prompt_text,
            **kwargs
        }
        response = requests.post(self.base_url, headers=self.headers, json=payload)
       ##printf"Response : {response.json()}")
        if response.status_code == 429:
           ##print"Rate limit exceeded. Retrying in 5 seconds...")
            time.sleep(5)
            return self._send_completion_request_internal(prompt_text, **kwargs)
        elif response.status_code == 500:
           ##print"Server error. Retrying in 5 seconds...")
            time.sleep(5)
            return self._send_completion_request_internal(prompt_text, **kwargs)
        else:

            try:
                response_json = response.json()
                return response_json['output']
            except json.decoder.JSONDecodeError:
               ##print"Invalid response structure. Retrying in 5 seconds...")
                time.sleep(5)
                return self._send_completion_request_internal(prompt_text, **kwargs)

    @traceable(run_type="chain", name="math problem solver", tags=["dspy"])
    def _send_chat_request_internal(self, message_list: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        ##print"Requesting chat internal...")
        payload = {
            **self.kwargs,
            "messages": message_list,
            **kwargs
        }
        ##print"Payload: ",payload)
        response = requests.post(self.base_url, headers=self.headers, json=payload)
       ##printf"Response : {response.json()}")
        #logger.warning("Response: ",response.json())
        history = {
                "system_prompt": message_list[0]["content"],
                "user_prompt": message_list[1]["content"],
                "response": response.json()["output"]["choices"][0]["text"],
                "raw_response": response.json(),
                "kwargs": kwargs,
            }
        self.history.append(history)
        if response.status_code == 429:
           ##print"Rate limit exceeded. Retrying in 5 seconds...")
            time.sleep(5)
            return self._send_chat_request_internal(message_list, **kwargs)
        elif response.status_code == 500:
           ##print"Server error. Retrying in 5 seconds...")
            time.sleep(5)
            return self._send_chat_request_internal(message_list, **kwargs)
        elif response.status_code == 400:
           ##print"Bad request. Retrying in 5 seconds...")
            time.sleep(5)
            return self._send_chat_request_internal(message_list, **kwargs)
        elif response.status_code == 200:
            return response.json()['output']
        else:
           ##printf"Unknown error. Retrying in 5 seconds... {response.status_code}")
            time.sleep(5)
            return self._send_chat_request_internal(message_list, **kwargs)

    def complete(self, prompt_text: str, **kwargs) -> str:
       ##print"Requesting completion external...")
        json_response = self._send_completion_request_internal(prompt_text, **kwargs)
       ##print"Response: ",json_response)
        history = {
                "system_prompt": "",
                "user_prompt": prompt_text,
                "response": json_response['output']['choices'][0]['text'],
                "kwargs": kwargs,
            }
        self.history.append(history)
        try:
            return json_response['output']['choices'][0]['text']
        except KeyError:
            raise Exception("Invalid response structure")
    
    @traceable(run_type="chain", name="math problem solver - chat", tags=["dspy"])
    def chat(self, message_list: List[Dict[str, str]], **kwargs) -> str:
       ##print"Requesting chat external...")
        json_response = self._send_chat_request_internal(message_list, **kwargs)
       #print("Response intermediate: ",json_response)#['output']
        history = {
                "system_prompt": message_list[0]["content"],
                "user_prompt": message_list[1]["content"],
                "response": json_response['choices'][0]['text'],
                "kwargs": kwargs,
            }
        self.history.append(history)
        try:
            return json_response['choices'][0]['text']#['output']
        except KeyError:
            raise Exception("Invalid response structure")
        
    def basic_request(self, system_prompt: str, user_prompt: str, **kwargs):
       ##print"Basic request...")
        return self.chat([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], **kwargs)
    
    def request(self, prompt, **kwargs) -> str:
       ##print"Requesting completion external...")
        return self._send_completion_request_internal(prompt, **kwargs)
    
    def _get_choice_text(self, choice: dict[str, Any]) -> str:
       ##print"Getting choice text...")
        if self.model_type == "chat":
            return choice["message"]["content"]
        elif self.model_type == "text":
            return choice["text"]
        
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
        #response = self.request(prompt, **kwargs)
        response = self.basic_request(system_prompt, user_prompt, **kwargs)
       #print("Response 1: ",response)
        return [response]
        if "choices" not in response:
            raise Exception("Invalid response structure")
        choices = response["choices"]
        #print"Response 2: ",response)
        #print"Returning this: ",[choices[i]["text"] for i in range(len(choices))])
        return [choices[i]["text"] for i in range(len(choices))]

        completions = [self._get_choice_text(c) for c in choices]

        return completions


    
    def __completion_call__(
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

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}
       ##print"Requesting completion external...")
       ##print"Prompt: ",prompt)
        response = self.request(prompt, **kwargs)
        if "choices" not in response:
            raise Exception("Invalid response structure")
        choices = response["choices"]

        completions = [self._get_choice_text(c) for c in choices]

        return completions

    def request_completion(self, prompt_text: str, **kwargs) -> Dict[str, Any]:
        return self._send_completion_request_internal(prompt_text, **kwargs)

    def request_chat(self, message_list: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
       ##print"Requesting chat external...")
        return self._send_chat_request_internal(message_list, **kwargs)