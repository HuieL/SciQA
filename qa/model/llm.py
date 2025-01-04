import time
import requests
import json
from tqdm import tqdm
import threading
import os
from dotenv import load_dotenv
import torch
from torch_geometric.data import Data
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class APIModel:

    def __init__(self, model, api_key, api_url) -> None:
        self.__api_key = api_key
        self.__api_url = api_url
        self.model = model

    def __req(self, text, temperature, max_try = 5):
        url = f"{self.__api_url}"
        pay_load_dict = {"model": f"{self.model}","messages": [{
                "role": "user",
                "temperature":temperature,
                "content": f"{text}"}]}
        payload = json.dumps(pay_load_dict)
        headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {self.__api_key}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
        }
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            return json.loads(response.text)['choices'][0]['message']['content']
        except:
            for _ in range(max_try):
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    return json.loads(response.text)['choices'][0]['message']['content']
                except:
                    pass
                time.sleep(0.2)
            return None

    def chat(self, text, temperature=1):
        response = self.__req(text, temperature=temperature, max_try=5)
        return response

    def __chat(self, text, temperature, res_l, idx):

        response = self.__req(text, temperature=temperature)
        res_l[idx] = response
        return response

    def batch_chat(self, text_batch, temperature=0):
        max_threads=15 # limit max concurrent threads using model API
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self.__chat, args=(text, temperature, res_l, i))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads:
                for t in thread_l:
                    if not t .is_alive():
                        thread_l.remove(t)
                time.sleep(0.3) # Short delay to avoid busy-waiting

        for thread in tqdm(thread_l):
            thread.join()
        return res_l


def construct_prompts_and_query(data: Data, model: APIModel):
    prompts = []

    questions = data.questions
    retrieved_contexts = data.retrieved_contexts
    decisions = data.decisions

    count = 0  # 用于计数当前加入 prompts 的数量


    for i, (retrieved_context, question) in enumerate(zip(retrieved_contexts, questions)):
        if all(not element for element in question):
            continue

        # with context + 2 options
        # instruction = "Please answer the following question based on the provided context. Do not explain your answer, only reply with `yes` or `no`."
        # without context + 2 options
        # instruction = "Please answer the following question. Do not explain your answer, only reply with `yes` or `no`."

        # with context + 3 options
        instruction = "Please answer the following question based on the provided context. Do not explain your answer, only reply with `yes`, `no` or 'maybe'."
        # without context + 3 options
        # instruction = "Please answer the following question. Do not explain your answer, only reply with `yes`, `no` or 'maybe'."

        # with context
        prompt = f" {instruction} Context: {retrieved_context}. Question: {question}"
        print(f'prompt: {prompt}\n')
        # without context
        # prompt = f" {instruction} Question: {question}"

        # print(f'这是prompt{idx}，{prompt}\n')
        prompts.append((prompt, decisions[i]))  # 只有成功加入 prompt 后才增加计数
        count += 1

    responses = model.batch_chat(text_batch=[p[0] for p in prompts], temperature=1)
    correct_answers = 0
    total = len(responses)
    for idx, (response, (_, correct_decision)) in enumerate(zip(responses, prompts)):
        response_str = str(response).lower()
        correct_decision_str = str(correct_decision).lower()

        if correct_decision_str in response_str:
            correct_answers += 1

    accuracy = correct_answers / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return responses, accuracy



# Example Usage:
data = torch.load("../dataset/dataset/pubmed_qa/cache/3_labels/pubmed_bm25_encoded_graph.pt")
model = APIModel(model="gpt-4o-2024-05-13", api_key=OPENAI_API_KEY,
                 api_url="https://api.openai.com/v1/chat/completions")
responses, accuracy = construct_prompts_and_query(data, model)
print(f"回答是{responses}")
print(f"准确度是{accuracy}")



# for idx, response in enumerate(responses):
#     print(f"Response {idx + 1}: {response}")


# Example Usage:
# model  = APIModel(model = "gpt-4o-2024-05-13", api_key = OPENAI_API_KEY, api_url = "https://api.openai.com/v1/chat/completions")
# prompts = ["can you introduce youself?", "what data is today?"]
# outputs = model.batch_chat(text_batch=prompts, temperature=1)
# print(outputs)