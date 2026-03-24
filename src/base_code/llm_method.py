from openai import OpenAI
from zai import ZhipuAiClient
import yaml
from typing import Union


class zhipu_client:
    def __init__(self,cfg,ppp=False,plr=False):
        self.model_name=cfg['model_name']
        self.api_key=cfg['api_key']
        self.client=ZhipuAiClient(api_key=self.api_key)
        self.temp=cfg.get('temperature',0.5)
        self.ppp=ppp
        self.plr=plr

    def call_llm(self,system_prompt,user_prompt):
        try:
            if self.ppp:
                print(f"\n\n\nsystem_prompt:{system_prompt}")
                print(f"\nuser_prompt:{user_prompt}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                temperature=self.temp,
            )
            ans = response.choices[0].message.content
            if self.plr:
                print(f"llm reply:{ans}\n\n\n")
            return ans
        except KeyError:
            print("Something wrong with the config, please check!")
            return "error"
        except Exception as e:
            print(f"Something wrong with the api (〒︿〒)! Here is: \n{e}")

class openai_client:
    def __init__(self,cfg,ppp=False,plr=False):
        self.model_name=cfg['model_name']
        self.api_key=cfg['api_key']
        self.base_url=cfg['base_url']
        self.client=OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.temp=cfg.get('temperature',0.5)
        self.ppp=ppp
        self.plr=plr
    
    def call_llm(self,system_prompt,user_prompt):
        try:
            if self.ppp:
                print(f"\n\n\nsystem_prompt:{system_prompt}")
                print(f"\nuser_prompt:{user_prompt}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                temperature=self.temp,
            )
            ans = response.choices[0].message.content
            if self.plr:
                print(f"llm reply:{ans}\n\n\n")
            return ans
        except KeyError:
            print("Something wrong with the config, please check!")
            return "error"
        except Exception as e:
            print(f"Something wrong with the api (〒︿〒)! Here is: \n{e}")    

class embedding_client:
    def __init__(self,cfg):
        self.model_name=cfg['model_name']
        self.api_key=cfg['api_key']
        self.base_url=cfg['base_url']
        self.client=OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def call_llm(self,text):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        ans = response.data[0].embedding
        return ans
    
#Debugging, only print prompts, this type will not call llms' api
class print_pp:
    def __init__(self,cfg):
        pass

    def call_llm(self,text):
        print("Here are prompts:\n")
        print(text)
#if you want to call llm and print prompt, please revise user.yaml-->print_prompt: yes



class llm_method:
    def __init__(self,status):
        with open('config/user.yaml','r',encoding='utf-8') as f:
            self.user_cfg=yaml.safe_load(f)
        with open("config/llm_api.yaml", 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self.llm_cfg=self.cfg['llm'][self.user_cfg[status].get('llm','test')]
        self.ppp=self.user_cfg['print_prompt']
        self.plr=self.user_cfg['print_llm_reply']
        self.ebdllm_cfg=self.cfg['embedding_llm'][self.user_cfg[status].get('embedding_llm','test')]

        self.set_client()

    def set_client(self) -> Union[zhipu_client, openai_client, embedding_client]:
        llm_clients={
            'zhipu':zhipu_client,
            'openai':openai_client,
            'embedding':embedding_client,
            'print': print_pp
        }
        self.llm_clt=llm_clients.get(self.llm_cfg['client_module'])
        self.ebdllm_clt=llm_clients.get(self.ebdllm_cfg['client_module'])
        if not self.llm_clt:
            raise ValueError(f"Unsupported model type!!(/□＼*) Maybe you can set it in llm_method.py")
        if not self.ebdllm_clt:
            raise ValueError(f"Unsupported embedding model type!!(/□＼*) Maybe you can set it in llm_method.py")
        self.llm=self.llm_clt(self.llm_cfg,self.ppp,self.plr)
        self.ebdllm=self.ebdllm_clt(self.ebdllm_cfg)