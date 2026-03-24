from load_df_config import load_df_config
from src.base_code.load_data import load_data
import json

class load_df_data(load_df_config,load_data):
    def __init__(self):
        load_df_config.__init__(self)
        load_data.__init__(self)

    def load_filter_prompt(self,type,raw_text):
        prompt_type=self.pp[type]
        prompt=prompt_type['intro']+prompt_type['rules']+self.pp['give_doc']
        prompt+=f"{raw_text}\n"
        prompt+=self.pp['output_control']
        sys_prom=self.pp['system']
        return sys_prom,prompt