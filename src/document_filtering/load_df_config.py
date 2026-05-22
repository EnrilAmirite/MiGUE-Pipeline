import json
from src.base_code.load_config import load_config
import threading
import pandas as pd

class load_df_config(load_config):
    def __init__(self):
        super().__init__(status_cfg_path='imdconfig/document_filtering.yaml',status='document_filtering')
        with open (self.get_path('pp','pp'),'r',encoding='utf-8') as f:
            self.pp=json.load(f)
        self.df = pd.read_csv(self.get_path('input','raw'), encoding='utf-8', engine='python', on_bad_lines='warn')
        self.df_lock=threading.Lock()
        self.imd_path=self.get_path('imd','filter')
        self.op_path=self.get_path('output','filtered')

    