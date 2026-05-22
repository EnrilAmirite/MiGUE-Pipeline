import json
import pandas as pd
from src.base_code.load_config import load_config
from src.base_code.load_data import load_data

class load_erai_config(load_config,load_data):
    def __init__(self):
        load_config.__init__(self,status_cfg_path='imdconfig/event_relation_anno_intra.yaml', status='event_relation_anno_intra')
        load_data.__init__(self)
        self.e_p=self.get_path('input','event')
        self.doc_p=self.get_path('input','doc')

        self.ep_p=self.get_path('imd','ep')
        self.era_p=self.get_path('imd','era')
        self.er_p=self.get_path('output','eri')


        self.opt_p=self.get_path('pp','opt')
        self.des_p=self.get_path('pp','des')
        self.pp_p=self.get_path('pp','pp')
        
        self.rre_p=self.get_path('rule','rel_rvs')
        self.rule_p=self.get_path('rule','rule')
        with open(self.rule_p,'r',encoding='utf-8') as f:
            self.rule_f=json.load(f)

