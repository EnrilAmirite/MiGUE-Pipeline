import json

from src.base_code.load_config import load_config
from src.base_code.load_data import load_data

class load_eva_config(load_config,load_data):
    def __init__(self):
        super().__init__(status_cfg_path='config/event_anno.yaml', status='event_anno')
        self.ip_p=self.get_path('input','doc')
        self.ev_p=self.get_path('output','event')

        self.eva_p=self.get_path('imd','eva')
        self.stc_p=self.get_path('imd','stc')
        self.docNem_p=self.get_path('imd','docNem')

        self.stc_ebd_p=self.get_path('ebd','stc_ebd')
        self.case_ebd_p=self.get_path('ebd','case_ebd')

        self.pp_p=self.get_path('pp','pp')
        self.case_p=self.get_path('case','case')

        with open (self.case_p,'r',encoding='utf-8') as f:
            self.case_f= json.load(f)
        with open (self.pp_p,'r',encoding='utf-8') as f:
            self.pp_f= json.load(f)


