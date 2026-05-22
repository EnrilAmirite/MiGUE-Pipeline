from src.base_code.load_config import load_config
from src.base_code.load_data import load_data

class load_erac_config(load_config,load_data):
    def __init__(self):
        load_config.__init__(self,status_cfg_path='imdconfig/event_relation_anno_cross.yaml', status='event_relation_anno_cross')
        load_data.__init__(self)
        self.ep_p=self.get_path('imd','ep')
        self.create_file(self.ep_p)
        self.era_p=self.get_path('imd','era')
        self.core_p=self.get_path('imd','core')
        self.cotrans_p=self.get_path('imd','cotrans')
        self.reltrans_p=self.get_path('imd','reltrans')

        self.eri_p=self.get_path('input','eri')
        self.doc_p=self.get_path('input','doc')
        self.e_p=self.get_path('input','event')
        self.erc_p=self.get_path('output','erc')

        self.em_ebd_p=self.get_path('ebd','em_ebd')
        self.emid2idx_p=self.get_path('ebd','emid2idx')
        self.emid_p=self.get_path('ebd','emid')
        self.idx2ebd_p=self.get_path('ebd','idx2ebd')

        self.opt_p=self.get_path('pp','opt')
        self.des_p=self.get_path('pp','des')
        self.pp_p=self.get_path('pp','pp')
        
        self.rre_p=self.get_path('rule','rel_rvs')
        self.rule_p=self.get_path('rule','rule')
        self.reltrule_p=self.get_path('rule','rel_trans')
