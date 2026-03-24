from src.base_code.load_config import load_config
from src.base_code.load_data import load_data





class load_dc_config(load_config,load_data):
    def __init__(self):
        load_config.__init__(self,status_cfg_path='config/document_clustering.yaml', status='document_clustering')
        load_data.__init__(self)

        self.cdcore_p=self.get_path('input','cdcore')
        self.doc_p=self.get_path('input','doc')
        
        self.coreline_p=self.get_path('imd','coreline')

        self.dc_p=self.get_path('output','dc')

