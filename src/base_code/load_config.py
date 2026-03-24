import yaml
from pathlib import Path
import os
from src.base_code.llm_method import llm_method
import json
import pandas
import csv

class load_config(llm_method):
    def __init__(self, status_cfg_path, status):
        llm_method.__init__(self,status)
        with open(status_cfg_path,'r',encoding='utf-8') as f:
            self.stts_cfg=yaml.safe_load(f)
        self.language=self.user_cfg.get('language','zh')
        self.top_k=self.stts_cfg.get('top_k',1)
        self.max_workers=self.user_cfg.get('max_workers',6)
        self.thsh=self.stts_cfg.get('threshold',0.8)

        self.minedge=self.user_cfg[status].get('min_core_num',2)
        self.mindoc=self.user_cfg[status].get('min_cluster_doc',2)
        
    def get_path(self,category,step='ph'):#placeholder
        match category:

            # in prompt (need step and language)
            case 'pp':
                folder=self.stts_cfg['prompt'][self.language][step]['folder']
                name=self.stts_cfg['prompt'][self.language][step]['name']
            
            # in case (need step and language)
            case 'case':
                folder=self.stts_cfg[category][self.language][step]['folder']
                name=self.stts_cfg[category][self.language][step]['name']

            # in rule (need step)
            case 'rule':
                folder=self.stts_cfg[category][step]['folder']
                name=self.stts_cfg[category][step]['name']                    

            #in storage (need step)
            case "input"|"output"|"ebd"|"rule"|'imd':
                folder=self.stts_cfg['storage'][category][step]['folder']
                name=self.stts_cfg['storage'][category][step]['name']    

        path_str=os.path.join(folder,name)
        path=Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    def create_file(self,file_path,default_content=None):
        path = Path(file_path)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == '.json':
            content = default_content if default_content is not None else {}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4) 
        elif suffix == '.csv':
            with open(path, 'w', newline='', encoding='utf-8') as f:
                if default_content and isinstance(default_content, list):
                    writer = csv.writer(f)
                    writer.writerow(default_content)
                else:
                    path.touch()
        elif suffix in ['.txt', '.log','.jsonl']:
            path.touch()
        
