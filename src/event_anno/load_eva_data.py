import pandas as pd
import tqdm
import pickle
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

from src.event_anno.load_eva_config import load_eva_config

class load_eva_data(load_eva_config):
    def __init__(self):
        super().__init__()
        with open(self.stc_ebd_p,"rb") as f:
            self.ebd=pickle.load(f)
        with open(self.case_ebd_p,"rb") as f:
            self.case_ebd=pickle.load(f)
        self.sys_pp=self.pp_f['system']
    
    # find similar case
    def stc2case(self,case_type,stc_id):
        case_id_list=[]
        #em_id->em_ebd(alread save..)
        stc_ebd=self.ebd[stc_id]["stc_ebd"]
        case_ebd_list=[
            case for case in self.case_ebd
            if case["case_type"]==case_type
        ]
        #if no case in this type, return
        if len(case_ebd_list)==0:
            return case_ebd_list
        #case ebd list->numpy array 
        case_nump_ebd=np.array([
            case["ebd"] for case in case_ebd_list
        ])
        #stc_ebd->nparray
        em_nump_ebd=np.array(stc_ebd)
        #find top-k
        simi_list=cosine_similarity(
            em_nump_ebd.reshape(1,-1),
            case_nump_ebd
        )[0]
        top_k_case=simi_list.argsort()[-self.top_k:][::-1]
        for i in top_k_case:
            data=case_ebd_list[i]
            data={**data}
            case_id_list.append(data["id"])
        return case_id_list
    

    #load prompt
    def create_case(self,case_type,example_id_list):
        prompt=""
        num=1
        examples=self.case_f[case_type]
        # direct has no des
        describe=f"{examples["description"]}\n{self.pp_f['give']['case']}\n"
        for key,v in examples.items():
            if key not in example_id_list:
                continue
            if not any(v.values()):
                break
            prompt+=f"{self.pp_f['give']['case_n']}{num}:\n"
            prompt+=f"{self.pp_f['give']['case_em']}{v["event_mention"]}\n"
            prompt+=f"{self.pp_f['give']['case_true_tri']}{v["trigger"]}\n"
            if case_type != 'direct':
                prompt+=f"{self.pp_f['give']['case_msl_tri']}{v["wrong_trigger"]}\n"
            prompt+=f"{self.pp_f['give']['case_explain']}{v["explain"]}\n"
            num+=1
        if prompt!="":
            prompt=describe+prompt
        return prompt
    
    def create_first_anno(self,stc,stc_id):
        prompt=""        
        for k,v in self.pp_f['definition'].items():
            prompt+=v
        for k,v in self.pp_f['rules'].items():
            prompt+=v
        example_id_list=self.stc2case("direct",stc_id)
        prompt+=self.create_case("direct",example_id_list)
        # give task
        prompt+=self.pp_f['give']['em']
        prompt+=stc
        prompt+=self.pp_f['output_control']['contain_event']
        prompt+=self.pp_f['output_control']['no_event']
        return prompt
    
    def create_reflection(self,stc,stc_id,raw_triggers,round):
        prompt=""
        for k,v in self.pp_f['definition'].items():
            prompt+=v
        prompt+=f"{self.pp_f['give']['em']}{stc}\n"
        prompt+=f"{self.pp_f['give']['tri']}{raw_triggers}\n"
        prompt+=self.pp_f["reflection"]["intro"]
        match round:
            case 0:
                type_name="missing_check"
                prompt+=self.pp_f["reflection"][type_name]
                rules=self.pp_f["rules"]
                for key,v in rules.items():
                    prompt+=v
                example_id_list=self.stc2case("direct",stc_id)
                prompt+=self.create_case("direct",example_id_list)
            case 1:
                type_name="named_entity"
            case 2:
                type_name="narrative"
            case 3:
                type_name="no_occurrence"
            case 4:
                type_name="assumption"
            case 5:
                type_name="abstraction"
            case 6:
                type_name="negated_event"
        if round!=0:
            prompt+=self.pp_f["reflection"][type_name]
            example_id_list=self.stc2case(type_name,stc_id)
            prompt+=self.create_case(type_name,example_id_list)    
        if round==0:
            prompt+=self.pp_f['output_control']["add"]
        else:
            prompt+=self.pp_f['output_control']["delete"]
        return prompt