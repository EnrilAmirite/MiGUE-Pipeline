from openai import OpenAI
import os
from datetime import datetime
from tqdm import tqdm
import csv
from pydantic import BaseModel
from itertools import combinations
import pandas as pd
from collections import defaultdict
import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from src.event_relation_anno.load_erac_config import load_erac_config



class erc_trans(load_erac_config):
    def __init__(self):
        super().__init__()
        with open(self.reltrule_p,'r',encoding='utf-8') as f:
            self.reltrule_f=json.load(f)


    #filter cross-doc event relation annotation's co-reference
    def filtering_core(self):
        print('\nFiltering cross document co-reference event pairs.....')
        with open(self.era_p,'r',encoding='utf-8') as fin:
            with open(self.core_p,'w',encoding='utf-8') as fout:
                for line in fin:
                    line=line.strip()#防止有空格或者空行
                    if not line:
                        continue
                    cd_ere = json.loads(line)
                    if cd_ere["RELATION"]["COREFERENCE"] != "COREFERENCE":
                        continue
                    else: 
                        fout.write(json.dumps(cd_ere,ensure_ascii=False)+"\n")
        print('\nFinish filtering cross document co-reference event pairs ^w^!')

    def create_erc_uni_index(self,path_jsonl,key):
        index = defaultdict(list)
        with open(path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                data=json.loads(line)
                index[data["bg"][key]].append(data)
                #双向建立的话可以再append
                #index[data["bg"][key]].append(data)
        return index

    def create_erc_bil_index(self,path_jsonl,key_1,key_2):
        index = defaultdict(list)
        with open(path_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                index[data["bg"][key_1]].append(data)
                #双向建立的话可以再append
                index[data["bg"][key_2]].append(data)
        return index

    # use cross-doc co-reference to transmit relation
    def create_cotrans(self):
        print('\nUse cross-doc co-reference to transmit relation.....')
        index=self.create_erc_uni_index(self.core_p,'e_id_a')
        #print(f"DEBUG: index size is {len(index)}")              
        with open(self.eri_p,'r',encoding='utf-8') as fin:
            with open(self.cotrans_p,'w',encoding='utf-8') as fout:
                for line in fin:
                    line=line.strip()
                    if not line:
                        continue
                    sd_ere=json.loads(line)
                    event_id_a=sd_ere["bg"]["e_id_a"]
                    event_id_b=sd_ere["bg"]["e_id_b"]
                    core_list_a=index[event_id_a]
                    core_list_b=index[event_id_b]
                    #A'=<---(A--->B)
                    #print(f"DEBUG:现在正在查找{event_id_a},list为{core_list_a}")
                    if len(core_list_a)!=0:
                        for row in core_list_a:
                            save_data={
                                "em_a":row['em_b'],
                                "tri_a":row["tri_b"],
                                "offset_a": row["offset_b"],
                                "em_b": sd_ere["em_b"],
                                "tri_b": sd_ere["tri_b"],
                                "offset_b": sd_ere["offset_b"],
                                "RELATION": sd_ere["RELATION"],
                                "bg": {
                                    "e_id_a":row['bg']['e_id_b'],
                                    "e_id_b":sd_ere['bg']['e_id_b'],
                                    "imd_e_id":sd_ere['bg']['e_id_a'],# intermediate event id
                                    "transmit":'head'#The head event is the one that was propagated. 
                                }
                            }
                            fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")
                    #(A--->B)--->=B'
                    #print(f"DEBUG:现在正在查找{event_id_b},list为{core_list_b}")
                    if len(core_list_b)!=0:
                        for row in core_list_b:
                            save_data={
                                "em_a":sd_ere['em_a'],
                                "tri_a":sd_ere["tri_a"],
                                "offset_a": sd_ere["offset_a"],
                                "em_b": row["em_b"],
                                "tri_b": row["tri_b"],
                                "offset_b": row["offset_b"],
                                "RELATION": sd_ere["RELATION"],
                                "bg": {
                                    "e_id_a":sd_ere['bg']['e_id_a'],
                                    "e_id_b":row['bg']['e_id_b'],
                                    "imd_e_id":sd_ere['bg']['e_id_b'],
                                    "transmit":'tail'
                                }
                            }
                            fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")


    # use relation's transmit rules to transmit relation
    def create_ruletrans(self):
        #所有关系都被补全成双向边了
        #for each A--->B,consider ?--->A  and  B --->?
        #只从跨文档的core Event那端jump
        index_out=self.create_erc_uni_index(self.eri_p,'e_id_a')
        index_in=self.create_erc_uni_index(self.eri_p,'e_id_b')
        with open(self.cotrans_p,'r',encoding='utf-8') as fin:
            with open(self.reltrans_p,'w',encoding='utf-8') as fout:
                for line in fin:
                    line=line.strip()
                    if not line:
                        continue
                    cd_ere=json.loads(line)
                    transmit=cd_ere["bg"].get("transmit")
                    #?--->A'--->B
                    # re_1   re_2
                    if transmit=='head':
                        sd_ere_list=index_in[cd_ere["bg"]["e_id_a"]] 
                        if len(sd_ere_list)!=0 :
                            for row in sd_ere_list:
                                change=0
                                re_save={
                                    "TEMPORAL":"NO_TEMPORAL",
                                    "CAUSAL":"NO_CAUSAL",
                                    "SUBEVENT":"NO_SUBEVENT",
                                    "COREFERENCE":"NO_COREFERENCE"
                                }
                                re_1=row["RELATION"]
                                re_2=cd_ere["RELATION"]
                                for rela_type in ["CAUSAL","TEMPORAL","SUBEVENT","COREFERENCE"]:
                                    for k,v in self.reltrule_f[rela_type].items():
                                        if k==re_1[rela_type]:
                                            for m,n in v.items():
                                                if m==re_2[rela_type]:
                                                    re_save[rela_type]=n
                                                    change+=1
                                if change !=0:
                                    save_data={
                                        "em_a":row['em_a'],
                                        "tri_a":row["tri_a"],
                                        "offset_a": row["offset_a"],
                                        "em_b": cd_ere["em_b"],
                                        "tri_b": cd_ere["tri_b"],
                                        "offset_b": cd_ere["offset_b"],
                                        "RELATION": re_save,
                                        "bg": {
                                            "e_id_a":row["bg"]["e_id_a"],
                                            "e_id_b":  cd_ere["bg"]["e_id_b"],
                                            "imd_e_id": row["bg"]["e_id_b"],
                                            "transmit":'head'
                                        }
                                    }
                                    fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")
                    #A--->B'--->?
                    # re_1   re_2
                    if transmit=='tail':
                        sd_ere_list=index_out[cd_ere["bg"]["e_id_b"]] 
                        if len(sd_ere_list)!=0 :
                            for row in sd_ere_list:
                                change=0
                                re_save={
                                    "TEMPORAL":"NO_TEMPORAL",
                                    "CAUSAL":"NO_CAUSAL",
                                    "SUBEVENT":"NO_SUBEVENT",
                                    "COREFERENCE":"NO_COREFERENCE"
                                }
                                re_1=cd_ere["RELATION"]
                                re_2=row["RELATION"]
                                #relation trans
                                for rela_type in ["CAUSAL","TEMPORAL","SUBEVENT","COREFERENCE"]:
                                    for k,v in self.reltrule_f[rela_type].items():
                                        if k==re_1[rela_type]:
                                            for m,n in v.items():
                                                if m==re_2[rela_type]:
                                                    re_save[rela_type]=n
                                                    change+=1
                                if change !=0:#真的能传导再保存
                                    save_data={
                                        "em_a":cd_ere['em_a'],
                                        "tri_a":cd_ere["tri_a"],
                                        "offset_a": cd_ere["offset_a"],
                                        "em_b": row["em_b"],
                                        "tri_b": row["tri_b"],
                                        "offset_b": row["offset_b"],
                                        "RELATION": re_save,
                                        "bg": {
                                            "e_id_a":cd_ere["bg"]["e_id_a"],
                                            "e_id_b":  row["bg"]["e_id_b"],
                                            "imd_e_id": row["bg"]["e_id_a"],
                                            "transmit":'tail'
                                        }
                                    }
                                    fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")

    def final_erc(self):
        seen_ids=set() # record (e_id_a, e_id_b)
        total_count=0
        unique_count=0
        input_files_p=[self.era_p,self.cotrans_p,self.core_p,self.reltrans_p]
        with open(self.erc_p, 'w', encoding='utf-8') as f_out:
            for file_path in input_files_p:
                print(f"\n{file_path} being processing...\n")
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        total_count += 1
                        data = json.loads(line)
                        try:
                            id_a = data['bg']['e_id_a']
                            id_b = data['bg']['e_id_b']
                            identifier = (id_a, id_b)
                            
                            if identifier not in seen_ids:
                                seen_ids.add(identifier)
                                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                                unique_count += 1
                        except KeyError:
                            #if data has key missing
                            print(f"\nSkipping a line! Because a necessary key field is missing.{line[:50]}...")

                                    

        
