import json
import csv
import deepl
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
from pathlib import Path


class load_data:
    def __init__(self):
        pass

    def list2str(self,input_list):
        if len(input_list)==0:
            output_str='NULL'
            return output_str
        output_str=','.join(input_list)
        return output_str
    
    def str2list(self,input_str):
        output_list=[]
        if input_str=='NULL' or input_str is None:
            return output_list
        # if str is like [...] or have ' '
        clean_str = input_str.replace('[', '').replace(']', '')
        raw_list=clean_str.split(',')
        for trigger in raw_list:
            if trigger.strip():
                output_list.append(trigger)
        return output_list
    
    def  drop_csv_column(self,df,output_path,col_name):
        df = df.drop(columns=[col_name])
        df.to_csv(output_path, index=False)
        return
    
    def keep_csv_row(self,df,output_path,col_name,col_cont):
        df = df[df[col_name] == col_cont]
        df.to_csv(output_path, index=False)

    def find_offset(self,event_mention,trigger):
        offset_list=[]
        for m in re.finditer(re.escape(trigger),event_mention):
            start=m.start()
            end=m.end()
            offset_list.append([start, end])
        return offset_list
    
    def dedu_list(self,trigger_list):
        unique_list=[]
        trigger_set=set()
        for trigger in trigger_list:
            if trigger not in trigger_set:
                trigger_set.add(trigger)
                unique_list.append(trigger)
        return unique_list
    
    def write_csv_head(self,col,path):
        write_header = not os.path.exists(path) or os.path.getsize(path)==0
        if write_header:
            pd.DataFrame(columns=col).to_csv(
                path, index=False, encoding='utf-8'
            )
        return
    
    def eid2docidNemid(self,eid):
        doc_id=eid.split('_stc_')[0]
        n=eid.split('_stc_')[1].split('_')[0]
        em_id=f"{doc_id}_stc_{n}"
        return doc_id,em_id
    
    def emid2docid(self,emid):
        doc_id=emid.split('_stc_')[0]
        return doc_id


    def highlight(self,sentence, offset):
        if not sentence or not offset or len(offset)!=2:
            return sentence
        start,end=offset
        #ensure that the index is an integer and does not exceed the bounds.
        try:
            start, end = int(start), int(end)
            #boundary checking
            start=max(0, start)
            end=min(len(sentence), end)
            return f"{sentence[:start]}<{sentence[start:end]}>{sentence[end:]}"
        except (ValueError, TypeError):
            return sentence
        

    def checkfile(self,path):
        if os.path.exists(path) and os.path.isfile(path) and os.path.getsize(path) > 0:
            return 1
        else:
            return 0
        
    def path2name(self,path,tag):
        path_obj = Path(path)
        match tag:
            case 'name':
                return path_obj.name
            case 'stem':
                return path_obj.stem
            case 'suffix':
                return path_obj.suffix
            case 'folder':
                return path_obj.parent
            case 'foldername':
                return path_obj.parent.name