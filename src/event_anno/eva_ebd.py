import pickle
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import json

from src.event_anno.load_eva_config import load_eva_config

class eva_ebd(load_eva_config):
    def __init__(self):
        super().__init__()
    
    def create_case_ebd(self):
        if self.checkfile(self.case_ebd_p):
            return
        data=[]
        ex_num=0
        for case_type,value in self.case_f.items():
            for ex_id,v in value.items():
                if ex_id=="description":
                    continue
                if not any (v.values()):
                    break
                ex_num+=1

        pbar=tqdm(total=ex_num,desc="Case embedding..........")
        for case_type,value in self.case_f.items():
            for ex_id,v in value.items():
                if ex_id=="description":
                    continue
                if not any (v.values()):#empty cases
                    break
                em=v["event_mention"]
                embedding=self.ebdllm.call_llm(em)
                data.append({
                    "case_type":case_type,
                    "id":ex_id,
                    "ebd":embedding
                })
                pbar.update(1)
            with open(self.case_ebd_p, "wb") as f:
                pickle.dump(data, f)

    def create_stc_ebd(self):
        data={}
        df=pd.read_csv(self.stc_p)
        for idx,row in tqdm(df.iterrows(),total=len(df),desc="Embedding.............."):
            doc_id=row["doc_id"]
            stc_id=row["stc_id"]
            stc=row["stc"]
            embedding=self.ebdllm.call_llm(stc)
            data[stc_id]={
                "doc_id": doc_id,
                "stc_id": stc_id,
                "stc_ebd": embedding,
            }
        with open(self.stc_ebd_p, "wb") as f:
            pickle.dump(data,f)


    def create_stc_ebd_cc(self):
        if self.checkfile(self.stc_ebd_p):
            return
        df=pd.read_csv(self.stc_p)
        final_data = {}
        data_lock = threading.Lock()

        def process_row(row):
            doc_id=row["doc_id"]
            stc_id=row["stc_id"]
            stc=row["stc"]
            
            try:
                embedding=self.ebdllm.call_llm(stc)
                with data_lock:
                    final_data[stc_id]={
                        "doc_id": doc_id,
                        "stc_id": stc_id,
                        "stc_ebd": embedding,
                    }
                #print(f"finish {stc_id}")
                return True
            except Exception as e:
                print(f"There is something wrong!\n sentence: {stc_id}:{stc}\n error:{e}")
                return False
            
        total_tasks = len(df)

        print(f"Doing sentences' embedding ^w^!\n sentences' sum: {total_tasks}, \n concurrent workers: {self.max_workers}")
        #print(f"df length = {len(df)}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_row, row): row for _,row in df.iterrows()}        
            with tqdm(total=total_tasks, desc="Embedding...........") as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
        with open(self.stc_ebd_p, "wb") as f:
            pickle.dump(final_data, f)
        print(f"Finish sentences Embedding ~")