from src.event_relation_anno.load_erac_data import load_erac_data,record_status
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import pandas as pd
import json

class erac(load_erac_data):
    def __init__(self):
        super().__init__()
        self.w_lock=threading.Lock()
        self.ep_df=pd.read_csv(self.ep_p)
 
    def erac_one_row(self,row):
        try:
            now_stts=record_status()
            now_stts.clean_status()
            round=0
            while len(now_stts.ra_list)!=4:
                #relation choose
                now_stts.opt_list=[]
                prompt=self.create_intro_pp(row)
                model_rlt=""
                model_label=""
                #relation choose
                prompt+=self.create_rel_pp(now_stts)
                #model choose relation
                while model_rlt not in now_stts.opt_list:
                    model_rlt=self.llm.call_llm(self.sys_pp,prompt).strip()
                #opt--->rlt name
                model_rlt=self.opt2rel(model_rlt)

                #label choose
                now_stts.opt_list=[]
                prompt=self.create_intro_pp(row)
                prompt+=self.create_label_pp(model_rlt,now_stts)
                #turn to label
                while model_label not in now_stts.opt_list:
                    model_label=self.llm.call_llm(self.sys_pp,prompt).strip()
                #opt--->tabel name
                model_label=self.opt2label(model_rlt,model_label)

                #After thinking
                action=""
                while action not in ["ACT_A","ACT_B","ACT_C"]:
                    pp_thinking=self.after_thinking_pp(model_rlt,model_label,now_stts)
                    if pp_thinking=="":
                        action="ACT_A"
                    else: 
                        prompt=self.after_thinking_pp(model_rlt,model_label,now_stts)
                        prompt+=pp_thinking   
                        action=self.llm.call_llm(self.sys_pp,prompt)
                    #redo this round == without writing the answer.
                    if action=="ACT_B":
                        round+=1
                    #redo all= clean now stts
                    elif action=="ACT_C":
                        now_stts.clean_status()
                        round+=1
                    elif action=="ACT_A":
                        #renew MUST/MUST_NOT
                        now_stts.maintain_have_answer(model_rlt,model_label)
                        now_stts.maintain_must_and_must_not(model_rlt,model_label)
                        now_stts.upload_ans(model_rlt,model_label)

            answer_save = {
                "em_a":row["em_a"],
                "tri_a":row["tri_a"],
                "offset_a":row["offset_a"],
                "em_b":row["em_b"],
                "tri_b":row["tri_b"],
                "offset_b":row["offset_b"],
                "RELATION":now_stts.ans,
                "bg":{
                    "e_id_a":row['e_id_a'],
                    "e_id_b":row['e_id_b']
                }
            }
            return  answer_save
        except Exception as e:
            print(f"There is something wrong in intra-doc event relation annotation!! error:{e}")
            return {
                "em_a":row["em_a"],
                "tri_a":row["tri_a"],
                "offset_a":row["offset_a"],
                "em_b":row["em_b"],
                "tri_b":row["tri_b"],
                "offset_b":row["offset_b"],
                "RELATION":now_stts.ans,#maybe empty
                "bg":{
                    "e_id_a":row['e_id_a'],
                    "e_id_b":row['e_id_b'],
                }
            }
    
    def erac_cc(self):
        rows=self.ep_df.to_dict('records')
        tt_task=len(self.ep_df)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures=[
                executor.submit(self.erac_one_row,row) for row in rows
            ]
            with tqdm(
                total=tt_task,
                desc="Doing event relation annotation (cross documents)........."
            ) as pbar:
                for future in as_completed(futures):
                    save_answer = future.result()
                    with open(self.era_p,'a',encoding='utf-8') as f:
                        f.write(json.dumps(save_answer,ensure_ascii=False) + "\n")
                    pbar.update(1)  
        print("Finished event relation annotation (cross documents) !\n")      

    
    def era_rev(self):
        with open (self.era_p,'r',encoding='utf-8') as fin, \
        open (self.er_p,'w',encoding='utf-8') as fout :
            for line in fin:
                line=line.strip()
                if not line:
                    continue
                raw_line=json.loads(line)
                raw_rel=raw_line['RELATION']
                new_rel={
                    "TEMPORAL":"",
                    "CAUSAL":"",
                    "SUBEVENT":"",
                    "COREFERENCE":""
                }
                for rel in ["TEMPORAL","CAUSAL","SUBEVENT","COREFERENCE"]:
                    for k,v in self.relre_f[rel].items():
                        if k==raw_rel[rel]:
                            new_rel[rel]=v
                save_data={
                    "em_a":raw_line["em_b"],
                    "tri_a":raw_line["tri_b"],
                    "offset_a":raw_line["offset_b"],
                    "em_b":raw_line["em_a"],
                    "tri_b":raw_line["tri_a"],
                    "offset_b":raw_line["offset_a"],
                    "RELATION":new_rel,
                    "bg":{
                        "e_id_a":raw_line['bg']['e_id_b'],
                        "e_id_b":raw_line['bg']['e_id_a']
                    }
                }
                fout.write(line.rstrip("\n") + "\n")
                fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")

            