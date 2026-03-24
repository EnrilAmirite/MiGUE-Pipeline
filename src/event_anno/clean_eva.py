from src.event_anno.load_eva_config import load_eva_config
import pandas as pd
from tqdm import tqdm

class clean_eva(load_eva_config):
    def __init__(self):
        super().__init__()
    
    def create_event(self):
        columns=["doc_id","em_id","event_id","em","trigger","offset"]
        self.write_csv_head(col=columns,path=self.ev_p)
        total_lines=sum(1 for _ in open(self.eva_p))-1
        ED_result=pd.read_csv(self.eva_p)
        for idx,row in tqdm(ED_result.iterrows(),total=len(ED_result),desc="Creating annotated event ..."):
            if pd.isna(row['final_tri_list']):#in pandas, NULL,null=NaN
                continue
            save_data={}
            trigger_list=self.str2list(row['final_tri_list'])
            trigger_list=self.dedu_list(trigger_list)
            for n,trigger in enumerate(trigger_list):
                offset_list=self.find_offset(row['stc'],trigger)
                for offset in offset_list:
                    event_id=f"{row['stc_id']}_ev_{n+1}"
                    save_data={
                        "doc_id":row['doc_id'],
                        "em_id":row['stc_id'],
                        "event_id":event_id,
                        "em":row['stc'],
                        "trigger":trigger,
                        "offset":offset
                    }
                    pd.DataFrame([save_data]).to_csv(
                        self.ev_p,
                        mode='a',
                        header=False,
                        index=False,
                        encoding='utf-8'
                    ) 
        
    def create_docNem_index(self):
        columns=["doc_id","em_id"]
        self.write_csv_head(col=columns,path=self.docNem_p)
        ED_event_file=pd.read_csv(self.eva_p)
        for idx,row in tqdm(ED_event_file.iterrows(),total=len(ED_event_file),desc="creating real event mention index..."):
            #if one 'event mention' has no trigger, then it isn't an event mention
            if pd.isna(row['final_tri_list']):#in pandas, NULL,null=NaN
                continue
            doc_id=row['doc_id']
            em_id=row['stc_id']
            save_data={
                "doc_id":doc_id,
                "em_id":em_id,
            }
            pd.DataFrame([save_data]).to_csv(
                self.docNem_p,
                mode='a',
                header=False,
                index=False,
                encoding='utf-8'
            )