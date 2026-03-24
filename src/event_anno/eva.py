import pandas as pd
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.event_anno.load_eva_data import load_eva_data

class eva(load_eva_data):
    def __init__(self):
        super().__init__()
        self.stc_df=pd.read_csv(self.stc_p)
    
    def eva_one_row(self,row):
        try:
            doc_id=row["doc_id"]
            stc_id=row["stc_id"]
            stc=row["stc"]

            answer=[]
            delete_tri_list=[]

            #first anno
            prompt=self.create_first_anno(stc,stc_id)
            first_tri_str=self.llm.call_llm(self.sys_pp,prompt)
            first_tri_list=self.str2list(first_tri_str)
            answer.append(first_tri_str)

            # Missing check
            prompt=self.create_reflection(stc=stc,stc_id=stc_id,raw_triggers=answer[0],round=0)
            add_tri_str=self.llm.call_llm(self.sys_pp,prompt)
            add_tri_list=self.str2list(add_tri_str)
            for tri in add_tri_list:
                if tri not in first_tri_list:
                    first_tri_list.append(tri)
            answer.append(add_tri_str)

            #reflection
            for round_id in [1,2,3,4,5,6]:
                prompt = self.create_reflection(
                    stc,
                    stc_id,
                    first_tri_list,
                    round_id
                )
                trigger_str=self.llm.call_llm(self.sys_pp,prompt)
                trigger_list=self.str2list(trigger_str)

                delete_tri_list.extend(trigger_list)
                answer.append(trigger_str)

            # Final triggers
            final_trigger_list = [
                x for x in first_tri_list if x not in delete_tri_list
            ]
            final_trigger_str=self.list2str(final_trigger_list)

            save_answer = {
                "doc_id": doc_id,
                "stc_id": stc_id,
                "stc": stc,
                "raw_tri_list": answer[0],
                "missing_check": answer[1],
                "named_entity": answer[2],
                "narrative": answer[3],
                "no_occurence": answer[4],
                "assumption": answer[5],
                "abstraction": answer[6],
                "negated_event": answer[7],
                "final_tri_list": final_trigger_str
            }

            return save_answer

        except Exception as e:
            print(e)
            return {
                "doc_id": doc_id,
                "stc_id": stc_id,
                "stc": stc,
                "raw_tri_list": "",
                "missing_check": "",
                "named_entity": "",
                "narrative": "",
                "no_occurence": "",
                "assumption": "",
                "abstraction": "",
                "negated_event": "",
                "final_tri_list": f"ERROR: {type(e).__name__}"
            }

    def eva_cc(self):
        columns = [
            "doc_id", "stc_id", "stc",
            "raw_tri_list",
            "missing_check",
            "named_entity",
            "narrative",
            "no_occurence",
            "assumption",
            "abstraction",                
            "negated_event",
            "final_tri_list"
        ]
        self.write_csv_head(columns,self.eva_p)
        write_lock = threading.Lock()
        task=self.stc_df.to_dict('records')
        total_task= len(task)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures=[
                executor.submit(self.eva_one_row,row) for row in task
            ]
            with tqdm(
                total=total_task,
                desc="Doing event annotation........."
            ) as pbar:
                for future in as_completed(futures):
                    save_answer = future.result()
                    with write_lock:
                        pd.DataFrame([save_answer]).to_csv(
                            self.eva_p,
                            mode="a",
                            header=False,
                            index=False,
                            encoding="utf-8"
                        )
                    pbar.update(1)  
        print("Finished event annotation!\n")           