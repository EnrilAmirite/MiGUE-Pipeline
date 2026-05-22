from src.event_relation_anno.load_erai_config import load_erai_config
import pandas as pd
import os


class ev2pair(load_erai_config):
    def __init__(self):
        load_erai_config.__init__(self)
        self.e_df=pd.read_csv(self.e_p)
        self.doc_df=pd.read_csv(self.doc_p)

    def create_epair(self):
        columns=["em_a","e_id_a","tri_a","offset_a",
                "em_b","e_id_b","tri_b","offset_b",
                "doc_time","doc"]
        file_exists = os.path.exists(self.ep_p) and os.path.getsize(self.ep_p) > 0
        if file_exists:
            self.ep_df=pd.read_csv(self.ep_p)
            if len(self.ep_df) > 0:
                return
            print(f"{self.ep_p} has no event pairs. Regenerating it...")
        pd.DataFrame(columns=columns).to_csv(
            self.ep_p,
            index=False,
            encoding='utf-8'
        )
        self.doc_df=self.doc_df.set_index('doc_id')
        #intra=same doc id
        for group,group_df in self.e_df.groupby('doc_id'):
            #reset index
            group_df = group_df.reset_index(drop=True)
            doc_id=group_df.iloc[0]['doc_id']
            doc_row=self.doc_df.loc[doc_id]
            doc_time=doc_row['doc_time']
            doc=doc_row['doc']
            for i in range(len(group_df)):
                e_a_row=group_df.iloc[i]
                em_a=e_a_row['em']
                e_id_a=e_a_row['event_id']
                tri_a=e_a_row['trigger']
                offset_a=e_a_row['offset']
                for j in range(i+1, len(group_df)):
                    e_b_row=group_df.iloc[j]
                    em_b=e_b_row['em']
                    e_id_b=e_b_row['event_id']
                    tri_b=e_b_row['trigger']
                    offset_b=e_b_row['offset']
                    save_data={
                        "em_a":em_a,
                        "e_id_a":e_id_a,
                        "tri_a":tri_a,
                        "offset_a":offset_a,
                        "em_b":em_b,
                        "e_id_b":e_id_b,
                        "tri_b":tri_b,
                        "offset_b":offset_b,
                        "doc_time":doc_time,
                        "doc":doc
                    }
                    pd.DataFrame([save_data]).to_csv(
                        self.ep_p,
                        mode='a',#add.........
                        header=False, #do not need write head
                        index=False,
                        encoding='utf-8'
                    )
        self.ep_df=pd.read_csv(self.ep_p)
