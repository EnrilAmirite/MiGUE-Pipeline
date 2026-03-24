import pickle
import numpy as np
from openai import OpenAI
import json
import os
import pandas as pd
from tqdm import tqdm
import faiss


from src.event_relation_anno.load_erac_config import load_erac_config


class ev2pair_cd(load_erac_config):
    def __init__(self):
        load_erac_config.__init__(self)
        self.e_df=pd.read_csv(self.e_p)
        #em_df: drop rows with same event mention
        self.em_df = self.e_df.drop_duplicates('em_id')
        self.doc_df=pd.read_csv(self.doc_p)

    # event mention embedding as numpy array
    def create_em_ebd(self):
        if self.checkfile(self.em_ebd_p):
            return
        all_em_ebd_list=[]
        all_em_id_list=[]
        emid2ebd={}
        for idx,row in tqdm(self.em_df.iterrows(),total=len(self.em_df),desc="Creating all event mentions' embedding......."):
            em_id=row["em_id"]
            event_mention=row["em"]
            doc_id=row["doc_id"]
            embedding=self.ebdllm.call_llm(event_mention)
            all_em_ebd_list.append(embedding)
            all_em_id_list.append(em_id)
            emid2ebd[em_id]={
                'em_id':em_id,
                'ebd':np.array(embedding,dtype='float32'),
                'doc_id':doc_id,
            }
        #em ebd list---> numpy array
        all_em_ebd=np.array(
            all_em_ebd_list,dtype='float32'
        )
        #normalize
        faiss.normalize_L2(all_em_ebd)
        #faiss index...
        dimention=all_em_ebd.shape[1]
        all_em_ebd_normalize=faiss.IndexFlatIP(dimention)
        all_em_ebd_normalize.add(all_em_ebd)
        #idx ---> em ebd(index)
        faiss.write_index(all_em_ebd_normalize,self.idx2ebd_p)

        #save emid index as pkl (as emid--->idx)
        with open(self.emid_p, "wb") as f:
            pickle.dump(all_em_id_list,f)

        emid2idx={}
        #emid and idx
        for i,em_id in enumerate(all_em_id_list):
            emid2idx[em_id]={
                "IDX":i,
            }
        
        #save emid ---> idx as pkl
        with open(self.emid2idx_p, "wb") as f:
            pickle.dump(emid2idx, f)

        #save emid-->ebd as pkl
        with open(self.em_ebd_p,"wb") as f:
            pickle.dump(emid2ebd,f)
        print("Finish all event mention vectorization !")
        #self.idx2ebd_f=faiss.read_index(self.idx2ebd_p)



    def find_semantic_similarity(self,em_id):
        #idx2ebd_f is a np array 
        self.idx2ebd_f=faiss.read_index(self.idx2ebd_p)
        query_event_embedding=self.em_ebd_f[em_id]['ebd'].reshape(1, -1)
        query_event_doc_id=self.em_ebd_f[em_id]['doc_id']
        #找top-k
        scores,top_idx_list=self.idx2ebd_f.search(query_event_embedding,self.top_k)
        #idx--->emid and doc id
        top_em_id_list=[]
        top_doc_id_list=[]
        for score,idx in zip(scores[0],top_idx_list[0]):
            #print(f"debug打印:{em_id}的前5个分数是...... {scores[0][:5]}")
            #print(f"My threshold setting is:{self.thsh}")
            if score >= self.thsh:
                if em_id!=self.emid_f[idx] and query_event_doc_id!=self.em_ebd_f[self.emid_f[idx]]['doc_id']:#不要存自己..也不要存同个doc里的event
                    top_em_id_list.append(self.emid_f[idx])
                    top_doc_id_list.append(self.em_ebd_f[self.emid_f[idx]]['doc_id'])
            else:
                break
        return top_em_id_list



    def create_cdep(self):
        #print(self.top_k)
        with open(self.emid_p,"rb") as f:
            self.emid_f=pickle.load(f)
        with open(self.em_ebd_p,"rb") as f:
            self.em_ebd_f=pickle.load(f)
        file_exists = os.path.exists(self.ep_p) and os.path.getsize(self.ep_p) > 0
        if file_exists:
            self.ep_df=pd.read_csv(self.ep_p)
            return
        columns=["em_a","e_id_a","tri_a","offset_a",
                "em_b","e_id_b","tri_b","offset_b",
                "doc_a_time","doc_b_time","doc_a","doc_b"]
        self.write_csv_head(columns,self.ep_p)

        #turn id 2 string
        self.e_df['em_id'] = self.e_df['em_id'].astype(str)
        emid2em_map = self.e_df.drop_duplicates('em_id').set_index('em_id')['em'].to_dict()

        #set index
        self.doc_df = self.doc_df.set_index("doc_id")
        self.doc_df.index = self.doc_df.index.astype(str)
        e_df_indexed = self.e_df.set_index('em_id')

        #group
        groups = list(self.e_df.groupby('em_id'))
        pbar = tqdm(total=len(groups), desc='Creating event pairs')

        #buffer
        save_buffer = []

        for em_id_a, event_mention_df in groups:
            em_id_a = str(em_id_a)
            event_mention_a = emid2em_map.get(em_id_a)
            
            doc_id_a = str(self.emid2docid(em_id_a))
            
            #if can't find doc id, then pass
            if doc_id_a not in self.doc_df.index:
                pbar.update(1)
                print("\n\nCan't find this doc id TT\n\n")
                continue
                
            doc_time_a = self.doc_df.loc[doc_id_a]['doc_time']
            doc_a = self.doc_df.loc[doc_id_a]['doc']

            top_event_mention_id_list = self.find_semantic_similarity(em_id_a)
            #print (top_event_mention_id_list)
            
            for em_id_b in top_event_mention_id_list:
                em_id_b = str(em_id_b)
                if em_id_b not in emid2em_map or em_id_b not in e_df_indexed.index:
                    print("\n\nCan't find this event mention TT\n\n")
                    continue
                
                doc_id_b = str(self.emid2docid(em_id_b))
                if doc_id_b not in self.doc_df.index:
                    print("\n\nCan't find this doc id TT\n\n")
                    continue

                #event b
                event_mention_b = emid2em_map.get(em_id_b)
                doc_time_b = self.doc_df.loc[doc_id_b]['doc_time']
                doc_b = self.doc_df.loc[doc_id_b]['doc']
                
                # get em_id_b--->all events
                event_mention_b_rows = e_df_indexed.loc[[em_id_b]]

                for _, event_a in event_mention_df.iterrows():
                    for _, event_b in event_mention_b_rows.iterrows():
                        save_buffer.append({
                            "em_a": event_mention_a, 
                            "e_id_a": event_a['event_id'],
                            "tri_a": event_a['trigger'], 
                            "off_a": event_a['offset'],
                            "em_b": event_mention_b, 
                            "e_id_b": event_b['event_id'],
                            "tri_b": event_b['trigger'], 
                            "off_b": event_b['offset'],
                            "doc_a_time": doc_time_a, 
                            "doc_b_time": doc_time_b,
                            "doc_a": doc_a, 
                            "doc_b": doc_b,
                        })

            #500 save
            if len(save_buffer) >= 500:
                pd.DataFrame(save_buffer).to_csv(self.ep_p, mode='a', header=False, index=False, encoding='utf-8')
                save_buffer = [] 

            pbar.update(1)

        if save_buffer:
            pd.DataFrame(save_buffer).to_csv(self.ep_p, mode='a', header=False, index=False, encoding='utf-8')
            
        pbar.close()