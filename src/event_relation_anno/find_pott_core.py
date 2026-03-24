import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from src.event_relation_anno.load_erac_config import load_erac_config
from src.base_code.load_data import load_data


class find_pott_core(load_erac_config):
    def __init__(self, status_cfg_path='config/event_relation_anno_cross.yaml', status='event_relation_annotation_cross'):
        super().__init__(status_cfg_path, status)
        self.e_df=pd.read_csv(self.e_p)
        with open (self.stc_ebd_p,'rb') as f:
            self.stc_ebd_pkl=pickle.load(f)

    def find_semantic_similarity(self, target_stc_id, data_dict):
        threshold=self.thsh
        if target_stc_id not in data_dict:
            return []
        all_ids = list(data_dict.keys())
        #embedding-->[D], [N,D]
        all_ebds = np.array([data_dict[sid]['stc_ebd'] for sid in all_ids])
        #target-->[1, D]
        target_ebd = data_dict[target_stc_id]['stc_ebd'].reshape(1, -1)

        #cos simi
        #norm
        norm_target = target_ebd / np.linalg.norm(target_ebd, axis=1, keepdims=True)
        norm_all = all_ebds / np.linalg.norm(all_ebds, axis=1, keepdims=True)
        #find simi
        #[1, D] * [D, N]-->[1, N]
        similarities = np.dot(norm_target, norm_all.T).flatten()
        results = []
        for idx, sim in enumerate(similarities):
            # from different documents
            if all_ids[idx] != target_stc_id and data_dict[target_stc_id]['doc_id']!= data_dict[all_ids[idx]]['doc_id'] and sim > threshold:
                results.append({
                    "stc_id": all_ids[idx],
                    "similarity": sim,
                    "doc_id": data_dict[all_ids[idx]]['doc_id']
                })
        #sort
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def create_epc(self):
        valid_indices = []
        # ebd matrix...
        embeddings = []
        
        for idx, row in self.e_df.iterrows():
            em_id = str(row['em_id'])
            if em_id in self.stc_ebd_pkl:
                embeddings.append(self.stc_ebd_pkl[em_id]['stc_ebd'])
                valid_indices.append(idx)
        # matrix [N, D]
        df_valid = self.e_df.iloc[valid_indices].reset_index(drop=True)
        ebds_matrix = np.array(embeddings)
        
        #cosin similarity
        #norm
        norm = np.linalg.norm(ebds_matrix, axis=1, keepdims=True)
        norm_ebds = ebds_matrix / (norm + 1e-9) # 防止除以0
        sim_matrix = np.dot(norm_ebds, norm_ebds.T)

        #differen document
        doc_ids = df_valid['doc_id'].values
        diff_doc_mask = doc_ids[:, None] != doc_ids[None, :]
        #upper triangle
        final_mask = np.triu(sim_matrix, k=1) > self.thsh
        combined_mask = final_mask & diff_doc_mask
        rows, cols = np.where(combined_mask)

        results = []
        for r, c in zip(rows, cols):
            row_a = df_valid.iloc[r]
            row_b = df_valid.iloc[c]
            pair = {
                "doc_id_a": row_a['doc_id'],
                "em_id_a": row_a['em_id'],
                "e_id_a": row_a['event_id'],
                "em_a": row_a['em'],
                "tri_a": row_a['trigger'],
                "offset_a": row_a['offset'],
                
                "doc_id_b": row_b['doc_id'],
                "em_id_b": row_b['em_id'],
                "e_id_b": row_b['event_id'],
                "em_b": row_b['em'],
                "tri_b": row_b['trigger'],
                "offset_b": row_b['offset']

                #"similarity": sim_matrix[r, c]
            }
            results.append(pair)
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.ep_p, index=False, encoding='utf-8-sig')
        else:
            print("\n\nNo potential event pairs were found, so file was empty.\n\n")
            return
        print('\n\nAll potential cross-document coreference event pairs have been found ~\n\n')
        return 
        
        
        
