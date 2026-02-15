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
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CLUSTER_SUFFIX
suffix=GLOBAL_CLUSTER_SUFFIX
#----------------------------------------------------------------------------------------
#输入输出路径,输出名称
#----------------------------------------------------------------------------------------
#embedding-cluster
path_cls_event_mention_normal=f"PIPE_0_CLUSTER/CLUSTER_embedding/EVENT_NORMAL_INDEX_{suffix}.index"#顺序存储event_mention_embedding
path_cls_event_mention_id=f"PIPE_0_CLUSTER/CLUSTER_embedding/EVENT_ID_{suffix}.pkl"#顺序存储event_mention_id
path_cls_event_mention_id_to_idx=f"PIPE_0_CLUSTER/CLUSTER_embedding/EVENT_ID_TO_IDX_{suffix}.pkl"#event_mention_id到idx的索引
path_cls_event_mention_id_to_embedding=f"PIPE_0_CLUSTER/CLUSTER_embedding/EVENT_ID_TO_EMBEDDING_{suffix}.pkl"#event_mention_id到embedding的索引



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#信息检索函数
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def find_semantic_similarity_cluster(event_mention_id,top_k,similarity_threshold):
    with open(path_cls_event_mention_id,"rb") as f:
        all_event_id_list=pickle.load(f)
    with open(path_cls_event_mention_id_to_embedding,"rb") as f:
        event_id_to_embedding=pickle.load(f)
    all_event_embedding=faiss.read_index(path_cls_event_mention_normal)
    #这里我Embedding就保存的np array..只用重构形状即可
    query_event_embedding=event_id_to_embedding[event_mention_id]['Embedding'].reshape(1, -1)
    query_event_doc_id=event_id_to_embedding[event_mention_id]['Doc_ID']
    #找top-k
    scores,top_idx_list=all_event_embedding.search(query_event_embedding,top_k)
    #转换回真正的event_id和对应的doc_id
    top_event_mention_id_list=[]
    top_doc_id_list=[]
    for score,idx in zip(scores[0],top_idx_list[0]):
        if score >= similarity_threshold:
            if event_mention_id!=all_event_id_list[idx] and query_event_doc_id!=event_id_to_embedding[all_event_id_list[idx]]['Doc_ID']:#不要存自己..也不要存同个doc里的event
                top_event_mention_id_list.append(all_event_id_list[idx])
                top_doc_id_list.append(event_id_to_embedding[all_event_id_list[idx]]['Doc_ID'])
        else:
            break
    return top_event_mention_id_list



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#文档聚类函数CLUSTER 部分
#主要是给task-3 和 task-4准备的
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------
#保存"连线的" event_ment_id : similar event_ment_id list
#-------------------------------------------------------------------------------------
def line_similar_event_mention(path_input_doc_em_index,path_output):
    doc_em_file=pd.read_csv(path_input_doc_em_index)
    with open(path_output, "a", encoding="utf-8") as f:
        for idx,row in tqdm(
            doc_em_file.iterrows(),
            total=len(doc_em_file),
            desc="正在查找相似 Event Mention......"
        ):
            em_id=row["Event_Mention_ID"]
            similar_em_id_list=find_semantic_similarity_cluster(
                event_mention_id=em_id,
                top_k=20,
                similarity_threshold=0.8
            )
            data={
                "Event_Mention_ID":em_id,
                "Similar_EM_ID_List":similar_em_id_list
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
