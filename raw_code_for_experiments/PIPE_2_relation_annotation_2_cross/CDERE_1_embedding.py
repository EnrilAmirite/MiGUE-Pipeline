import pickle
import numpy as np
from openai import OpenAI
import json
import os
import pandas as pd
from tqdm import tqdm
import faiss
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CDERE_SUFFIX
suffix=GLOBAL_CDERE_SUFFIX
#-----------------------------------------------------------------------------------------
#路径
#这里只用跑doc&real event mention information的文件
#-----------------------------------------------------------------------------------------
path_event_file=f"DATA/ED_output/ED_doc_and_real_event_mention_{suffix}.csv"
folder_CDERE_embedding="PIPE_3_ERE_CROSS_DOC/CDERE_embedding"
output_name_event_id=f"EVENT_ID_{suffix}"
output_name_event_id_to_idx=f"EVENT_ID_TO_IDX_{suffix}"
output_name_event_embedding=f"EVENT_ID_TO_EMBEDDING_{suffix}"
output_name_event_index=f"EVENT_NORMAL_INDEX_{suffix}"
path_output_event_id=os.path.join(folder_CDERE_embedding,f"{output_name_event_id}.pkl")
path_output_event_id_to_idx=os.path.join(folder_CDERE_embedding,f"{output_name_event_id_to_idx}.pkl")
path_output_event_embedding=os.path.join(folder_CDERE_embedding,f"{output_name_event_embedding}.pkl")
path_output_event_index=os.path.join(folder_CDERE_embedding,f"{output_name_event_index}.index")
#----------------------------------------------------------------------------------
#API配置
#----------------------------------------------------------------------------------
HOLDAI_URL = "https://api.holdai.top/v1"
holdai_key_vip="......................."
#模型选择
MODEL="text-embedding-3-small"

# 配置 
client = OpenAI(
    api_key=holdai_key_vip, 
    base_url=HOLDAI_URL
)
#召唤代码
def call_embedding_model(text):
    response=client.embeddings.create(
        model=MODEL,
        input=text
    )
    return response.data[0].embedding


#----------------------------------------------------------------------------------
#event mention embedding函数
#但是这里我们直接存储成numpy array
#----------------------------------------------------------------------------------
def create_event_mention_embedding_as_numpy_array():    
    all_event_embedding_list=[]
    all_event_id_list=[]
    event_id_to_embedding={}
    df=pd.read_csv(path_event_file)
    for idx,row in tqdm(df.iterrows(),total=len(df),desc="正在生成所有event mention的embedding"):
        event_mention_id=row["Event_Mention_ID"]
        event_mention=row["Event_Mention"]
        doc_id=row["Doc_ID"]
        embedding=call_embedding_model(event_mention)
        all_event_embedding_list.append(embedding)
        all_event_id_list.append(event_mention_id)
        event_id_to_embedding[event_mention_id]={
            'Event_Mention_ID':event_mention_id,
            'Embedding':np.array(embedding,dtype='float32'),
            'Doc_ID':doc_id,
        }
    #转成numpy array
    all_event_embedding=np.array(
        all_event_embedding_list,dtype='float32'
    )
    #先做L2归一化(想象一下后面的cosin)
    faiss.normalize_L2(all_event_embedding)
    #构建faiss index...
    dimention=all_event_embedding.shape[1]
    all_event_embedding_normalize=faiss.IndexFlatIP(dimention)
    all_event_embedding_normalize.add(all_event_embedding)
    #保存index
    faiss.write_index(all_event_embedding_normalize,path_output_event_index)
    #保存event_id
    with open(path_output_event_id, "wb") as f:
        pickle.dump(all_event_id_list,f)
    event_id_to_idx={}
    #保存event_id_to_idx的映射
    for i,event_mention_id in enumerate(all_event_id_list):
        event_id_to_idx[event_mention_id]={
            "IDX":i,
        }
    with open(path_output_event_id_to_idx, "wb") as f:
        pickle.dump(event_id_to_idx, f)
    with open(path_output_event_embedding,"wb") as f:
        pickle.dump(event_id_to_embedding,f)
    print("已经完成所有event mention的向量化啦~")
    
if __name__=='__main__':
    create_event_mention_embedding_as_numpy_array()
    