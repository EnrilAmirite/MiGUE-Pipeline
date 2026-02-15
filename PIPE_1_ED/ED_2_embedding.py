#提前跑好放在本地就行(这样会快点)
#我这里用的gpt的embedding
import pickle
from openai import OpenAI
import json
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_ED_SUFFIX,CROPUS
suffix=GLOBAL_ED_SUFFIX

#----------------------------------------------------------------------------------
#API配置
#----------------------------------------------------------------------------------
HOLDAI_URL = "........."
holdai_key_vip=".........."
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
#地址
#分为:对few_shot进行embedding(1)
#和对需要ED的句子进行embedding两种(2)
#----------------------------------------------------------------------------------
CHOOSE=2
folder_embedding='PIPE_1_ED/ED_embedding'

path_few_shot="PIPE_1_ED/ED_few_shot.json"
output_name_few_shot="few_shot_embedding"
path_output_few_shot_embedding=os.path.join(folder_embedding,f"{output_name_few_shot}_{MODEL}.pkl")


if CROPUS=="RU_CLUSTER":
    path_event=f"DATA/ED_input/cluster/event_mention_{suffix}.csv"
else:
    path_event=f"DATA/ED_input/event_mention_{suffix}.csv"
output_name_event="event_embedding"
path_output_event_embedding=os.path.join(folder_embedding,f"{output_name_event}_{suffix}_{MODEL}.pkl")



#----------------------------------------------------------------------------------
#few_shot_embedding函数
#----------------------------------------------------------------------------------
def create_few_shot_embedding():
    data=[]
    with open(path_few_shot,'r',encoding='utf-8') as f:
        file_few_shot=json.load(f)
        ex_num=0
        for type_name,value in file_few_shot.items():
            for ex_id,v in value.items():
                if ex_id=="Description":
                    continue
                if not any (v.values()):
                    break
                ex_num+=1

        pbar=tqdm(total=ex_num,desc="正在生成few-shot example的embedding")
        for type_name,value in file_few_shot.items():
            for ex_id,v in value.items():
                if ex_id=="Description":
                    continue
                if not any (v.values()):#空例子(一般在尾部),直接退出
                    break
                ID=ex_id
                event_mention=v["Event_Mention"]
                embedding=call_embedding_model(event_mention)
                data.append({
                    "Type_Name":type_name,
                    "ID":ex_id,
                    "Embedding":embedding
                })
                pbar.update(1)#更新进度,pbar是一个进度条对象
    with open(path_output_few_shot_embedding, "wb") as f:
            pickle.dump(data, f)

#----------------------------------------------------------------------------------
#event_mention_embedding函数
#----------------------------------------------------------------------------------
def create_event_mention_embedding():
    data={}
    df=pd.read_csv(path_event)
    for idx,row in tqdm(df.iterrows(),total=len(df),desc="正在生成所有event mention的embedding"):
        doc_id=row["Doc_ID"]
        event_mention_id=row["Event_Mention_ID"]
        event_mention=row["Event_Mention"]
        embedding=call_embedding_model(event_mention)
        data[event_mention_id]={
            "Doc_ID": doc_id,
            "Event_Mention_ID": event_mention_id,
            "Embedding": embedding,
        }
    with open(path_output_event_embedding, "wb") as f:
        pickle.dump(data,f)

def create_event_mention_embedding_concurrent(max_workers=10):
    df = pd.read_csv(path_event)
    final_data = {}
    data_lock = threading.Lock()
    def process_row(row):
        doc_id = row["Doc_ID"]
        em_id = row["Event_Mention_ID"]
        em_text = row["Event_Mention"]
        
        try:
            embedding = call_embedding_model(em_text)
            with data_lock:
                final_data[em_id] = {
                    "Doc_ID": doc_id,
                    "Event_Mention_ID": em_id,
                    "Embedding": embedding,
                }
            return True
        except Exception as e:
            print(f"\n[Error] Event_Mention_ID {em_id} 失败: {e}")
            return False
    total_tasks = len(df)
    print(f"开始并发生成 Embedding，总数: {total_tasks}, 线程数: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row): row for _, row in df.iterrows()}        
        with tqdm(total=total_tasks, desc="正在生成所有event mention的embedding") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
    with open(path_output_event_embedding, "wb") as f:
        pickle.dump(final_data, f)
    print(f"处理完成！Embedding 已保存至: {path_output_event_embedding}")




#启动
if __name__=='__main__':
    if CHOOSE==1 :
        create_few_shot_embedding()
    else:
        create_event_mention_embedding_concurrent()