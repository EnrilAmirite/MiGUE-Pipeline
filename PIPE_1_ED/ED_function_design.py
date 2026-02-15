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
import re

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_ED_END_ROW_ID,GLOBAL_ED_PATH_RAW,GLOBAL_ED_START_ROW_ID
from CONFIG.config import GLOBAL_ED_END_ROW_ID,GLOBAL_ED_START_ROW_ID,CROPUS
suffix=f"{CROPUS}_{GLOBAL_ED_START_ROW_ID}to{GLOBAL_ED_END_ROW_ID}"

#-------------------------------------------------------------------------
#一些路径和名称和全局变量
#-------------------------------------------------------------------------
path_prompt="PIPE_1_ED/ED_prompt.json"
path_few_shot="PIPE_1_ED/ED_few_shot.json"
id_first=GLOBAL_ED_START_ROW_ID
id_last=GLOBAL_ED_END_ROW_ID
path_event_embedding=f"PIPE_1_ED/ED_embedding/event_embedding_{suffix}_text-embedding-3-small.pkl"
path_few_shot_embedding="PIPE_1_ED/ED_embedding/few_shot_embedding_text-embedding-3-small.pkl"
TOP_K=2



#-------------------------------------------------------------------------
#list to str
#-------------------------------------------------------------------------
def list_to_str(trigger_list):
    if len(trigger_list)==0:
        trigger_str='NULL'
        return trigger_str
    trigger_str=','.join(trigger_list)
    return trigger_str



#-------------------------------------------------------------------------
#str to list
#-------------------------------------------------------------------------
def str_to_list(trigger_str):
    trigger_list=[]
    if trigger_str=='NULL' or trigger_str is None:
        return trigger_list
    raw_list=trigger_str.split(',')
    for trigger in raw_list:
        if trigger.strip():
            trigger_list.append(trigger)
    return trigger_list
    


#-------------------------------------------------------------------------
#Top-k few-shot
#-------------------------------------------------------------------------
def find_top_example(type_name,event_mention_id,top_k):
    example_id_list=[]
    with open(path_event_embedding,"rb") as f:
        file_event_embedding=pickle.load(f)
    with open(path_few_shot_embedding,"rb") as f:
        file_few_shot_embedding=pickle.load(f)
    #event_id索引在embedding时已经建好了
    event_embedding=file_event_embedding[event_mention_id]["Embedding"]
    exa_list=[
        exa for exa in file_few_shot_embedding
        if exa["Type_Name"]==type_name
    ]
    #如果这个type下example个数为0,直接返回(虽然更新few-shot.json文件之后不可能没有...)
    if len(exa_list)==0:
        return example_id_list
    #example embedding list 转成numpy array 批量计算相似度...
    exa_embedding_list=np.array([
        exa["Embedding"] for exa in exa_list
    ])
    #event embedding也要转成nparray
    event_embedding=np.array(event_embedding)
    #找top-k
    #计算余弦相似度(想象一下..)最后第一行就是所有examples和当前句子的余弦相似度
    similarity_list=cosine_similarity(
        event_embedding.reshape(1,-1),
        exa_embedding_list
    )[0]
    #相似度升序排序(argsort返回的是索引...)
    #取出最后k个(即相似度最高的k个)
    #倒序
    top_k_few_shot=similarity_list.argsort()[-top_k:][::-1]
    for i in top_k_few_shot:
        data=exa_list[i]
        data={**data}#解包
        example_id_list.append(data["ID"])
    return example_id_list




#-------------------------------------------------------------------------
#few-shot样例构造
#example_name_list是已经选中的top-k 语义最相似的例子(前面的EX_1,EX_2...)
#-------------------------------------------------------------------------
def create_example(type_name,example_id_list):
    prompt=""
    now_exa=1
    with open(path_few_shot,'r',encoding='utf-8') as f:
        few_shot_file=json.load(f)
        examples=few_shot_file[type_name]
        if type_name=="Direct":
            describe=f"{examples["Description"]}\n"#direct没有描述
        else:
            describe=f"{examples["Description"]}\n下面是一些例子:\n"

        for key,v in examples.items():
            if key not in example_id_list:
                continue
            if not any(v.values()):
                break
            prompt+=f"例子{now_exa}:\n"
            prompt+=f"事件提及:{v["Event_Mention"]}\n"
            prompt+=f"正确的触发词:{v["Trigger_Words"]}\n"
            if type_name != 'Direct':
                prompt+=f"容易错误抽取的触发词:{v["Wrong_Trigger_Words"]}\n"
            prompt+=f"解释:{v["Explain"]}\n"
            now_exa+=1
        if prompt!="":
            prompt=describe+prompt
    return prompt

#print(f"这里应该没有输出:{create_example("Abstraction")}")
#print (create_example("Named_Entity"))



#-------------------------------------------------------------------------
#task介绍+第一次直接抽取trigger
#-------------------------------------------------------------------------
def create_task_intro(event_mention,event_id):
    prompt=""
    with open(path_prompt,'r',encoding='utf-8') as f:
        prompt_file=json.load(f)
        task_intro=prompt_file["Task_Intro"]
        for key,v in task_intro.items():
            if key!="Given_Event_mention":
                prompt+=v
        rules=prompt_file["Task_Rules"]
        for key,v in rules.items():
            prompt+=v
        example_id_list=find_top_example("Direct",event_id,TOP_K)
        prompt+=create_example("Direct",example_id_list)
        prompt+=task_intro["Given_Event_mention"]
        prompt+=f"{event_mention}\n"
        output_control=prompt_file["Task_Output_Control"]
        for key,v in output_control.items():
            prompt+=v
    return prompt



#-------------------------------------------------------------------------
#self reflection
#-------------------------------------------------------------------------
def create_reflection(event_mention,event_id,raw_triggers,round):
    prompt=""
    with open(path_prompt,'r',encoding='utf-8') as f:
        prompt_file=json.load(f)
        prompt_reflection=prompt_file["Self_Reflection"]
        prompt+=prompt_reflection["Previous_Intro"]
        prompt+=f"事件提及:{event_mention}\n"
        prompt+=f"当前您抽取的所有触发词:{raw_triggers}\n"
        prompt+=prompt_reflection["Reflection_Intro"]
        match round:
            case 0:
                #先给定义和rules
                type_name="Missing_Check"
                prompt+=prompt_reflection[type_name]
                task_intro=prompt_file["Task_Intro"]
                for key,v in task_intro.items():
                    if key!="Given_Event_mention":
                        prompt+=v
                rules=prompt_file["Task_Rules"]
                for key,v in rules.items():
                    prompt+=v
                example_id_list=find_top_example("Direct",event_id,TOP_K)
                prompt+=create_example("Direct",example_id_list)
            case 1:
                type_name="Named_Entity"
                prompt+=prompt_reflection[type_name]
                example_id_list=find_top_example(type_name,event_id,TOP_K)
                prompt+=create_example(type_name,example_id_list)
            case 2:
                type_name="Narrative"
                prompt+=prompt_reflection[type_name]
                example_id_list=find_top_example(type_name,event_id,TOP_K)
                prompt+=create_example(type_name,example_id_list)
            case 3:
                type_name="No_Occurrence"
                prompt+=prompt_reflection[type_name]
                example_id_list=find_top_example(type_name,event_id,TOP_K)
                prompt+=create_example(type_name,example_id_list)
            case 4:
                type_name="Assumption"
                prompt+=prompt_reflection[type_name]
                example_id_list=find_top_example(type_name,event_id,TOP_K)
                prompt+=create_example(type_name,example_id_list)
            case 5:
                type_name="Abstraction"
                prompt+=prompt_reflection[type_name]
                example_id_list=find_top_example(type_name,event_id,TOP_K)
                prompt+=create_example(type_name,example_id_list)
            case 6:
                type_name="Negated_Event"
                prompt+=prompt_reflection[type_name]
                example_id_list=find_top_example(type_name,event_id,TOP_K)
                prompt+=create_example(type_name,example_id_list)
        if round==1:
            prompt+=prompt_reflection["Output_Control_Add"]
        else:
            prompt+=prompt_reflection["Output_Control_Delete"]
        return prompt
    


#-------------------------------------------------------------------------
#找到trigger在句子中的offset
#因为可能出现的地方不止一处,所以返回的是个list
#-------------------------------------------------------------------------
def find_trigger_offset(event_mention,trigger):
    offset_list=[]
    for m in re.finditer(re.escape(trigger),event_mention):
        start=m.start()
        end=m.end()
        offset_list.append([start, end])
    return offset_list



#-------------------------------------------------------------------------
#trigger list 去重(反正都要重新算offset啊啊啊啊啊啊啊啊)
#-------------------------------------------------------------------------
def deduplication_trigger(trigger_list):
    unique_list=[]
    trigger_set=set()
    for trigger in trigger_list:
        if trigger not in trigger_set:
            trigger_set.add(trigger)
            unique_list.append(trigger)
    return unique_list



#-------------------------------------------------------------------------
#顾名思义的简单排序函数
#主要应对并发之后排序乱了的问题...
#-------------------------------------------------------------------------
def sort_by_id(path_input):
    df=pd.read_csv(path_input)
    df["Doc_ID"] = df["Doc_ID"].astype(int)
    #event mention id末尾的数值作为二次索引
    df["Event_Index"] = (
        df["Event_Mention_ID"]
        .str.extract(r'_(\d+)$')
        .astype(int)
    )
    #排序..排序
    df = df.sort_values(
        by=["Doc_ID", "Event_Index"],
        ascending=[True, True]
    )
    #排序完删掉辅助列
    df = df.drop(columns=["Event_Index"])
    df.to_csv(path_input,index=False)



#-------------------------------------------------------------------------
#最后一步:把trigger List分了变成真正的event: 
# event mention,event id, event mention id,event trigger
#-------------------------------------------------------------------------
def create_final_event(path_input,path_output):
    columns=["Doc_ID","Event_Mention_ID","Event_ID","Event_Mention","Trigger","Trigger_Offset"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    total_lines=sum(1 for _ in open(path_input))-1
    ED_result=pd.read_csv(path_input)
    for idx,row in tqdm(ED_result.iterrows(),total=len(ED_result),desc="正在创建event..."):
        if pd.isna(row['Final_Trigger_List']):#pandas读取NULL,null都是NaN
            continue
        save_data={}
        trigger_list=str_to_list(row['Final_Trigger_List'])
        trigger_list=deduplication_trigger(trigger_list)
        for n,trigger in enumerate(trigger_list):
            offset_list=find_trigger_offset(row['Event_Mention'],trigger)
            for m,offset in enumerate(offset_list):
                event_id=f"{row['Event_Mention_ID']}_{n+1}_{m+1}"
                save_data={
                    "Doc_ID":row['Doc_ID'],
                    "Event_Mention_ID":row['Event_Mention_ID'],
                    "Event_ID":event_id,
                    "Event_Mention":row['Event_Mention'],
                    "Trigger":trigger,
                    "Trigger_Offset":offset
                }
                pd.DataFrame([save_data]).to_csv(
                    path_output,
                    mode='a',#追加写.........
                    header=False, #不用写表头了
                    index=False,
                    encoding='utf-8'
                )

                

#----------------------------------------------------------------------------------------
#最后一步:构建doc-(真正的)event mention组
#有些不含event 的'event mention'就不包含在内啦,所以我这里放在ED了(因为要在ED完成后后处理)
#主要是方便后面ERE
#Doc_ID,Doc_Time,Event_Mention_ID,Event_Mention,Doc
#----------------------------------------------------------------------------------------
def create_doc_and_real_event_mention(path_input_event,path_input_doc,path_output):
    columns=["Doc_ID","Doc_Time","Event_Mention_ID","Event_Mention","Doc"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    ED_event_file=pd.read_csv(path_input_event)
    raw_doc_file=pd.read_csv(path_input_doc)
    raw_doc_file=raw_doc_file.set_index("doc_id")
    for idx,row in tqdm(ED_event_file.iterrows(),total=len(ED_event_file),desc="正在构建最终的doc & event mention文档..."):
        #没有Trigger的'event mention'就不在后面ERE的考虑范围内了,直接跳过
        if pd.isna(row['Final_Trigger_List']):#pandas读取NULL,null都是NaN
            continue
        doc_id=row['Doc_ID']
        event_mention_id=row['Event_Mention_ID']
        doc=raw_doc_file.loc[doc_id]['content']
        doc_time=raw_doc_file.loc[doc_id]['created_at']
        event_mention=row['Event_Mention']
        save_data={
            "Doc_ID":doc_id,
            "Doc_Time":doc_time,
            "Event_Mention_ID":event_mention_id,
            "Event_Mention":event_mention,
            "Doc":doc
        }
        pd.DataFrame([save_data]).to_csv(
            path_output,
            mode='a',#追加写.........
            header=False, #不用写表头了
            index=False,
            encoding='utf-8'
        )


