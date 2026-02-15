from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count, Lock, Manager
from functools import partial


#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_SDERE_SUFFIX,CROPUS
suffix=GLOBAL_SDERE_SUFFIX


from PIPE_2_relation_annotation_intra.SDERE_function_design_concurrency import SDERE_prompt_intro
from PIPE_2_relation_annotation_intra.SDERE_function_design_concurrency import maintain_clean_global,maintain_must_and_must_not,maintain_have_answer
from PIPE_2_relation_annotation_intra.SDERE_function_design_concurrency import option_to_relation_name,option_to_specific_name
from PIPE_2_relation_annotation_intra.SDERE_function_design_concurrency import create_specific_question,create_relation_question,after_thinking
from PIPE_2_relation_annotation_intra.SDERE_function_design_concurrency import init_state
import re




#---------------------------------------------------------------------
#api配置...
#---------------------------------------------------------------------
#模型选择
#MODEL="gpt-4o-mini"
MODEL="gpt-4o-2024-08-06"
#MODEL="gpt-5.2-pro"
#MODEL="gpt-5.1"
#MODEL="deepseek-reasoner"
#MODEL="claude-haiku-4-5-20251001"
#MODEL="qwen3-max"
#MODEL="gemini-3-pro-preview-thinking-*"
#MODEL="qwen3-235b-a22b-instruct-2507"
#MODEL="qwen3-30b-a3b-instruct-2507"
#MODEL="qwen3-8b"
#MODEL="claude-opus-4-5-20251101-thinking"
#MODEL="glm-4.7"
#MODEL="llama-3.1-8b-instruct"
#MODEL="llama-3-8b-instruct"
#MODEL="llama-3-70b-instruct"


def api_setting(model_name):
    model_url=""
    model_key=""
    if model_name in ["gpt-4o-mini","gpt-4o-2024-08-06","gpt-5.2-pro","gpt-5.1","claude-haiku-4-5-20251001","qwen3-max","gemini-3-pro-preview-thinking-*","qwen3-235b-a22b-instruct-2507","qwen3-30b-a3b-instruct-2507","qwen3-8b","claude-opus-4-5-20251101-thinking","glm-4.7","llama-3.1-8b-instruct","llama-3-8b-instruct","llama-3-70b-instruct"]:
        model_url="https://api.holdai.top/v1"
        model_key="......................."
    elif model_name in ["deepseek-reasoner","deepseek-chat"]:
        model_url="https://api.deepseek.com/v1"
        model_key="......................."
    return model_url,model_key

MODEL_URL,MODEL_KEY=api_setting(MODEL)
# 配置
client=OpenAI(
    api_key=MODEL_KEY, 
    base_url=MODEL_URL
    )
#-------------------------------------------------------------------------
#调用llm...
#-------------------------------------------------------------------------
def call_llm(prompt, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": "You are an expert in the field of event. Extract the event relation."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=1.0,
    )

    ans= response.choices[0].message.content
    return ans



#==================================输入输出路径,输出名称=================================
if CROPUS=="RU_CLUSTER":
    path_input=f"DATA/SDERE_input/cluster/SDERE_1_event_pair_{suffix}.csv"
    path_output="DATA/SDERE_output/cluster"
else:
    path_input=f"DATA/SDERE_input/SDERE_1_event_pair_{suffix}.csv"
    path_output="DATA/SDERE_output"

output_name=f"SDERE_2"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = os.path.join(path_output, f"{output_name}_{suffix}_{timestamp}.jsonl")





#防止model不按要求输出...
OPTION_LIST=[]





#-------------------------------------------------
#并发主要是改两个地方
#第一个是每个循环先创建一个init_state
#第二个是所有的maintain函数/create relation 和create specific 都需要传state
#-------------------------------------------------

def each_row_processing(row,lock):
    state=init_state()
    doc=row["Doc"]
    doc_id=row["Doc_ID"]
    event_mention_a=row["Event_Mention_A"]
    event_mention_b=row["Event_Mention_B"]
    trigger_a=row["Trigger_A"]
    trigger_b=row["Trigger_B"]
    offset_a=row["Trigger_Offset_A"]
    offset_b=row["Trigger_Offset_B"]
    ANSWER={
        "TEMPORAL":"",
        "CAUSAL":"",
        "SUBEVENT":"",
        "COREFERENCE":""
    }
    round=0
    while len(state["SPECIFIC_ANSWER_LIST"])!=4:
        #首先描述问题
        OPTION_LIST=[]
        prompt=SDERE_prompt_intro(
            doc=doc,
            event_mention_A=event_mention_a,
            event_mention_B=event_mention_b,
            trigger_A=trigger_a,
            trigger_B=trigger_b,
            offset_A=offset_a,
            offset_B=offset_b
        )
        model_relation_option=""
        model_specific_option=""
        #可以回答的schema_relation选项
        temp,OPTION_LIST=create_relation_question(state)
        prompt+=temp
        #模型返回选择的schema_relation 选项
        while model_relation_option not in OPTION_LIST:
            model_relation_option = call_llm(prompt).strip()
            #print(model_relation_option)
        #schema_relation 选项转换成真实的relation
        model_relation_answer=option_to_relation_name(model_relation_option)
        #print(prompt)
        #print(OPTION_LIST)
        #print(model_relation_answer)
        

        #可以选择的Specific_relation选项
        OPTION_LIST=[]
        prompt=SDERE_prompt_intro(
            doc=doc,
            event_mention_A=event_mention_a,
            event_mention_B=event_mention_b,
            trigger_A=trigger_a,
            trigger_B=trigger_b,
            offset_A=offset_a,
            offset_B=offset_b
        )
        temp,OPTION_LIST=create_specific_question(state,model_relation_answer)
        prompt+=temp
        #模型返回选择的Specific_relation选项
        while model_specific_option not in OPTION_LIST:
            model_specific_option=call_llm(prompt).strip()
        #转换..
        model_specific_answer=option_to_specific_name(model_relation_answer,model_specific_option)
        #print(prompt)
        #print(model_specific_answer)

        #后验环节
        action=""
        while action not in ["ACT_A","ACT_B","ACT_C"]:
            prompt_thinking=after_thinking(model_relation_answer,model_specific_answer)
            if prompt_thinking=="":
                action="ACT_A"
            else: 
                prompt=after_thinking(model_relation_answer,model_specific_answer)
                prompt+=prompt_thinking   
                action=call_llm(prompt)
            #print(prompt)
            #print(action)
            #重做这轮不写入答案即可
            if action=="ACT_B":
                round+=1
            #全部重做即清空现在所有的全局变量
            elif action=="ACT_C":
                maintain_clean_global(state)
                round+=1
            elif action=="ACT_A":
                #如果坚持答案是对的,那么更新MUST/MUST_NOT/两个answer_list
                maintain_have_answer(state,model_relation_answer,model_specific_answer)
                maintain_must_and_must_not(state,model_relation_answer,model_specific_answer)
                #记录最终答案
                ANSWER[model_relation_answer]=model_specific_answer

    answer_save = {
        "Event_Mention_A":row["Event_Mention_A"],
        "Trigger_A":row["Trigger_A"],
        "Trigger_Offset_A":row["Trigger_Offset_A"],
        "Event_Mention_B":row["Event_Mention_B"],
        "Trigger_B":row["Trigger_B"],
        "Trigger_Offset_B":row["Trigger_Offset_B"],
        "RELATION": ANSWER,
        "Background":{
            "Event_Mention_ID_A":row["Event_Mention_ID_A"],
            "Event_ID_A":row["Event_ID_A"],
            "Event_Mention_ID_B":row["Event_Mention_ID_B"],
            "Event_ID_B":row["Event_ID_B"],
            "Doc_ID":row['Doc_ID'],
            "Doc_Time":row['Doc_Time'],
            "Doc":row['Doc']
        }
    }
    #Lock保护文件写入
    with lock:
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(answer_save, ensure_ascii=False) + "\n")
    return 1  #处理成功返回1

def SDERE_main():
    df = pd.read_csv(path_input)
    num_workers = max(1, cpu_count() - 1)
    manager = Manager()
    lock = manager.Lock()
    pool = Pool(num_workers)
    #map异步处理....带tqdm的
    results = []
    for _ in tqdm(pool.imap_unordered(partial(each_row_processing, lock=lock), [row for idx, row in df.iterrows()]), total=len(df)):
        results.append(results)  #用来更新进度条的
    pool.close()
    pool.join()

if __name__ == "__main__":
    SDERE_main()