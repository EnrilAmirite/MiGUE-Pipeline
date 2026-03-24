import json
import jsonlines
from openai import OpenAI
import os
from datetime import datetime
from tqdm import tqdm
import csv
from pydantic import BaseModel
import pandas as pd
from ED_function_design import create_example, create_reflection, create_task_intro, list_to_str
import re #正则
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CONFIG.config import GLOBAL_ED_END_ROW_ID,GLOBAL_ED_START_ROW_ID,CROPUS
'''
抽取trigger words+多轮self-reflection检查
'''
#============API配置=============
#模型选择
MODEL="gpt-4o-mini"
#MODEL="gpt-4o-2024-08-06"
#MODEL="gpt-5.2-pro"
#MODEL="gpt-5.1"
#MODEL="deepseek-reasoner"

def api_setting(model_name):
    model_url=""
    model_key=""
    if model_name in ["gpt-4o-mini","gpt-4o-2024-08-06","gpt-5.2-pro","gpt-5.1"]:
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
#一些路径和名称
#-------------------------------------------------------------------------
id_first=GLOBAL_ED_START_ROW_ID
id_last=GLOBAL_ED_END_ROW_ID
path_input=f"DATA/ED_input/event_mention_{CROPUS}_{id_first}to{id_last}.csv"
folder_output="DATA/ED_output" 
output_name="ED111"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
path_output= os.path.join(folder_output, f"{output_name}_{CROPUS}_{id_first}to{id_last}.csv")

#-------------------------------------------------------------------------
#结构化输出
#-------------------------------------------------------------------------
'''
我这里写了结构化输出
'''
class trigger_output_format(BaseModel):
    triggers:list[str]

def call_llm(prompt,model=MODEL):#这里只传一个model...不区分act和reflection
    response=client.responses.parse(
        model=model,
        input=[
            {
                "role": "system", 
                "content": "You are an expert in the field of event detection. Extract the event triggers."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=trigger_output_format,
    )
    return response.output_parsed
#ps:这里的返回是一个Pydantic对象,例子:triggers=['发布', '发射', '打击']
#真正的答案是从这个结构化对象里,取出名为''triggers''的字段
#answer.triggers

#-------------------------------------------------------------------------
#主程序
#-------------------------------------------------------------------------
def ED_main(path_input,path_output):
    columns=["Doc_ID","Event_Mention_ID","Event_Mention","Raw_Trigger_List","Missing_Check","Named_Entity","Narrative","No_Occurrence","Assumption","Abstraction","Negated_Event","Final_Trigger_List"]
    df=pd.read_csv(path_input)
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        doc_id=row["Doc_ID"]
        event_mention_id=row["Event_Mention_ID"]
        event_mention=row["Event_Mention"]
        #置空环节
        answer=[]
        add_trigger_list=[]
        delete_trigger_list=[]
        prompt=create_task_intro(event_mention,event_mention_id)
        #print(prompt)
        #第一次直接回答
        first_trigger_list=call_llm(prompt).triggers#返回的是个list
        first_trigger_str=list_to_str(first_trigger_list)
        answer.append(first_trigger_str)
        #print(answer[0])
        #检查遗漏环节missing check
        prompt=create_reflection(event_mention,event_mention_id,answer[0],0)
        add_trigger_list=call_llm(prompt).triggers
        add_trigger_str=list_to_str(add_trigger_list)
        for trigger in add_trigger_list:#如果有增加,加到first_trigger_list里
            if trigger not in first_trigger_list:
                first_trigger_list.extend(trigger)
        answer.append(add_trigger_str)
        #print(f"增加的trigger:{add_trigger_str}")
        #print(f"最终的full trigger list:{first_trigger_list}")
        #正儿八经的反思错误环节
        #自选反思的内容...
        for round in [1,2,3,4,5,6]:
            prompt=create_reflection(event_mention,event_mention_id,first_trigger_list,round)
            #print(f"当前反思轮数!{round}\n{prompt}")
            trigger_list=call_llm(prompt).triggers
            trigger_str=list_to_str(trigger_list)
            #print(f"当前的回答:{trigger_str}")
            delete_trigger_list.extend(trigger_list)
            answer.append(trigger_str)
        #结算时刻...
        final_trigger_list=[
            x for x in first_trigger_list if x not in delete_trigger_list
        ]
        final_trigger_str=list_to_str(final_trigger_list)
        save_answer={
            "Doc_ID":doc_id,
            "Event_Mention_ID":event_mention_id,
            "Event_Mention":event_mention,
            "Raw_Trigger_List":answer[0],
            "Missing_Check":answer[1],
            "Named_Entity":answer[2],
            "Narrative":answer[3],
            "No_Occurrence":answer[4],
            "Assumption":answer[5],
            "Abstraction":answer[6],
            "Negated_Event":answer[7],
            "Final_Trigger_List":final_trigger_str
        }
        pd.DataFrame([save_answer]).to_csv(
            path_output,
            mode='a',#追加写.........
            header=False, #不用写表头了
            index=False,
            encoding='utf-8'
        )



#启动
if __name__ == "__main__":
    ED_main(path_input,path_output)
    print("完成啦,已经保存到",path_output)
