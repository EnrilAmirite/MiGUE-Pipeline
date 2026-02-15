from openai import OpenAI
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------


from PIPE_3_ERE_CROSS_DOC.CDERE_function_design import maintain_clean_global,maintain_must_and_must_not,maintain_have_answer
from PIPE_3_ERE_CROSS_DOC.CDERE_function_design import CDERE_prompt_intro,option_to_relation_name,option_to_specific_name
from PIPE_3_ERE_CROSS_DOC.CDERE_function_design import create_specific_question,create_relation_question,after_thinking
from PIPE_3_ERE_CROSS_DOC.CDERE_function_design import RELATION_ANSWER_LIST,SPECIFIC_ANSWER_LIST,MUST_NOT,MUST
import re




#========================================API配置======================================
HOLDAI_URL = "https://api.holdai.top/v1"
holdai_key_vip="......................."
'''
模型选择
这里感觉推理模型更合适一点
'''
#MODEL="gpt-4o-mini"
MODEL="gpt-4o-2024-08-06"

# 配置
client = OpenAI(
    api_key=holdai_key_vip, 
    base_url=HOLDAI_URL
)
#调用LLM函数
def call_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user", 
                "content": prompt
            }],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print("API出问题了TT", e)
        return ""
    


#==================================输入输出路径,输出名称=================================
path_input="DATA/CDERE_input/CDERE_event_pair_sample.csv"
output_folder="DATA/CDERE_output"
output_name=f"CDERE_1.0_test_{MODEL}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
path_output = os.path.join(output_folder, f"{output_name}_{timestamp}.jsonl")




#防止model不按要求输出...
OPTION_LIST=[]

#Event pair的文件长这样:
#Event_Mention_ID_A,Event_Mention_A,Event_ID_A,Trigger_A,Trigger_Offset_A,
#Event_Mention_ID_B,Event_Mention_B,Event_ID_B,Trigger_B,Trigger_Offset_B,
#Doc_ID_A,Doc_Time_A,Doc_A,Doc_ID_B,Doc_Time_B,Doc_B
def CDERE_main():
    df=pd.read_csv(path_input)
    with open(path_output,"a",encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(),total=len(df)):
            doc_a=row["Doc_A"]
            doc_b=row["Doc_B"]
            doc_time_a=row["Doc_Time_A"]
            doc_time_b=row["Doc_Time_B"]
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
            while len(SPECIFIC_ANSWER_LIST)!=4:
                #首先描述问题
                #相对于SDERE多给一些信息...
                OPTION_LIST=[]
                prompt=CDERE_prompt_intro(
                    doc_A=doc_a,
                    doc_B=doc_b,
                    event_mention_A=event_mention_a,
                    event_mention_B=event_mention_b,
                    doc_time_A=doc_time_a,
                    doc_time_B=doc_time_b,
                    trigger_A=trigger_a,
                    trigger_B=trigger_b,
                    offset_A=offset_a,
                    offset_B=offset_b
                    )
                model_relation_option=""
                model_specific_option=""
                #可以回答的schema_relation选项
                temp,OPTION_LIST=create_relation_question()
                prompt+=temp
                #模型返回选择的schema_relation 选项
                while model_relation_option not in OPTION_LIST:
                    model_relation_option = call_llm(prompt).strip()
                    print(model_relation_option)
                #schema_relation 选项转换成真实的relation
                model_relation_answer=option_to_relation_name(model_relation_option)
                print(prompt)
                print(OPTION_LIST)
                print(model_relation_answer)
                

                #可以选择的Specific_relation选项
                #选项一样,但是CDERE相对于SDERE要多给model一些信息
                OPTION_LIST=[]
                prompt=prompt=CDERE_prompt_intro(
                    doc_A=doc_a,
                    doc_B=doc_b,
                    event_mention_A=event_mention_a,
                    event_mention_B=event_mention_b,
                    doc_time_A=doc_time_a,
                    doc_time_B=doc_time_b,
                    trigger_A=trigger_a,
                    trigger_B=trigger_b,
                    offset_A=offset_a,
                    offset_B=offset_b
                    )
                temp,OPTION_LIST=create_specific_question(model_relation_answer)
                prompt+=temp
                #模型返回选择的Specific_relation选项
                while model_specific_option not in OPTION_LIST:
                    model_specific_option=call_llm(prompt).strip()
                #转换..
                model_specific_answer=option_to_specific_name(model_relation_answer,model_specific_option)
                print(prompt)
                print(model_specific_answer)

                #后验环节,这个和SDERE完全一样的
                action=""
                while action not in ["ACT_A","ACT_B","ACT_C"]:
                    prompt_thinking=after_thinking(model_relation_answer,model_specific_answer)
                    if prompt_thinking=="":
                        action="ACT_A"
                    else: 
                        prompt=after_thinking(model_relation_answer,model_specific_answer)
                        prompt+=prompt_thinking   
                        action=call_llm(prompt)
                    print(prompt)
                    print(action)
                    #重做这轮不写入答案即可
                    if action=="ACT_B":
                        round+=1
                    #全部重做即清空现在所有的全局变量
                    elif action=="ACT_C":
                        maintain_clean_global()
                        round+=1
                    elif action=="ACT_A":
                        #如果坚持答案是对的,那么更新MUST/MUST_NOT/两个answer_list
                        maintain_have_answer(model_relation_answer,model_specific_answer)
                        maintain_must_and_must_not(model_relation_answer,model_specific_answer)
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
                    "Doc_ID_A":row["Doc_ID_A"],
                    "Doc_A":row["Doc_A"],
                    "Doc_Time_A":row["Doc_Time_A"],
                    "Doc_ID_B":row["Doc_ID_B"],
                    "Doc_B":row["Doc_B"],
                    "Doc_Time_B":row["Doc_Time_B"],
                }
            }
            f.write(json.dumps(answer_save, ensure_ascii=False) + "\n")
            #清理
            maintain_clean_global()

if __name__ == "__main__":
    CDERE_main()