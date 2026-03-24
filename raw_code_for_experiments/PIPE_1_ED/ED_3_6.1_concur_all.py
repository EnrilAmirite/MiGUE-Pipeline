import json
import jsonlines
from openai import OpenAI
import os
from datetime import datetime
from tqdm import tqdm
import csv
from pydantic import BaseModel
import pandas as pd
from ED_function_design import create_example, create_reflection, create_task_intro, list_to_str,str_to_list
import re #正则
from multiprocessing import Pool, cpu_count
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CONFIG.config import GLOBAL_ED_END_ROW_ID,GLOBAL_ED_START_ROW_ID,CROPUS
suffix=f"{CROPUS}_{GLOBAL_ED_START_ROW_ID}to{GLOBAL_ED_END_ROW_ID}"

'''
抽取trigger words+多轮self-reflection检查
'''
#---------------------------------------------------------------------
#api配置...
#---------------------------------------------------------------------
#模型选择
#MODEL="gpt-4o-mini"
#MODEL="gpt-4o-2024-08-06"
#MODEL="gpt-5.2-pro"
#MODEL="gpt-5.1"
MODEL="deepseek-reasoner"
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
#一些路径和名称
#-------------------------------------------------------------------------
path_input=f"DATA/ED_input/event_mention_{suffix}.csv"
folder_output="DATA/ED_output" 
output_name="ED"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
path_output= os.path.join(folder_output, f"{output_name}_{suffix}.csv")

#-------------------------------------------------------------------------
#结构化输出
#-------------------------------------------------------------------------
def call_llm(prompt, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": "You are an expert in the field of event detection. Extract the event triggers."
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
#-------------------------------------------------------------------------
#主程序
#-------------------------------------------------------------------------

def process_one_row(row):
    try:
        doc_id = row["Doc_ID"]
        event_mention_id = row["Event_Mention_ID"]
        event_mention = row["Event_Mention"]

        answer = []
        delete_trigger_list = []

        # 第一次抽取
        prompt = create_task_intro(event_mention, event_mention_id)
        first_trigger_str = call_llm(prompt)
        first_trigger_list = str_to_list(first_trigger_str)
        answer.append(first_trigger_str)

        # Missing check
        prompt = create_reflection(event_mention, event_mention_id, answer[0], 0)
        add_trigger_str = call_llm(prompt)
        add_trigger_list = list_to_str(add_trigger_str)

        for trigger in add_trigger_list:
            if trigger not in first_trigger_list:
                first_trigger_list.append(trigger)

        answer.append(add_trigger_str)

        # 反思环节
        for round_id in [1, 2, 3, 4, 5, 6]:
            prompt = create_reflection(
                event_mention,
                event_mention_id,
                first_trigger_list,
                round_id
            )
            trigger_str = call_llm(prompt)
            trigger_list = list_to_str(trigger_str)

            delete_trigger_list.extend(trigger_list)
            answer.append(trigger_str)

        # Final triggers
        final_trigger_list = [
            x for x in first_trigger_list if x not in delete_trigger_list
        ]
        final_trigger_str = list_to_str(final_trigger_list)

        save_answer = {
            "Doc_ID": doc_id,
            "Event_Mention_ID": event_mention_id,
            "Event_Mention": event_mention,
            "Raw_Trigger_List": answer[0],
            "Missing_Check": answer[1],
            "Named_Entity": answer[2],
            "Narrative": answer[3],
            "No_Occurrence": answer[4],
            "Assumption": answer[5],
            "Abstraction": answer[6],
            "Negated_Event": answer[7],
            "Final_Trigger_List": final_trigger_str
        }

        return save_answer

    except Exception as e:
        print(e)
        return {
            "Doc_ID": row.get("Doc_ID"),
            "Event_Mention_ID": row.get("Event_Mention_ID"),
            "Event_Mention": row.get("Event_Mention"),
            "Raw_Trigger_List": "ERROR",
            "Missing_Check": "ERROR",
            "Named_Entity": "ERROR",
            "Narrative": "ERROR",
            "No_Occurrence": "ERROR",
            "Assumption": "ERROR",
            "Abstraction": "ERROR",
            "Negated_Event": "ERROR",
            "Final_Trigger_List": f"ERROR: {type(e).__name__}"
        }

def ED_main(path_input, path_output, num_workers=8):
    columns = [
        "Doc_ID", "Event_Mention_ID", "Event_Mention",
        "Raw_Trigger_List", "Missing_Check", "Named_Entity",
        "Narrative", "No_Occurrence", "Assumption",
        "Abstraction", "Negated_Event", "Final_Trigger_List"
    ]

    df = pd.read_csv(path_input)

    # 初始化输出文件
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output, index=False, encoding="utf-8"
        )

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    with Pool(processes=num_workers) as pool:
        for save_answer in tqdm(
            pool.imap_unordered(process_one_row, df.to_dict("records")),
            total=len(df)
        ):
            pd.DataFrame([save_answer]).to_csv(
                path_output,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8"
            )



#启动
if __name__ == "__main__":
    ED_main(path_input,path_output)
    print("完成啦,已经保存到",path_output)
