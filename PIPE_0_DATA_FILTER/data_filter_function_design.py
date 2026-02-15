import json
import csv
import deepl
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import api_setting



#---------------------------------------------------------------
#删除csv进程标识列
#---------------------------------------------------------------
def  drop_csv_column(path_input,col_name):
    df = pd.read_csv(path_input)
    df = df.drop(columns=[col_name])
    df.to_csv(path_input, index=False)

#---------------------------------------------------------------
#保留csv列==xxx的行
#---------------------------------------------------------------
def  keep_csv_row(path_input,col_name,col_cont):
    df = pd.read_csv(path_input)
    df = df[df[col_name] == col_cont]
    df.to_csv(path_input, index=False)

#---------------------------------------------------------------
#调用deepl进行语料翻译
#---------------------------------------------------------------
def translate_en2ch(path_input,path_output,api_key):
    translator = deepl.Translator(api_key)
    columns = [
        "id",
        "doc_id","created_at","content",
        "raw_en_content","title"
    ]
    # 初始化输出文件
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output, index=False, encoding="utf-8"
        )
    df=pd.read_csv(path_input)
    for idx,row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="正在英译中......."
    ):
        text_en=row['raw_en_content']
        try:
            text_zh=translator.translate_text(
                text_en,
                source_lang="EN",
                target_lang="ZH"
            ).text
        except Exception as e:
            print(f"翻译出了点问题...line {idx}: {e}")
            text_zh = ""
        save_answer={
            "id": idx,
            "doc_id":row['doc_id'],
            "created_at":"2022",
            "content":text_zh,
        }
        pd.DataFrame([save_answer]).to_csv(
                path_output,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8"
            )



#---------------------------------------------------------------
#评估文本质量的prompt
#type有"Event""Text""Theme"
#翻译的prompt
#type为"Trans"
#---------------------------------------------------------------
#path="PIPE_0_DATA_FILTER/df_rule.json"
def create_eval_trans_prompt(path_prompt,type,raw_text):
    with open (path_prompt,'r',encoding='utf-8') as f:
        prompt_file=json.load(f)
    prompt_type=prompt_file[type]
    prompt=prompt_type['Intro']+prompt_type['Rules']+prompt_type['Give_Doc']
    prompt+=f"{raw_text}\n"
    prompt+=prompt_type['Output_Control']
    return prompt

#print(create_eval_trans_prompt(path,'Event',11111))





#---------------------------------------------------------------
#调用openai的llm
#---------------------------------------------------------------

#调用LLM函数:评估版
def call_openai_llm_eval(model,url,key,prompt):
    client = OpenAI(
    api_key=key, 
    base_url=url
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "system", 
                "content": "You are an expert in the field of text analysis. Please analyze the text quality."
            },
                {
                "role": "user", 
                "content": prompt
            }
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print("API出问题了TT", e)
        return ""

#翻译函数
def call_openai_llm_trans(model,url,key,prompt):
    client = OpenAI(
    api_key=key, 
    base_url=url
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "system", 
                "content": "You are an expert in the field of text analysis. Please analyze the text quality."
            },
                {
                "role": "user", 
                "content": prompt
            }
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print("API出问题了TT", e)
        return ""
    

#---------------------------------------------------------------
#保存格式统一的源文档(主要是TCE的json)为csv
#---------------------------------------------------------------
def raw_doc_json2csv(path_input,path_output):
    columns = [
        "id",
        "doc_id","created_at","content",
        "raw_en_content","title"
    ]
    # 初始化输出文件
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output, index=False, encoding="utf-8"
        )
    with open(path_input,'r',encoding='utf-8') as f:
        raw_corpus=json.load(f)
    row_id=1
    for doc_id, doc_items in tqdm(
        raw_corpus.items(),
        total=len(raw_corpus),
        desc="正在保存为统一csv格式"
    ):
        text_en_list = doc_items["Text"]
        text_en = "。".join(text_en_list)
        title=doc_items["Title"]
        save_answer={
            "id": row_id,
            "doc_id":doc_id,
            "created_at":"2022",
            "content":"",
            "raw_en_content":text_en,
            "title":title
        }
        pd.DataFrame([save_answer]).to_csv(
                path_output,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8"
            )
        row_id+=1
        #if row_id==200:
            #break


def raw_doc_json2csv_gemini(path_input, path_output):
    columns = ["id", "doc_id", "created_at", "content", "raw_en_content", "title"]
    with open(path_input, 'r', encoding='utf-8') as f:
        raw_corpus = json.load(f)
    #使用 utf-8-sig 是为了让 Excel 能直接识别希伯来文和特殊符号
    with open(path_output, 'w', encoding='utf-8-sig', newline='') as f_out:
        # quoting=csv.QUOTE_ALL 会给每个字段强制加上双引号，这是处理长文本最安全的方法
        writer = csv.DictWriter(f_out, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        row_id = 1
        for doc_id, doc_items in tqdm(raw_corpus.items(), desc="正在转换数据"):            
            # 处理文本列表：建议用 \n 分隔段落，保留原文结构
            text_list = doc_items.get("Text", [])
            # 这里的 \n 会被 csv.QUOTE_ALL 安全地包裹在单元格内，不会导致破坏结构
            text_en = " ".join(text_list) 
            title = doc_items.get("Title", "")
            # 构造行数据
            row = {
                "id": row_id,
                "doc_id": doc_id,
                "created_at": "2022",
                "content": "",  # 保持空字符串
                "raw_en_content": text_en,
                "title": title
            }            
            writer.writerow(row)
            row_id += 1
    print(f"\n转换完成！文件已保存至: {path_output}")



#---------------------------------------------------------------
#调用llm进行评估
#---------------------------------------------------------------
def eval_raw_doc(input_path,output_path,prompt_path,url,key,model):
    df = pd.read_csv(
    input_path, 
    encoding='utf-8-sig', 
    engine='python', 
    on_bad_lines='warn'
    )
    #processed主要用于断点续传...
    if "processed" not in df.columns:
        df["processed"]=0
    if "vote_Text" not in df.columns:
        df["vote_Text"]=""
    if "vote_Theme" not in df.columns:
        df["vote_Theme"]=""
    if "vote_Event" not in df.columns:
        df["vote_Event"]=""
    if "vote_Final" not in df.columns:
        df["vote_Final"]=""
    total = (df["processed"] == 0).sum()
    for idx, row in tqdm(
            df[df["processed"]==0].iterrows(),
            total=total,
            desc="正在评估原始文档..."
        ):
        raw_doc=row['raw_en_content']
        vote_num=0
        for eval_type in ["Text","Theme","Event"]:
            prompt=create_eval_trans_prompt(
                path_prompt=prompt_path,
                type=eval_type,
                raw_text=raw_doc
            )
            answer=call_openai_llm_eval(
                prompt=prompt,
                url=url,
                key=key,
                model=model
            )
            if answer=='Pass':
                vote_num+=1
            df.loc[idx,f"vote_{eval_type}"]=answer
        if vote_num>=3:
            df.loc[idx,"vote_Final"]='Pass'
        else:
            df.loc[idx,"vote_Final"]='Fail'
        df.loc[idx,f"processed"]=1
        df.to_csv(output_path, index=False)

#并发版本
def eval_raw_doc_concurrent(input_path, output_path, prompt_path, url, key, model, max_workers):
    # 1. 读取并初始化数据
    df = pd.read_csv(input_path, encoding='utf-8-sig', engine='python', on_bad_lines='warn')
    for col in ["vote_Text", "vote_Theme", "vote_Event", "vote_Final"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)
    if "processed" not in df.columns:
        df["processed"] = 0
    # 筛选待处理的任务
    todo_indices = df[df["processed"] == 0].index.tolist()
    total_tasks = len(todo_indices)
    if total_tasks == 0:
        print("没有需要处理的数据。")
        return
    # 创建锁，确保多线程修改 DataFrame 时的安全
    df_lock = threading.Lock()
    # 2. 定义单个任务的逻辑
    def process_single_row(idx):
        try:
            raw_doc = df.loc[idx, 'raw_en_content']
            vote_results = {}
            vote_num = 0
            
            for eval_type in ["Text", "Theme", "Event"]:
                prompt = create_eval_trans_prompt(
                    path_prompt=prompt_path,
                    type=eval_type,
                    raw_text=raw_doc
                )
                answer = call_openai_llm_eval(
                    prompt=prompt,
                    url=url,
                    key=key,
                    model=model
                )
                vote_results[f"vote_{eval_type}"] = answer
                if answer == 'Pass':
                    vote_num += 1
            
            final_vote = 'Pass' if vote_num >= 3 else 'Fail'
            
            # 安全写回
            with df_lock:
                for col, val in vote_results.items():
                    df.loc[idx, col] = val
                df.loc[idx, "vote_Final"] = final_vote
                df.loc[idx, "processed"] = 1
                return True
        except Exception as e:
            print(f"\n[Error] 索引 {idx} 处理失败: {e}")
            return False

    # 3. 启动线程池并正确初始化 pbar
    print(f"开始并发评估，任务总数: {total_tasks}, 线程数: {max_workers}")
    
    # 使用 with 语句确保 pbar 在结束后自动关闭
    with tqdm(total=total_tasks, desc="正在并发评估原始文档...") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(process_single_row, idx): idx for idx in todo_indices}
            
            for i, future in enumerate(as_completed(futures)):
                # 更新进度条 (现在 pbar 已经定义好了)
                pbar.update(1)
                
                # 每 10 次成功或完成后保存一次，防止意外断电丢失进度
                if (i + 1) % 10 == 0:
                    with df_lock:
                        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    # 4. 最终保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n所有任务已完成，结果保存至: {output_path}")


#---------------------------------------------------------------
#调用llm进行翻译
#---------------------------------------------------------------
def trans_raw_doc(input_path,output_path,prompt_path,url,key,model):
    df = pd.read_csv(
    input_path, 
    encoding='utf-8-sig', 
    engine='python', 
    on_bad_lines='warn'
    )
    columns = ["id", "doc_id", "created_at", "content"]
    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
    #processed主要用于断点续传...
    if "processed" not in df.columns:
        df["processed"]=0
    total = (df["processed"]==0).sum()
    for idx, row in tqdm(
            df[df["processed"]==0].iterrows(),
            total=total,
            desc="正在翻译原始文档..."
        ):
        raw_doc=row['raw_en_content']
        eval_type='Trans'
        prompt=create_eval_trans_prompt(
            path_prompt=prompt_path,
            type=eval_type,
            raw_text=raw_doc
        )
        trans_answer=call_openai_llm_trans(
            prompt=prompt,
            url=url,
            key=key,
            model=model
        )
        save_answer={
            "id": row['id'],
            "doc_id":row['doc_id'],
            "created_at":"2022",
            "content":trans_answer,
        }
        pd.DataFrame([save_answer]).to_csv(
                output_path,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8"
            )

def trans_raw_doc_concurrent(input_path, output_path, prompt_path, url, key, model, max_workers):
    # 1. 读入原始文件
    df = pd.read_csv(input_path, encoding='utf-8-sig', engine='python')
    # 2. 如果输出文件不存在，先创建并写入表头
    columns = ["id", "doc_id", "created_at", "content"]
    if not os.path.exists(output_path):
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
    # 初始化进度管理
    if "processed" not in df.columns:
        df["processed"] = 0
    todo_df = df[df["processed"] == 0]
    # 【关键】创建一个互斥锁
    write_lock = threading.Lock()
    # 3. 定义单个处理函数（包含写入逻辑）
    def single_task(row):
        try:
            raw_doc=row['raw_en_content']
            eval_type='Trans'
            prompt=create_eval_trans_prompt(
            path_prompt=prompt_path,
            type=eval_type,
            raw_text=raw_doc
        )
            trans_answer = call_openai_llm_trans(
                model=model,
                url=url,
                key=key,
                prompt=prompt
            )
            # 构造要写入的数据行
            save_data = {
                "id": row['id'],
                "doc_id": row['doc_id'],
                "created_at": "2022",
                "content": trans_answer
            }         
            # 【核心步骤】拿到数据立刻写入，但必须加锁排队
            with write_lock:
                with open(output_path, 'a', encoding='utf-8-sig', newline='') as f:
                    # 使用 QUOTE_ALL 确保翻译内容中的引号和换行不会破坏 CSV 结构
                    writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
                    writer.writerow(save_data) 
            return True
        except Exception as e:
            print(f"Error on {row['doc_id']}: {e}")
            return False
    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_task, row) for _, row in todo_df.iterrows()]    
        for _ in tqdm(as_completed(futures), total=len(futures), desc="并发翻译中"):
            pass