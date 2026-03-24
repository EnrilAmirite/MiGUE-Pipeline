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
from CONFIG.config import GLOBAL_CDERE_SUFFIX
suffix=GLOBAL_CDERE_SUFFIX
#----------------------------------------------------------------------------------------
#输入输出路径,输出名称
#----------------------------------------------------------------------------------------
#规则和选项
path_option="PIPE_3_ERE_CROSS_DOC/options.json"
path_rule="PIPE_3_ERE_CROSS_DOC/rules.json"
path_description="PIPE_3_ERE_CROSS_DOC/relation_description.json"

#embedding
path_event_mention_normal=f"PIPE_3_ERE_CROSS_DOC/CDERE_embedding/EVENT_NORMAL_INDEX_{suffix}.index"#顺序存储event_mention_embedding
path_event_mention_id=f"PIPE_3_ERE_CROSS_DOC/CDERE_embedding/EVENT_ID_{suffix}.pkl"#顺序存储event_mention_id
path_event_mention_id_to_idx=f"PIPE_3_ERE_CROSS_DOC/CDERE_embedding/EVENT_ID_TO_IDX_{suffix}.pkl"#event_mention_id到idx的索引
path_event_mention_id_to_embedding=f"PIPE_3_ERE_CROSS_DOC/CDERE_embedding/EVENT_ID_TO_EMBEDDING_{suffix}.pkl"#event_mention_id到embedding的索引



#----------------------------------------------------------------------------------------
#全局变量
#----------------------------------------------------------------------------------------
#四种关系的问题
QUESTIONS=["TEMPORAL","CAUSAL","SUBEVENT","COREFERENCE"]
#已经回答过的关系
RELATION_ANSWER_LIST=[]
#目前的relation回答
SPECIFIC_ANSWER_LIST=[]
#一定不能选择的关系
#维护的时候用extend
MUST_NOT={
    "TEMPORAL":[],
    "CAUSAL":[],
    "SUBEVENT":[],
    "COREFERENCE":[]
}
#只有一个选项的关系
#维护的时候也用extend
MUST={
    "TEMPORAL":[],
    "CAUSAL":[],
    "SUBEVENT":[],
    "COREFERENCE":[]    
}
OPTION_LIST=[]

#检索需要的全局变量
SIMILAR_TOP_K=15
SIMILAR_THRESHOLD=0.85


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ERE函数部分
#SDERE/CDERE基本通用
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# 维护函数
# 每次两个事件的关系全部找完后清空所有的全局变量
#对于可变全局对象(list/dict/set),用.clear()是直接改变对象本身的值
#这种方法只对可变对象有效，对不可变对象(int/str/tuple)不起作用
#----------------------------------------------------------------------------------------
def maintain_clean_global():
    # 四种关系的问题
    QUESTIONS = ["TEMPORAL", "CAUSAL", "SUBEVENT", "COREFERENCE"]
    # 已经回答过的关系
    RELATION_ANSWER_LIST.clear()
    # 目前的relation回答
    SPECIFIC_ANSWER_LIST.clear()
    print(SPECIFIC_ANSWER_LIST)
    # 一定不能选择的关系
    # 维护的时候用extend
    # 清空MUST_NOT和MUST中每个键对应的列表
    for key in MUST_NOT:
        MUST_NOT[key].clear()
    # 只有一个选项的关系
    # 维护的时候也用extend
    # 清空MUST_NOT和MUST中每个键对应的列表
    for key in MUST:
        MUST[key].clear()
    return

#----------------------------------------------------------------------------------------
#把关系选项，变成schema level relation
#比如:把REL_1变成TEMPORAL
#----------------------------------------------------------------------------------------
def option_to_relation_name(model_option):
    with open(path_option,"r",encoding="utf-8")as f:
        options=json.load(f)
    mapping=options["RELATION_CHOOSE"]
    relation_name=mapping[model_option]["relation"]
    return relation_name



#----------------------------------------------------------------------------------------
#把关系选项，变成具体关系名称
#比如:把TEMP_1变成BEFORE,TEMP_2变成AFTER
#----------------------------------------------------------------------------------------
def option_to_specific_name(model_relation,model_option):
    with open(path_option,"r",encoding="utf-8")as f:
        options=json.load(f)    
    mapping=options[model_relation]
    specific_name=mapping[model_option]["name"]
    return specific_name




#----------------------------------------------------------------------------------------
#根据model的回答维护MUST_NOT表格和MUST表格
#需要用rules.json维护MUST_NOT,减去里面提到的关系
#用rules.json维护MUST,加上must里的关系
#MUST_NOT,MUST维护的时候用extend(虽然MUST好像没有extend的必要...)
#----------------------------------------------------------------------------------------
def maintain_must_and_must_not(relation_answer,specific_answer):
    with open(path_rule,"r",encoding="utf-8")as f:
        rules=json.load(f)  
    delete_specific=rules[relation_answer][specific_answer]["no"]
    add_specific=rules[relation_answer][specific_answer]["must"]
    #维护四种关系里面的禁选项
    MUST_NOT["CAUSAL"].extend(delete_specific["CAUSAL"])
    MUST_NOT["COREFERENCE"].extend(delete_specific["COREFERENCE"])
    MUST_NOT["SUBEVENT"].extend(delete_specific["SUBEVENT"])
    MUST_NOT["TEMPORAL"].extend(delete_specific["TEMPORAL"])
    #维护四种关系里面的必选项
    MUST["CAUSAL"].extend(add_specific["CAUSAL"])
    MUST["COREFERENCE"].extend(add_specific["COREFERENCE"])
    MUST["SUBEVENT"].extend(add_specific["SUBEVENT"])
    MUST["TEMPORAL"].extend(add_specific["TEMPORAL"])
    return 

#----------------------------------------------------------------------------------------
#维护已经回答的问题列表
#----------------------------------------------------------------------------------------
def maintain_have_answer(relation_answer,specific_answer):
    #relation
    RELATION_ANSWER_LIST.append(relation_answer)
    #specific answer
    SPECIFIC_ANSWER_LIST.append(specific_answer)


#----------------------------------------------------------------------------------------
#生成specific的问题
#----------------------------------------------------------------------------------------
'''
比如,前面模型才选择了时间关系:BEFORE,这次想回答因果关系
那么输出应该是:
CAU_1:A导致B(CAUSE):
即事件A的发生,事件B一定会发生
CAU_2:A是B的前提(PRECONDITION):
即如果事件A不发生,则事件B一定不会发生
CAU_5:A与B无因果关系(NO_CAUSAL):
即事件A的发生与事件B的发生之间没有任何因果关系
'''
def create_specific_question(relation_answer):
    with open(path_option,"r",encoding="utf-8")as f:
        options=json.load(f)  
    specific_options=options[relation_answer]
    prompt_options=f"您想要首先回答事件间的{relation_answer}关系,目前可选的选项有:\n"
    #删除逻辑冲突的option
    filtered_options={
        key:v for key,v in specific_options.items()
        if v["name"] not in MUST_NOT[relation_answer]
    }
    prompt_options+="\n".join([
        f"{key}:{item["text"]}"
        for key,item in filtered_options.items()
    ])
    prompt_options+="\n请只输出您选择的选项前面的序号,即"
    option_list=",".join([
        f"{key}"
        for key in filtered_options
    ])
    prompt_options+=option_list
    prompt_options+="\n请一定按格式输出序号,不要给我其他任何无关的字符,或者推理过程"
    OPTION_LIST = [key for key in filtered_options]
    return prompt_options,OPTION_LIST


#----------------------------------------------------------------------------------------
#task intro-prompt
#----------------------------------------------------------------------------------------

def CDERE_prompt_intro(doc_A,doc_B,doc_time_A,doc_time_B,event_mention_A,event_mention_B,trigger_A,trigger_B,offset_A,offset_B):
    path_ED_prompt="PIPE_1_ED/ED_prompt.json"
    prompt_intro_event=""
    with open(path_ED_prompt,'r',encoding='utf-8') as f:
        prompt_file=json.load(f)
        task_intro=prompt_file["Task_Intro"]
        for key,v in task_intro.items():
            if key!="Given_Event_mention":
                prompt_intro_event+=v
    prompt_intro_ere="事件关系抽取(Events Relation Extract,ERE)即指:判断两个触发词对应的两个事件在'时序','因果','子事件','共指'四个层面的关系\n"
    prompt_intro_doc_and_event=f"""
请根据上述定义,结合原文,在下面的事件对中,判断事件A和事件B在'时序','因果','子事件','共指'四个层面的关系.
事件A的原文档A:{doc_A}
原文档A的发出时间:{doc_time_A}
事件A的事件提及:{event_mention_A}
事件A的触发词:{trigger_A}
事件A的触发词在事件提及中的位置:{offset_A}
事件B的原文档B:{doc_B}
原文档B的发出时间:{doc_time_B}
事件B的事件提及:{event_mention_B}
事件B的触发词:{trigger_B}
事件B的触发词在事件提及中的位置:{offset_B}
\n
"""
    prompt=prompt_intro_ere+prompt_intro_doc_and_event
    return prompt

#----------------------------------------------------------------------------------------
#生成relation问题
#----------------------------------------------------------------------------------------
'''
比如:model前面已经选择了temporal进行回答
那么现在的选项就应该是:
REL_B: 因果关系(Causal)
REL_C: 子事件关系(Subevent)
REL_D: 共指关系(Coreference)
'''
def create_relation_question():
    #读读读options.json...........
    with open(path_option, "r", encoding="utf-8") as f:
        options= json.load(f)
    relation_choose = options["RELATION_CHOOSE"] # 所有关系选项
    prompt_options=""
    if len(RELATION_ANSWER_LIST)!=0:
        prompt_options+=f"您前面已经回答了{RELATION_ANSWER_LIST}这些关系\n"
    prompt_options+="现在还没有回答的事件关系有:\n"
    #删除已经回答过的relation
    filtered_options = {
        key: v for key, v in relation_choose.items()
        if v["relation"] not in RELATION_ANSWER_LIST
    }
    #prompt拼接
    prompt_options+="\n".join([
        f"{key}: {item['text']}"
        for key, item in filtered_options.items()
    ])
    prompt_options+="\n请从上述选项中选择一个你认为当前最容易判断的关系,注意只输出您选择的选项前面的序号,即"
    option_list=",".join([
        f"{key}"
        for key in filtered_options
    ])
    prompt_options+=option_list
    prompt_options+="\n请一定按格式输出序号,不要给我其他任何无关的字符,或者推理过程"
    OPTION_LIST = [key for key in filtered_options]
    return prompt_options,OPTION_LIST


#----------------------------------------------------------------------------------------
#后验步骤(1):选择下一步的操作ACT_1 ACT_2 ACT_3
#主要思路:输出MUST_NOT表格,MUST表格,让model思考自己是否选错了
#有一种情况是,现在的rules约束不会有改变,所以用这个函数的时候需要判断一下返回的prompt是不是为空
#----------------------------------------------------------------------------------------
def after_thinking(relation_answer,specific_answer):
    with open(path_option,"r",encoding="utf-8") as f:
        options=json.load(f)
    with open(path_rule,"r",encoding="utf-8") as f:
        rules=json.load(f)
    with open(path_description,"r",encoding="utf-8") as f:
        describe=json.load(f)
    prompt_thinking=f"您当前选择回答的事件关系为:{relation_answer}\n您当前的答案是:{specific_answer}\n"
    #前几步(如果有的话)没有异议的选择就不在这里重复说了...节约tokens
    #建议确定了再维护MUST和MUST_NOT,所以后验应该在维护之前
    must=rules[relation_answer][specific_answer]["must"]
    must_not=rules[relation_answer][specific_answer]["no"]
    prompt=""
    must_list=[]
    #一定有的
    #如果must全是空的就跳过了
    if all(len(v)==0 for key,v in must.items()):
        prompt+=""
    else: 
        prompt+=f"这意味着您认为两个事件中一定有如下关系:\n"
        #拿出must有内容的几项
        thinking_must={
            key:v for key,v in must.items()
            if len(v)!=0
        }
        for key,v_list in thinking_must.items():#这里记得后面是个列表
            must_list.extend(key)
            prompt+=f"对于{key}层面,您认为肯定存在关系:"
            for v in v_list:
                des=describe[key][v]["text"]
                prompt+=f"{des}\n"
    #一定不能选的:
    if all(len(v)==0 for key,v in must_not.items()):
        prompt+=""
    else:
        prompt+="这同样意味着,您认为这两个事件一定不存在如下关系:\n"
        #拿出must_not有内容的几项
        #如果must里面本来就有了,就不输出了(不然超级多...)
        thinking_must_not={
            key:v for key,v in must_not.items()
            if len(v)!=0 and key not in must_list
        }
        for key,v_list in thinking_must_not.items():
            prompt+=f"对于{key}层面,您认为一定不存在的关系有:\n"
            for v in v_list:
                des=describe[key][v]["text"]
                prompt+=f"{des}\n"
    if prompt!="":
        prompt+="您认为自己的选择是正确的吗?请思考后选择您想进行的操作:\n"
        prompt+="\n".join([
            f"{k}:{v["text"]}"
            for k,v in options["ACTION_CHOOSE"].items()
        ])
        #输出控制
        prompt+="\n请只输出对应选项前面的序号,即:"
        prompt+=",".join([
            f"{k}"
            for k,v in options["ACTION_CHOOSE"].items()
        ])
        prompt_thinking+=prompt
        prompt_thinking+="\n请一定按格式输出序号,不要给我其他任何无关的字符,或者推理过程"
    else: return ""
    return prompt_thinking





#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#信息检索函数
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#CDERE 信息检索1: 语义最相似的event mention(不在同个文档中)
#这里用了FAISS
#----------------------------------------------------------------------------------------
def find_semantic_similarity(event_mention_id,top_k,similarity_threshold):
    with open(path_event_mention_id,"rb") as f:
        all_event_id_list=pickle.load(f)
    with open(path_event_mention_id_to_embedding,"rb") as f:
        event_id_to_embedding=pickle.load(f)
    all_event_embedding=faiss.read_index(path_event_mention_normal)
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




#----------------------------------------------------------------------------------------
#SDERE/CDERE 信息检索2: retrieve函数 
#用doc_id和event_id检索到具体的doc和event mention
#----------------------------------------------------------------------------------------
def retrieve_doc(path_input,doc_id):
    df_doc=pd.read_csv(path_input)
    df_doc.set_index('doc_id')
    row=df_doc.loc[doc_id]
    return row['content'],row['create_at']#返回原文和发表时间

def retrieve_event(path_input,event_mention_id):
    df_event=pd.read_csv(path_input)
    df_event.set_index('Event_Mention_ID')
    rows=df_event.loc[[event_mention_id]]
    return rows#返回event mention下所有的event






#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#数据构造函数
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------
#CrossDoc ERE 构建event pairs
# 思路是:
# 1.先用find_semantic_similarity找到大于threshold的所有event(注意:自己本身不算, 同个doc下的也不算(这是SDERE的范畴)<------这个在函数里面写好了不同管)
# 2.构建event pairs
#path_input_event是ED后的ED_event输出 列名为:Doc_ID,Event_Mention_ID,Event_ID,Event_Mention,Trigger,Trigger_Offset
#path_input_index是ED后的doc&real event mention(作为查询) 列名为:Doc_ID,Doc_Time,Event_Mention_ID,Event_Mention,Doc
#----------------------------------------------------------------------------------------
def create_CDERE_event_pair(path_input_event,path_input_index,path_output):
    columns=["Event_Mention_ID_A","Event_Mention_A","Event_ID_A","Trigger_A","Trigger_Offset_A",
             "Event_Mention_ID_B","Event_Mention_B","Event_ID_B","Trigger_B","Trigger_Offset_B",
             "Doc_ID_A","Doc_Time_A","Doc_A",
             "Doc_ID_B","Doc_Time_B","Doc_B"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    #两个文件都用event mention id建立索引
    raw_event_file=pd.read_csv(path_input_event)
    raw_event_file=raw_event_file.set_index("Event_Mention_ID")
    doc_event_mention_file=pd.read_csv(path_input_index)
    doc_event_mention_file=doc_event_mention_file.set_index("Event_Mention_ID")
    #手动tqdm
    pbar=tqdm(total=len(raw_event_file),desc='正在创建event pair.......')
    #某个event_mention_id下
    for event_mention_id_a, event_mention_df in raw_event_file.groupby('Event_Mention_ID'):
        #同个event mention只用找一次相似性
        #当然doc id都相同所以只用拿一次啦
        event_mention_a=doc_event_mention_file.loc[event_mention_id_a]['Event_Mention']
        doc_id_a=doc_event_mention_file.loc[event_mention_id_a]['Doc_ID']
        doc_time_a=doc_event_mention_file.loc[event_mention_id_a]['Doc_Time']
        doc_a=doc_event_mention_file.loc[event_mention_id_a]['Doc']
        #查询相似的event mention b
        top_event_mention_id_list=find_semantic_similarity(event_mention_id_a,SIMILAR_TOP_K,SIMILAR_THRESHOLD)
        if len(top_event_mention_id_list)==0:
            continue
        #所有的event a和相似的event mention b下的所有event b配对
        for _,event_a in event_mention_df.iterrows():
            event_id_a=event_a['Event_ID']
            trigger_a=event_a['Trigger']
            trigger_offset_a=event_a['Trigger_Offset']
            for event_mention_id_b in top_event_mention_id_list:
                #还是同样的,同个event mention只用找一次doc, doc_id, doc_time, event_mention_id
                event_mention_b=doc_event_mention_file.loc[event_mention_id_b]['Event_Mention']
                doc_id_b=doc_event_mention_file.loc[event_mention_id_b]['Doc_ID']
                doc_time_b=doc_event_mention_file.loc[event_mention_id_b]['Doc_Time']
                doc_b=doc_event_mention_file.loc[event_mention_id_b]['Doc']
                event_mention_b_df=raw_event_file.loc[[event_mention_id_b]]#返回一个dataframe
                for _,event_b in event_mention_b_df.iterrows():
                    event_id_b=event_b['Event_ID']
                    trigger_b=event_b['Trigger']
                    trigger_offset_b=event_b['Trigger_Offset']
                    save_data={
                        "Event_Mention_ID_A":event_mention_id_a,
                        "Event_Mention_A":event_mention_a,
                        "Event_ID_A":event_id_a,
                        "Trigger_A":trigger_a,
                        "Trigger_Offset_A":trigger_offset_a,
                        "Event_Mention_ID_B":event_mention_id_b,
                        "Event_Mention_B":event_mention_b,
                        "Event_ID_B":event_id_b,
                        "Trigger_B":trigger_b,
                        "Trigger_Offset_B":trigger_offset_b,
                        "Doc_ID_A":doc_id_a,
                        "Doc_Time_A":doc_time_a,
                        "Doc_A":doc_a,
                        "Doc_ID_B":doc_id_b,
                        "Doc_Time_B":doc_time_b,
                        "Doc_B":doc_b
                    }
                    pd.DataFrame([save_data]).to_csv(
                        path_output,
                        mode='a',#追加写.........
                        header=False, #不用写表头了
                        index=False,
                        encoding='utf-8'
                    )
            pbar.update(1)
    pbar.close()
    
                    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#cross doc 后的数据处理
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------------
#保存所有关系为共指的跨文档事件
#输入:CDERE的结果
#输出:共指的事件
#-------------------------------------------------------------------------------------
def filtering_CD_core(path_input,path_output):
    with open(path_input,'r',encoding='utf-8') as fin:
        with open(path_output,'a',encoding='utf-8') as fout:
            for line in fin:
                line=line.strip()#防止有空格或者空行
                if not line:
                    continue
                cd_ere = json.loads(line)
                if cd_ere["RELATION"]["COREFERENCE"] != "COREFERENCE":
                    continue
                else: 
                    fout.write(json.dumps(cd_ere,ensure_ascii=False)+"\n")


#-------------------------------------------------------------------------------------
#对jsonl建立索引!
#key可以为Background下的所有东西
#查询方式:results = index["具体的key值"]
#results是一个list[{},{},...,{}]
#-------------------------------------------------------------------------------------
def create_jsonl_uni_index(path_jsonl,key):
    index = defaultdict(list)
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data=json.loads(line)
            index[data["Background"][key]].append(data)
            #双向建立的话可以再append
            #index[data["Background"][key]].append(data)
    return index

def create_jsonl_bil_index(path_jsonl,key_1,key_2):
    index = defaultdict(list)
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            index[data["Background"][key_1]].append(data)
            #双向建立的话可以再append
            index[data["Background"][key_2]].append(data)
    return index


#-------------------------------------------------------------------------------------
#core的一跳关系
#-------------------------------------------------------------------------------------
def create_direct_CD_rela(path_sd,path_cd_core,path_output):
    index=create_jsonl_uni_index(path_cd_core,'Event_ID_A')              
    with open(path_sd,'r',encoding='utf-8') as fin:
        with open(path_output,'a',encoding='utf-8') as fout:
            for line in fin:
                line=line.strip()#防止有空格或者空行
                if not line:
                    continue
                sd_ere=json.loads(line)
                event_id_a=sd_ere["Background"]["Event_ID_A"]
                event_id_b=sd_ere["Background"]["Event_ID_B"]
                core_list_a=index[event_id_a]
                core_list_b=index[event_id_b]
                #A'=A--->B
                if len(core_list_a)!=0:
                    for row in core_list_a:
                        save_data={
                            "Event_Mention_A":row['Event_Mention_B'],
                            "Trigger_A":row["Trigger_B"],
                            "Trigger_Offset_A": row["Trigger_Offset_B"],
                            "Event_Mention_B": sd_ere["Event_Mention_B"],
                            "Trigger_B": sd_ere["Trigger_B"],
                            "Trigger_Offset_B": sd_ere["Trigger_Offset_B"],
                            "RELATION": sd_ere["RELATION"],
                            "Background": {
                                "Event_Mention_ID_A":row["Background"]['Event_Mention_ID_B'],
                                "Event_ID_A":row["Background"]["Event_ID_B"],
                                "Event_Mention_ID_B": sd_ere["Background"]["Event_Mention_ID_B"],
                                "Event_ID_B": sd_ere["Background"]["Event_ID_B"],
                                "Doc_ID_A": row["Background"]['Doc_ID_B'],
                                "Doc_Time_A":row["Background"]["Doc_Time_B"],
                                "Doc_A":row["Background"]["Doc_B"],
                                "Doc_ID_B": sd_ere["Background"]['Doc_ID'],
                                "Doc_Time_B":sd_ere["Background"]["Doc_Time"],
                                "Doc_B":sd_ere["Background"]["Doc"],
                                "Event_ID_transit": row["Background"]["Event_ID_A"],
                                "Event_Core_Trans":'A'#头事件是被传导的跨文档共指关系
                            }
                        }
                        fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")
                #A--->B=B'
                if len(core_list_b)!=0:
                    for row in core_list_b:
                        save_data={
                            "Event_Mention_A":sd_ere["Event_Mention_A"],
                            "Trigger_A":sd_ere["Trigger_A"],
                            "Trigger_Offset_A": sd_ere["Trigger_Offset_A"],
                            "Event_Mention_B": row["Event_Mention_B"],
                            "Trigger_B": row["Trigger_B"],
                            "Trigger_Offset_B": row["Trigger_Offset_B"],
                            "RELATION": sd_ere["RELATION"],
                            "Background": {
                                "Event_Mention_ID_A":sd_ere["Background"]['Event_Mention_ID_A'],
                                "Event_ID_A":sd_ere["Background"]["Event_ID_A"],
                                "Event_Mention_ID_B": row["Background"]["Event_Mention_ID_B"],
                                "Event_ID_B": row["Background"]["Event_ID_B"],
                                "Doc_ID_A": sd_ere["Background"]['Doc_ID'],
                                "Doc_Time_A":sd_ere["Background"]["Doc_Time"],
                                "Doc_A":sd_ere["Background"]["Doc"],
                                "Doc_ID_B": row["Background"]['Doc_ID_B'],
                                "Doc_Time_B":row["Background"]["Doc_Time_B"],
                                "Doc_B":row["Background"]["Doc_B"],
                                "Event_ID_transit": row["Background"]["Event_ID_A"],#中转的core事件点
                                "Event_Core_Trans":'B'#尾事件是被传导的跨文档共指关系
                            }
                        }
                        fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")



#-------------------------------------------------------------------------------------
#关系传导!
#这时候已经有了用cross-doc coreference一跳的关系了,再在已经跨文档的关系上进行关系传导
#注意应该传跨文档的core的那个事件(因为事件内部的是全连接,所以传导没有意义)
#这里只考虑jump3次的
#-------------------------------------------------------------------------------------
def create_rela_trans(path_cd_trans,path_sd,path_output,path_rela_trans_rule):
    #所有关系都被补全成双向边了
    #对于每个A--->B,考虑 ?--->A 和B --->?
    #只从跨文档的core Event那端jump
    index_out=create_jsonl_uni_index(path_sd,'Event_ID_A')
    index_in=create_jsonl_uni_index(path_sd,'Event_ID_B')
    with open(path_rela_trans_rule,'r',encoding='utf-8') as f:
        rela_trans_rule=json.load(f)
    with open(path_cd_trans,'r',encoding='utf-8') as fin:
        with open(path_output,'a',encoding='utf-8') as fout:
            for line in fin:
                line=line.strip()#防止有空格或者空行
                if not line:
                    continue
                cd_ere=json.loads(line)
                #?--->A'--->B
                # re_1   re_2
                if cd_ere["Background"]["Event_Core_Trans"]=='A':
                    sd_ere_list=index_in[cd_ere["Background"]["Event_ID_A"]] 
                    if len(sd_ere_list)!=0 :
                        for row in sd_ere_list:
                            change=0
                            re_save={
                                "TEMPORAL":"NO_TEMPORAL",
                                "CAUSAL":"NO_CAUSAL",
                                "SUBEVENT":"NO_SUBEVENT",
                                "COREFERENCE":"NO_COREFERENCE"
                            }
                            re_1=row["RELATION"]
                            re_2=cd_ere["RELATION"]
                            #进行关系传导
                            for rela_type in ["CAUSAL","TEMPORAL","SUBEVENT","COREFERENCE"]:
                                for k,v in rela_trans_rule[rela_type].items():
                                    if k==re_1[rela_type]:
                                        for m,n in v.items():
                                            if m==re_2[rela_type]:
                                                re_save[rela_type]=n
                                                change+=1
                            if change !=0:
                                save_data={
                                    "Event_Mention_A":row['Event_Mention_A'],
                                    "Trigger_A":row["Trigger_A"],
                                    "Trigger_Offset_A": row["Trigger_Offset_A"],
                                    "Event_Mention_B": cd_ere["Event_Mention_B"],
                                    "Trigger_B": cd_ere["Trigger_B"],
                                    "Trigger_Offset_B": cd_ere["Trigger_Offset_B"],
                                    "RELATION": re_save,
                                    "Background": {
                                        "Event_Mention_ID_A":row["Background"]['Event_Mention_ID_A'],
                                        "Event_ID_A":row["Background"]["Event_ID_A"],
                                        "Event_Mention_ID_B": cd_ere["Background"]["Event_Mention_ID_B"],
                                        "Event_ID_B":  cd_ere["Background"]["Event_ID_B"],
                                        "Doc_ID_A": row["Background"]['Doc_ID'],
                                        "Doc_Time_A":row["Background"]["Doc_Time"],
                                        "Doc_A":row["Background"]["Doc"],
                                        "Doc_ID_B": cd_ere["Background"]['Doc_ID_B'],
                                        "Doc_Time_B":cd_ere["Background"]["Doc_Time_B"],
                                        "Doc_B":cd_ere["Background"]["Doc_B"],
                                        "Event_ID_transit": row["Background"]["Event_ID_B"],
                                        "Event_Trans":'A'#头事件是被传导的关系
                                    }
                                }
                                fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")
                #A--->B'--->?
                # re_1   re_2
                if cd_ere["Background"]["Event_Core_Trans"]=='B':
                    sd_ere_list=index_out[cd_ere["Background"]["Event_ID_B"]] 
                    if len(sd_ere_list)!=0 :
                        for row in sd_ere_list:
                            change=0
                            re_save={
                                "TEMPORAL":"NO_TEMPORAL",
                                "CAUSAL":"NO_CAUSAL",
                                "SUBEVENT":"NO_SUBEVENT",
                                "COREFERENCE":"NO_COREFERENCE"
                            }
                            re_1=cd_ere["RELATION"]
                            re_2=row["RELATION"]
                            #进行关系传导
                            for rela_type in ["CAUSAL","TEMPORAL","SUBEVENT","COREFERENCE"]:
                                for k,v in rela_trans_rule[rela_type].items():
                                    if k==re_1[rela_type]:
                                        for m,n in v.items():
                                            if m==re_2[rela_type]:
                                                re_save[rela_type]=n
                                                change+=1
                            if change !=0:#真的能传导再保存
                                save_data={
                                    "Event_Mention_A":cd_ere['Event_Mention_A'],
                                    "Trigger_A":cd_ere["Trigger_A"],
                                    "Trigger_Offset_A": cd_ere["Trigger_Offset_A"],
                                    "Event_Mention_B": row["Event_Mention_B"],
                                    "Trigger_B": row["Trigger_B"],
                                    "Trigger_Offset_B": row["Trigger_Offset_B"],
                                    "RELATION": re_save,
                                    "Background": {
                                        "Event_Mention_ID_A":cd_ere["Background"]['Event_Mention_ID_A'],
                                        "Event_ID_A":cd_ere["Background"]["Event_ID_A"],
                                        "Event_Mention_ID_B": row["Background"]["Event_Mention_ID_B"],
                                        "Event_ID_B":  row["Background"]["Event_ID_B"],
                                        "Doc_ID_A": cd_ere["Background"]['Doc_ID_A'],
                                        "Doc_Time_A":cd_ere["Background"]["Doc_Time_A"],
                                        "Doc_A":cd_ere["Background"]["Doc_A"],
                                        "Doc_ID_B": row["Background"]['Doc_ID'],
                                        "Doc_Time_B":row["Background"]["Doc_Time"],
                                        "Doc_B":row["Background"]["Doc"],
                                        "Event_ID_transit": row["Background"]["Event_ID_A"],
                                        "Event_Trans":'B'#尾事件是被传导的关系
                                    }
                                }
                                fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")




