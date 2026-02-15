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


#==================================输入输出路径,输出名称=================================
path_option="PIPE_2_relation_annotation_intra/options.json"
path_rule="PIPE_2_relation_annotation_intra/rules.json"
path_description="PIPE_2_relation_annotation_intra/relation_description.json"



#============================================全局变量===================================
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


# ==========================================================================
# 维护函数
# 每次两个事件的关系全部找完后清空所有的全局变量
#对于可变全局对象(list/dict/set),用.clear()是直接改变对象本身的值
#这种方法只对可变对象有效，对不可变对象(int/str/tuple)不起作用
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

#=====================================================================
#把关系选项，变成schema level relation
#比如:把REL_1变成TEMPORAL
def option_to_relation_name(model_option):
    with open(path_option,"r",encoding="utf-8")as f:
        options=json.load(f)
    mapping=options["RELATION_CHOOSE"]
    relation_name=mapping[model_option]["relation"]
    return relation_name



#=====================================================================
#把关系选项，变成具体关系名称
#比如:把TEMP_1变成BEFORE,TEMP_2变成AFTER
def option_to_specific_name(model_relation,model_option):
    with open(path_option,"r",encoding="utf-8")as f:
        options=json.load(f)    
    mapping=options[model_relation]
    specific_name=mapping[model_option]["name"]
    return specific_name


def SDERE_prompt_intro(doc,event_mention_A,event_mention_B,trigger_A,trigger_B,offset_A,offset_B):
    path_ED_prompt="PIPE_1_ED/ED_prompt.json"
    prompt_intro_event=""
    with open(path_ED_prompt,'r',encoding='utf-8') as f:
        prompt_file=json.load(f)
        task_intro=prompt_file["Task_Intro"]
        for key,v in task_intro.items():
            if key!="Given_Event_mention":
                prompt_intro_event+=v
    prompt_intro_ere="事件关系抽取(Events Relation Extract,ERE)即指判断两个触发词对应的两个事件在'时序','因果','子事件','共指'四个层面的关系\n"
    prompt_intro_doc_and_event=f"""
请根据上述定义,结合原文,在下面的事件对中,判断事件A和事件B在'时序','因果','子事件','共指'四个层面的关系.
原文档:{doc}
事件A的事件提及:{event_mention_A}
事件A的触发词:{trigger_A}
事件A的触发词在事件提及中的位置:{offset_A}
事件B的事件提及:{event_mention_B}
事件B的触发词:{trigger_B}
事件B的触发词在事件提及中的位置:{offset_B}
\n
"""
    prompt=prompt_intro_event+prompt_intro_ere+prompt_intro_doc_and_event
    return prompt

#=====================================================================
#根据model的回答维护MUST_NOT表格和MUST表格
#需要用rules.json维护MUST_NOT,减去里面提到的关系
#用rules.json维护MUST,加上must里的关系
#MUST_NOT,MUST维护的时候用extend(虽然MUST好像没有extend的必要...)
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

#=====================================================================
#维护已经回答的问题列表
def maintain_have_answer(relation_answer,specific_answer):
    #relation
    RELATION_ANSWER_LIST.append(relation_answer)
    #specific answer
    SPECIFIC_ANSWER_LIST.append(specific_answer)


#=====================================================================
#生成specific的问题
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




#=====================================================================
#生成relation问题
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


#=====================================================================
#后验步骤(1):选择下一步的操作ACT_1 ACT_2 ACT_3
#主要思路:输出MUST_NOT表格,MUST表格,让model思考自己是否选错了
#有一种情况是,现在的rules约束不会有改变,所以用这个函数的时候需要判断一下返回的prompt是不是为空
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



#-----------------------------------------------------------
#SDERE创建event pair
#-----------------------------------------------------------
def create_SDERE_event_pair(path_event,path_doc,path_output):
    columns=["Event_Mention_ID_A","Event_Mention_A","Event_ID_A","Trigger_A","Trigger_Offset_A",
             "Event_Mention_ID_B","Event_Mention_B","Event_ID_B","Trigger_B","Trigger_Offset_B",
             "Doc_ID","Doc_Time","Doc"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    raw_event_file=pd.read_csv(path_event)
    raw_event_and_mention_file=pd.read_csv(path_doc)
    raw_event_and_mention_file=raw_event_and_mention_file.set_index('Doc_ID')
    #只组合同个doc_id下的
    for group,group_df in raw_event_file.groupby('Doc_ID'):
        #重设index(不然容易乱)
        group_df = group_df.reset_index(drop=True)
        doc_id=group_df.iloc[0]['Doc_ID']
        doc=raw_event_and_mention_file.loc[[doc_id]].iloc[0]#返回一个dataframe,再取第一行
        doc_time=doc['Doc_Time']
        doc_mention=doc["Doc"]
        for i in range(len(group_df)):
            event_a=group_df.iloc[i]
            event_mention_id_a=event_a['Event_Mention_ID']
            event_mention_a=event_a['Event_Mention']
            event_id_a=event_a['Event_ID']
            trigger_a=event_a['Trigger']
            trigger_offset_a=event_a['Trigger_Offset']
            for j in range(i+1, len(group_df)):
                event_b=group_df.iloc[j]
                event_mention_id_b=event_b['Event_Mention_ID']
                event_mention_b=event_b['Event_Mention']
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
                    "Doc_ID":doc_id,
                    "Doc_Time":doc_time,
                    "Doc":doc_mention,
                }
                pd.DataFrame([save_data]).to_csv(
                    path_output,
                    mode='a',#追加写.........
                    header=False, #不用写表头了
                    index=False,
                    encoding='utf-8'
                )

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#后处理部分
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
#补全反向关系边
#输入是ERE后直接得到的单向结果~
#----------------------------------------------------------------------------------------
def complete_reverse_relation(path_input,path_output,path_rever_rule):
    with open(path_rever_rule,'r',encoding='utf-8') as f:
        reverse_rule=json.load(f)
    with open(path_input,'r',encoding='utf-8') as fin:
        with open(path_output,'a',encoding='utf-8') as fout:
            for line in fin:
                line=line.strip()#防止有空格或者空行
                if not line:
                    continue
                raw_ere=json.loads(line)
                raw_rela=raw_ere['RELATION']
                new_rela={
                    "TEMPORAL":"",
                    "CAUSAL":"",
                    "SUBEVENT":"",
                    "COREFERENCE":""
                }
                for rela_type in ["TEMPORAL","CAUSAL","SUBEVENT","COREFERENCE"]:
                    for k,v in reverse_rule[rela_type].items():
                        if k==raw_rela[rela_type]:
                            new_rela[rela_type]=v
                save_data={
                    "Event_Mention_A": raw_ere["Event_Mention_B"],
                    "Trigger_A": raw_ere["Trigger_B"],
                    "Trigger_Offset_A": raw_ere["Trigger_Offset_B"],
                    "Event_Mention_B": raw_ere["Event_Mention_A"],
                    "Trigger_B": raw_ere["Trigger_A"],
                    "Trigger_Offset_B": raw_ere["Trigger_Offset_A"],
                    "RELATION": new_rela,
                    "Background": {
                        "Event_Mention_ID_A": raw_ere["Background"]["Event_Mention_ID_B"],
                        "Event_ID_A": raw_ere["Background"]["Event_ID_B"],
                        "Event_Mention_ID_B": raw_ere["Background"]["Event_Mention_ID_A"],
                        "Event_ID_B": raw_ere["Background"]["Event_ID_A"],
                        "Doc_ID": raw_ere["Background"]["Doc_ID"],
                        "Doc_Time": raw_ere["Background"]["Doc_Time"],
                        "Doc": raw_ere["Background"]["Doc"]
                    }
                }
                fout.write(line.rstrip("\n") + "\n")
                fout.write(json.dumps(save_data,ensure_ascii=False)+"\n")
