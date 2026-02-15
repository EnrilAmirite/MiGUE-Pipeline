import pandas as pd
import re
from datetime import datetime
import os
from tqdm import tqdm
import hanlp
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_ED_SUFFIX,CROPUS
suffix=GLOBAL_ED_SUFFIX

#-------------------------------------------------------------------
#路径
#-------------------------------------------------------------------
if CROPUS=="RU_CLUSTER":
    path_input=f"DATA/raw_data/cluster/{suffix}.csv"
    fold_output="DATA/ED_input/cluster"
else:
    path_input=f"DATA/raw_data/raw_docs_{suffix}.csv"
    fold_output="DATA/ED_input"
name_output="event_mention"
path_output=os.path.join(fold_output,f"{name_output}_{suffix}.csv")

#-------------------------------------------------------------------
#分句...分句
#-------------------------------------------------------------------
def doc_to_sentence(text,doc_id):
    stack=[]
    sentence_list=[]
    temp=[]
    quote_begin_list=['"','“','‘',"'"]
    quote_end_list=['"','”','’',"'"]
    end_set={'。', '！', '？','!','?'}
    #只有list(有序)才能这样建立字典
    pair_quote=dict(zip(quote_begin_list,quote_end_list))
    for char in text:
        temp.append(char)
        if char in quote_begin_list:
            stack.append(char)
        elif char in quote_end_list:
            if len(stack)!=0:
                top=stack.pop()
                if pair_quote[top] != char:
                    print(f"语料有问题,前后引号不对齐,当前的doc_id为:{doc_id}")
                    return []
            else:
                print(f"语料有问题,缺少前引号,当前的doc_id为:{doc_id}")
                return []
        if char in end_set and len(stack)==0:
            sentence=''.join(temp).strip()#char和char中间不需要加任何东西,直接拼接
            if len(sentence)>=3:
                sentence_list.append(sentence)
            temp=[]
        elif char in quote_end_list and len(temp)>=2 and temp[-2]in end_set:
            sentence=''.join(temp).strip()#char和char中间不需要加任何东西,直接拼接
            if len(sentence)>=3:
                sentence_list.append(sentence)
            temp=[]
    if len(temp)!=0:
        sentence=''.join(temp).strip()
        sentence_list.append(sentence)
        print(f"语料有问题,可能缺少结束符,或者只有前引号没有后引号,doc_id为{doc_id},但是这里并不影响输出O(∩_∩)O")
    return sentence_list

#print (doc_to_sentence("我说：“今天天气好好。我们要不要出去玩呀？”说完我往外走",1))


    
#-------------------------------------------------------------------
#字符串变成list
#-------------------------------------------------------------------
def string_to_num_list(input_string):#这里不能传空值NaN
    list=[int(x) for x in input_string.split(',')]
    return list

def string_to_name_list(input_string):
    list=[x.strip() for x in input_string.split(',')]
    return list


#-------------------------------------------------------------------
#主函数
#-------------------------------------------------------------------
def split_doc_to_event_mention_raw():
    #因为是一行一行加,所以先建立文档
    columns=["Doc_ID","Event_Mention_ID","Event_Mention"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    #分句......
    df=pd.read_csv(path_input)
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        sentence_list=[]
        doc=row["content"]
        doc_id=row["doc_id"]
        vote_str=row["filtered_result"]
        name_str=row["filtered_by"]
        #跳过没有人筛的doc
        if pd.isna(vote_str):
            continue
        vote_list=string_to_num_list(vote_str)
        name_list=string_to_name_list(name_str)
        name_to_vote=dict(zip(name_list,vote_list))

        if "邵帅"in name_list and name_to_vote["邵帅"]==0:#如果邵帅师兄觉得不能保留那就不能保留
            continue 
        else:
            #除此之外,一个以上的人觉得不合格就跳过
            #print(f"row_id:{idx+1}")
            #print(vote_list.count(0))
            if vote_list.count(0)>1:
                #print(f"被筛掉的列:{idx+1}")
                continue
        #分句..
        sentence_list=doc_to_sentence(doc,doc_id)
        for i,event_mention in enumerate(sentence_list):
            event_id=f"EVENT_{doc_id}_{i+1}"
            data={
                "Doc_ID":doc_id,
                "Event_Mention_ID":event_id,
                "Event_Mention":event_mention
            }
            pd.DataFrame([data]).to_csv(
                path_output,
                mode='a',
                index=False,
                header=False,
                encoding='utf-8'
            )

def split_doc_to_event_mention():
    #因为是一行一行加,所以先建立文档
    columns=["Doc_ID","Event_Mention_ID","Event_Mention"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    #分句......
    df=pd.read_csv(path_input)
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        sentence_list=[]
        doc=row["content"]
        doc_id=row["doc_id"]
        vote_str=row["filtered_result"]
        name_str=row["filtered_by"]
        #后期没有人筛直接算过了
        if pd.isna(vote_str):
            sentence_list=doc_to_sentence(doc,doc_id)
            for i,event_mention in enumerate(sentence_list):
                event_id=f"EVENT_{doc_id}_{i+1}"
                data={
                    "Doc_ID":doc_id,
                    "Event_Mention_ID":event_id,
                    "Event_Mention":event_mention
                }
                pd.DataFrame([data]).to_csv(
                    path_output,
                    mode='a',
                    index=False,
                    header=False,
                    encoding='utf-8'
                )
            continue
        vote_list=string_to_num_list(vote_str)
        name_list=string_to_name_list(name_str)
        name_to_vote=dict(zip(name_list,vote_list))

        if "邵帅"in name_list and name_to_vote["邵帅"]==0:#如果邵帅师兄觉得不能保留那就不能保留
            continue 
        else:
            #除此之外,一个以上的人觉得不合格就跳过
            #print(f"row_id:{idx+1}")
            #print(vote_list.count(0))
            if vote_list.count(0)>1:
                #print(f"被筛掉的列:{idx+1}")
                continue
        #分句..
        sentence_list=doc_to_sentence(doc,doc_id)
        for i,event_mention in enumerate(sentence_list):
            event_id=f"EVENT_{doc_id}_{i+1}"
            data={
                "Doc_ID":doc_id,
                "Event_Mention_ID":event_id,
                "Event_Mention":event_mention
            }
            pd.DataFrame([data]).to_csv(
                path_output,
                mode='a',
                index=False,
                header=False,
                encoding='utf-8'
            )

def split_doc_to_event_mention_llm():
    #因为是一行一行加,所以先建立文档
    columns=["Doc_ID","Event_Mention_ID","Event_Mention"]
    if not os.path.exists(path_output):
        pd.DataFrame(columns=columns).to_csv(
            path_output,index=False,encoding='utf-8'
        )
    #分句......
    df=pd.read_csv(path_input)
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        sentence_list=[]
        doc=row["content"]
        doc_id=row["doc_id"]
        '''
        vote_str=row["filtered_result"]
        name_str=row["filtered_by"]
        #跳过没有人筛的doc
        if pd.isna(vote_str):
            continue
        vote_list=string_to_num_list(vote_str)
        name_list=string_to_name_list(name_str)
        name_to_vote=dict(zip(name_list,vote_list))

        if "邵帅"in name_list and name_to_vote["邵帅"]==0:#如果邵帅师兄觉得不能保留那就不能保留
            continue 
        else:
            #除此之外,一个以上的人觉得不合格就跳过
            #print(f"row_id:{idx+1}")
            #print(vote_list.count(0))
            if vote_list.count(0)>1:
                #print(f"被筛掉的列:{idx+1}")
                continue
        '''
        #分句..
        sentence_list=doc_to_sentence(doc,doc_id)
        for i,event_mention in enumerate(sentence_list):
            event_id=f"EVENT_{doc_id}_{i+1}"
            data={
                "Doc_ID":doc_id,
                "Event_Mention_ID":event_id,
                "Event_Mention":event_mention
            }
            pd.DataFrame([data]).to_csv(
                path_output,
                mode='a',
                index=False,
                header=False,
                encoding='utf-8'
            )

if __name__=="__main__":
    if os.path.exists(path_output):
        print("以前已经生成过这个文件啦,请删除后重新生成哦TT")
    else:
        #split_doc_to_event_mention()
        split_doc_to_event_mention_llm()



