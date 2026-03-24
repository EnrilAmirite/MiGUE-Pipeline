import pandas as pd
from tqdm import tqdm

path_input="DATA/raw_data/raw_docs_TCE_2022.csv"
path_output="DATA/raw_data/symbol_error_doc_id_TCE.txt"
#-------------------------------------------------------------------
#找到符号不正确的doc id
#-------------------------------------------------------------------
def find_wrong_doc(text):
    stack=[]
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
                    return False
                if len(temp)>2 and temp[-2]in end_set:
                    temp=[]
            else:
                return False
        if char in end_set and len(stack)==0:
            temp=[]
    if len(temp)!=0:
        return False
    return True




def find_symbol_error_doc_id(path_input):
    df=pd.read_csv(path_input)
    if "processed" not in df.columns:
        df["processed"]=0
    wrong_doc_id_list=[]
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        doc=row["content"]
        doc_id=row["doc_id"]
        if find_wrong_doc(doc):
            df.loc[idx,"processed"]=1
            #df.to_csv(path_input, index=False)
            continue
        else:
            wrong_doc_id_list.append(doc_id)
    with open(path_output, "w", encoding="utf-8") as f:
        f.write(",".join(map(str,wrong_doc_id_list)))
    print(f"符号错误的doc有{len(wrong_doc_id_list)}")

if __name__=="__main__":
    find_symbol_error_doc_id(path_input=path_input)


