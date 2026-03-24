import pandas as pd
from tqdm import tqdm

from src.event_anno.load_eva_config import load_eva_config
from src.base_code.load_data import load_data

class doc2stc(load_eva_config,load_data):
    def __init__(self):
        load_eva_config.__init__(self)
        load_data.__init__(self)

    def doc2stclist(self,text,doc_id):
        stack=[]
        sentence_list=[]
        temp=[]
        quote_begin_list=['"','“','‘',"'"]
        quote_end_list=['"','”','’',"'"]
        end_set={'。', '！', '？','!','?'}
        pair_quote=dict(zip(quote_begin_list,quote_end_list))
        for char in text:
            temp.append(char)
            if char in quote_begin_list:
                stack.append(char)
            elif char in quote_end_list:
                if len(stack)!=0:
                    top=stack.pop()
                    if pair_quote[top] != char:
                        print(f"The raw document has some issues; the quotation marks are misaligned. The current doc id is: {doc_id}")
                        return []
                else:
                    print(f"The raw document has some issues; it's missing an opening quotation mark. The current doc_id is: {doc_id}")
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
            print(f"The raw document has some issues; it may be missing a terminator, or it may only have an opening quotation mark but no closing quotation mark. doc id is {doc_id}`, but this doesn't affect the output.O(∩_∩)O")
        return sentence_list
    

    def split_doc2stc(self):
        if self.checkfile(self.stc_p):
            return
        columns = ["doc_id", "stc_id", "stc"]
        self.write_csv_head(col=columns,path=self.stc_p)
        #分句......
        df=pd.read_csv(self.ip_p)
        for idx,row in tqdm(df.iterrows(),total=len(df)):
            sentence_list=[]
            doc=row["doc"]
            doc_id=row["doc_id"]
            #分句..
            sentence_list=self.doc2stclist(doc,doc_id)
            for i,stc in enumerate(sentence_list):
                stc_id=f"{doc_id}_stc_{i+1}"
                data={
                    "doc_id":doc_id,
                    "stc_id":stc_id,
                    "stc":stc
                }
                pd.DataFrame([data]).to_csv(
                    self.stc_p,
                    mode='a',
                    index=False,
                    header=False,
                    encoding='utf-8'
                )