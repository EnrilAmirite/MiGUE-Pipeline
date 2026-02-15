


from data_filter_function_design import translate_en2ch

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import api_setting
from data_filter_function_design import trans_raw_doc_concurrent

path_input="DATA/raw_data/raw_docs_TCE_2022_vote.csv"
path_output="DATA/raw_data/raw_docs_TCE_2022.csv"
key="acadf7ba-ea28-42c2-b56f-889266aae630"




if __name__=="__main__":
    translate_en2ch(path_input,path_output,key)