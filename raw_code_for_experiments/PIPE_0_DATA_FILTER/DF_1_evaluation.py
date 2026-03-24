
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import api_setting
from data_filter_function_design import eval_raw_doc_concurrent

path_input="DATA/raw_data/raw_docs_TCE_2022.csv"
path_output="DATA/raw_data/raw_docs_TCE_2022.csv"
path_prompt="PIPE_0_DATA_FILTER/df_rule.json"
MODEL="gpt-4o-2024-08-06"
url,key=api_setting(MODEL)

if __name__=='__main__':
    eval_raw_doc_concurrent(
        input_path=path_input,
        output_path=path_output,
        prompt_path=path_prompt,
        url=url,
        key=key,
        model=MODEL,
        max_workers=10
    )