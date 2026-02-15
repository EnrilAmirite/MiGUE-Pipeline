import json
import jsonlines
from openai import OpenAI
import os
from datetime import datetime
from tqdm import tqdm
import csv
from pydantic import BaseModel
import pandas as pd
from ED_function_design import create_example, create_reflection, create_task_intro, list_to_str
import re #正则
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CONFIG.config import GLOBAL_ED_SUFFIX,CROPUS
suffix=GLOBAL_ED_SUFFIX
from ED_function_design import create_doc_and_real_event_mention


#-------------------------------------------------------------------------
#一些路径和名称
#-------------------------------------------------------------------------
if CROPUS=="RU_CLUSTER":
    path_raw_doc=f"DATA/raw_data/cluster/{suffix}.csv"
    path_input=f"DATA/ED_output/cluster/ED_{suffix}.csv"
    folder_output="DATA/ED_output/cluster" 
else:
    path_raw_doc=f"DATA/raw_data/raw_docs_{suffix}.csv"
    path_input=f"DATA/ED_output/ED_{suffix}.csv"
    folder_output="DATA/ED_output" 
output_name="ED_doc_and_real_event_mention"
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
path_output= os.path.join(folder_output, f"{output_name}_{suffix}.csv")


if __name__=='__main__':
    create_doc_and_real_event_mention(path_input,path_raw_doc,path_output)



