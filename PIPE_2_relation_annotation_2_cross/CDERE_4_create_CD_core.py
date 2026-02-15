import json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from CDERE_function_design import filtering_CD_core

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CDERE_SUFFIX
suffix=GLOBAL_CDERE_SUFFIX

path_input=f"DATA/CDERE_output/CDERE_3_{suffix}.jsonl"
path_output=f"DATA/CDERE_input/CDERE_4_CD_core_{suffix}.jsonl"

if __name__=="__main__":
    filtering_CD_core(path_input,path_output)

