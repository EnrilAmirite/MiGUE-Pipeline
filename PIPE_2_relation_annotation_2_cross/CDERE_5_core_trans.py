import json
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from CDERE_function_design import create_direct_CD_rela

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CDERE_SUFFIX
suffix=GLOBAL_CDERE_SUFFIX
#-------------------------------------------------------------------------------------
#这里传导的关系是: doc_1: A--->B  doc_2: A--->A'
#A--->A'--->B
#保存的全是跨文档的A'--->B的关系
#反过来也一样:doc_1: A--->B doc_2: B--->B'
#传导A--->B--->B'
#-------------------------------------------------------------------------------------

path_sdere = f"DATA/SDERE_output/SDERE_3_full_relation_{suffix}.jsonl"
path_cd_core = f"DATA/CDERE_input/CDERE_4_CD_core_{suffix}.jsonl"
path_output = f"DATA/CDERE_output/CDERE_5_core_trans_{suffix}.jsonl"

if __name__=="__main__":
    create_direct_CD_rela(
        path_cd_core=path_cd_core,
        path_output=path_output,
        path_sd=path_sdere 
    )
    