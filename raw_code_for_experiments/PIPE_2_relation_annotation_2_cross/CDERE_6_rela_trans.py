import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import tqdm
from CDERE_function_design import create_rela_trans
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CDERE_SUFFIX
suffix=GLOBAL_CDERE_SUFFIX
path_cd=f'DATA/CDERE_output/CDERE_5_core_trans_{suffix}.jsonl'
path_sd=f"DATA/SDERE_output/SDERE_3_full_relation_{suffix}.jsonl"
path_rel_trans_rule="PIPE_3_ERE_CROSS_DOC/relation_transmission.json"
path_output=f"DATA/CDERE_output/CDERE_6_3hop_{suffix}.json"

if __name__=='__main__':
    create_rela_trans(
        path_cd_trans=path_cd,
        path_sd=path_sd,
        path_rela_trans_rule=path_rel_trans_rule,
        path_output=path_output
    )