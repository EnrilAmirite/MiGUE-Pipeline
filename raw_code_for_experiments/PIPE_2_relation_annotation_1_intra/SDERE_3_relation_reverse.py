

from PIPE_2_relation_annotation_intra.function_design import complete_reverse_relation
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_SDERE_END_ROW_ID, GLOBAL_SDERE_START_ROW_ID,CROPUS


start_row_id=GLOBAL_SDERE_START_ROW_ID
end_row_id=GLOBAL_SDERE_END_ROW_ID
path_input=f"DATA/SDERE_output/SDERE_2_{CROPUS}_{start_row_id}to{end_row_id}.jsonl"
path_output=f"DATA/SDERE_output/SDERE_3_full_relation_{CROPUS}_{start_row_id}to{end_row_id}.jsonl"
path_rever_rule="PIPE_2_relation_annotation_intra/reverse_map.json"


if __name__=="__main__":
    complete_reverse_relation(
        path_input=path_input,
        path_output=path_output,
        path_rever_rule=path_rever_rule
    )