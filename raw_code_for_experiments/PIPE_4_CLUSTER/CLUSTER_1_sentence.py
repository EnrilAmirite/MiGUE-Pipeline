from collections import defaultdict

#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CLUSTER_function_design import line_similar_event_mention
from CONFIG.config import GLOBAL_CLUSTER_END_ROW_ID, GLOBAL_CLUSTER_START_ROW_ID,CROPUS
suffix=f"{CROPUS}_{GLOBAL_CLUSTER_START_ROW_ID}to{GLOBAL_CLUSTER_END_ROW_ID}"


#----------------------------------------------------------------------------------------
#一些路径和名称
#这里应该放想要聚类的所有doc&event mention information
#还有已经建好的emid :similar emid list
#----------------------------------------------------------------------------------------
path_doc_em_index=f"DATA/ED_input/event_mention_{suffix}.csv"
output_folder="DATA/CLUSTER_output"
output_name=f"similar_em_pair_{suffix}"
path_output=os.path.join(output_folder,f"{output_name}.jsonl")

if __name__=="__main__":
    line_similar_event_mention(
        path_input_doc_em_index=path_doc_em_index,
        path_output=path_output
    )