from SDERE_function_design import create_SDERE_event_pair
import os


#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_SDERE_SUFFIX,CROPUS
suffix=GLOBAL_SDERE_SUFFIX


#----------------------------------------------------------------------------------------
#一些路径和名称
#这里最终应该放: 全部的event, event mention/doc 对照csv 即可
#----------------------------------------------------------------------------------------
if CROPUS=="RU_CLUSTER":
    path_event=f"DATA/ED_output/cluster/ED_event_{suffix}.csv"
    path_doc=f"DATA/ED_output/cluster/ED_doc_and_real_event_mention_{suffix}.csv"
    output_folder="DATA/SDERE_input/cluster"
else:
    path_event=f"DATA/ED_output/ED_event_{suffix}.csv"
    path_doc=f"DATA/ED_output/ED_doc_and_real_event_mention_{suffix}.csv"
    output_folder="DATA/SDERE_input"

output_name=f"SDERE_1_event_pair"
path_output=os.path.join(output_folder,f"{output_name}_{suffix}.csv")


if __name__=="__main__":
    create_SDERE_event_pair(path_event=path_event,path_doc=path_doc,path_output=path_output)