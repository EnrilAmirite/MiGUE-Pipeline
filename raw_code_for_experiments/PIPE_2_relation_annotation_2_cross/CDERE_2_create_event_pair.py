from CDERE_function_design import create_CDERE_event_pair
import os
#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CDERE_SUFFIX
suffix=GLOBAL_CDERE_SUFFIX
#----------------------------------------------------------------------------------------
#一些路径和名称
#这里最终应该放全部的event和全部的doc&event mention information
#----------------------------------------------------------------------------------------

path_event=f'DATA/ED_output/ED_event_{suffix}.csv'
path_doc_and_real_event_mention=f"DATA/ED_output/ED_doc_and_real_event_mention_{suffix}.csv"
output_folder="DATA/CDERE_input"
output_name=f"CDERE_event_pair_{suffix}"
path_output=os.path.join(output_folder,f"{output_name}.csv")


if __name__=='__main__':
    create_CDERE_event_pair(path_event,path_doc_and_real_event_mention,path_output)