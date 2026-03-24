#这里是pipeline的**不能更改的**全局变量
#如果要更改需要直接在这里改



#-------------------------------------------------------------------
#如果调用这个config.py需要在顶部加:
#-------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------


#原始语料
#CROPUS="RU"
#CROPUS="TCE_2022"
CROPUS="RU_CLUSTER"
#RUCLUSTER这个的id就是0to0, 1to1...这种


#api设置
def api_setting(model_name):
    model_url=""
    model_key=""
    if model_name in ["gpt-4o-mini","gpt-4o-2024-08-06","gpt-5.2-pro","gpt-5.1"]:
        model_url="https://api.holdai.top/v1"
        model_key="......................."
    elif model_name in ["deepseek-reasoner","deepseek-chat"]:
        model_url="https://api.deepseek.com/v1"
        model_key="......................."
    return model_url,model_key


#-------------------------------------------------------------------
#ED部分
#-------------------------------------------------------------------
#当前处理的raw doc的行数起止(对应database中的id列)
GLOBAL_ED_START_ROW_ID=3
GLOBAL_ED_END_ROW_ID=3
#从database直接保存的raw doc路径
GLOBAL_ED_PATH_RAW="DATA/raw_data/raw_docs_202512291224.csv"
GLOBAL_ED_SUFFIX=f"{CROPUS}_{GLOBAL_ED_START_ROW_ID}to{GLOBAL_ED_END_ROW_ID}"

#-------------------------------------------------------------------
#SDERE部分
#-------------------------------------------------------------------
#当前处理的event行数起止(对应原本database中的id列)
GLOBAL_SDERE_START_ROW_ID=20
GLOBAL_SDERE_END_ROW_ID=20
GLOBAL_SDERE_SUFFIX=f"{CROPUS}_{GLOBAL_SDERE_START_ROW_ID}to{GLOBAL_SDERE_END_ROW_ID}"

#-------------------------------------------------------------------
#CDERE部分
#-------------------------------------------------------------------
#当前处理的event行数起止(对应原本database中的id列)
GLOBAL_CDERE_START_ROW_ID=20
GLOBAL_CDERE_END_ROW_ID=20
GLOBAL_CDERE_SUFFIX=f"{CROPUS}_{GLOBAL_CDERE_START_ROW_ID}to{GLOBAL_CDERE_END_ROW_ID}"


#-------------------------------------------------------------------
#CLUSTER部分
#-------------------------------------------------------------------
#当前处理的event行数起止(对应原本database中的id列)
GLOBAL_CLUSTER_START_ROW_ID=6001
GLOBAL_CLUSTER_END_ROW_ID=10000
GLOBAL_CLUSTER_SUFFIX=f"{CROPUS}_{GLOBAL_CLUSTER_START_ROW_ID}to{GLOBAL_CLUSTER_END_ROW_ID}"