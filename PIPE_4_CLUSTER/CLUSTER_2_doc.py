from collections import defaultdict
import networkx as nx
import json
import pandas as pd
from collections import defaultdict, Counter
import networkx as nx
import igraph as ig
import leidenalg


#-------------------------------------------------------------------
import sys
import os
#将当前脚本的父目录(即Pipeline目录)添加到路径中
#添加目标:即根目录Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#-------------------------------------------------------------------
from CONFIG.config import GLOBAL_CLUSTER_SUFFIX
suffix=GLOBAL_CLUSTER_SUFFIX
from CLUSTER_2_sentence import path_output,path_doc_em_index

doc_edge_weight = defaultdict(int)
#----------------------------------------------------------------------------------------
#一些路径和名称
#这里应该放想要聚类的所有doc&event mention information
#还有已经建好的emid :similar emid list
#----------------------------------------------------------------------------------------
path_docid_doc_index=f"DATA/raw_data/raw_docs_{suffix}.csv"
path_em_simliar_index=path_output
output_folder="DATA/CLUSTER_output"
output_name=f"doc_cluster_{suffix}"
path_output=os.path.join(output_folder,f"{output_name}.jsonl")
MIN_EDGE_WEIGHT = 2   # sentence 对至少多少个才连 doc 边
MIN_CLUSTER_SIZE = 2  # 至少几个 doc 才输出



df = pd.read_csv(path_doc_em_index)
em2doc=dict(zip(df["Event_Mention_ID"], df["Doc_ID"]))
df=pd.read_csv(path_docid_doc_index)
docid2doc=dict(zip(df["doc_id"], df["content"]))
docid2doctime=dict(zip(df["doc_id"], df["created_at"]))
# =========================
# 3. sentence → doc 投票
# =========================
doc_edge_weight = defaultdict(int)

with open(path_em_simliar_index, "r", encoding="utf-8") as f:
    for line in f:
        obj=json.loads(line)
        em_id=obj["Event_Mention_ID"]
        sim_em=obj["Similar_EM_ID_List"]
        if em_id not in em2doc:
            continue
        doc_i = em2doc[em_id]
        for em_j in sim_em:
            if em_j not in em2doc:
                continue
            doc_j = em2doc[em_j]
            if doc_i == doc_j:
                continue
            #无向图防止重复...
            if doc_i < doc_j:
                doc_edge_weight[(doc_i, doc_j)] += 1
            else:
                doc_edge_weight[(doc_j, doc_i)] += 1


#构建 doc graph
G = nx.Graph()

for (d1, d2), w in doc_edge_weight.items():
    if w >= MIN_EDGE_WEIGHT:
        G.add_edge(d1, d2, weight=w)

print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")


#NetworkX → igraph
doc2idx = {doc: i for i, doc in enumerate(G.nodes())}
idx2doc = {i: doc for doc, i in doc2idx.items()}

edges = []
weights = []

for d1, d2, data in G.edges(data=True):
    edges.append((doc2idx[d1], doc2idx[d2]))
    weights.append(data["weight"])

ig_graph = ig.Graph(
    n=len(doc2idx),
    edges=edges,
    edge_attrs={"weight": weights}
)


#Leiden 聚类
partition = leidenalg.find_partition(
    ig_graph,
    leidenalg.RBConfigurationVertexPartition,
    weights="weight",
    resolution_parameter=1.0
)

doc_cluster = {
    idx2doc[i]: cid
    for i, cid in enumerate(partition.membership)
}


# 7. 组织 cluster 结果
cluster2docs = defaultdict(list)

for doc, cid in doc_cluster.items():
    cluster2docs[cid].append(doc)

degree = dict(G.degree(weight="weight"))


# =========================
# 8. 写出 jsonl
# =========================
with open(path_output, "w", encoding="utf-8") as f:
    for cid, docs in cluster2docs.items():
        if len(docs) < MIN_CLUSTER_SIZE:
            continue
        docs_info = [
            {
                "Doc_ID": d,
                "Doc_Time":docid2doctime[d],
                "Doc":docid2doc[d],
                "Degree": degree.get(d, 0)
            }
            for d in sorted(docs, key=lambda x: degree.get(x, 0), reverse=True)
        ]
        record = {
            "Cluster_ID": cid,
            "Num_Docs": len(docs),
            "Docs": docs_info
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"已经把doc筛成一坨一坨的了!")
