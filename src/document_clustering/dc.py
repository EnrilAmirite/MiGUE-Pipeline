import pandas as pd
import json

from collections import defaultdict, Counter
import networkx as nx
import igraph as ig
import leidenalg

from src.document_clustering.load_dc_config import load_dc_config



class dc(load_dc_config):
    def __init__(self):
        load_dc_config.__init__(self)

        self.doc_df=pd.read_csv(self.doc_p)
    

    def linecore(self):
        adj_list = defaultdict(set)
        print(f"\nLining co-reference events (in different documents)...")
        with open(self.cdcore_p,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    eida = data['bg'].get('e_id_a')
                    eidb = data['bg'].get('e_id_b')
                    if eida and eidb:
                        adj_list[eida].add(eidb)
                        adj_list[eidb].add(eida)
                except (KeyError, json.JSONDecodeError):
                    continue
        with open(self.coreline_p, 'w', encoding='utf-8') as f_out:
            #sort e_id set
            for e_id in sorted(adj_list.keys()):
                neighbors = sorted(list(adj_list[e_id]))
                result = {
                    "e_id": e_id,
                    "core_e_id_list": neighbors
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n Finish linking ~ ~")


    def clusterdoc(self):
        self.doc_df["doc_id"] = self.doc_df["doc_id"].astype(str)
        docid2doc=dict(zip(self.doc_df["doc_id"], self.doc_df["doc"]))
        docid2doctime=dict(zip(self.doc_df["doc_id"], self.doc_df["doc_time"]))
        doc_edge_weight = defaultdict(int)
        with open(self.coreline_p, "r", encoding="utf-8") as f:
            for line in f:
                obj=json.loads(line)
                eiid=obj["e_id"]
                core_e_list=obj["core_e_id_list"]
                dociid,emiid=self.eid2docidNemid(eiid)
                if dociid not in docid2doc:
                    continue
                for ejid in core_e_list:
                    docjid,emjid= self.eid2docidNemid(ejid)
                    if docjid not in docid2doc:
                        continue
                    if dociid == docjid:
                        continue
                    #无向图防止重复...
                    if dociid < docjid:
                        doc_edge_weight[(dociid, docjid)] += 1
                    else:
                        doc_edge_weight[(docjid, dociid)] += 1
        #create doc graph
        G = nx.Graph()
        for (d1, d2), w in doc_edge_weight.items():
            if w >= self.minedge:
                G.add_edge(d1, d2, weight=w)
        print(f"\nGraph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
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
        #Leiden
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
        cluster2docs = defaultdict(list)
        for doc, cid in doc_cluster.items():
            cluster2docs[cid].append(doc)
        degree = dict(G.degree(weight="weight"))
        # save datas...
        with open(self.dc_p,"w",encoding="utf-8") as f:
            for cid, docs in cluster2docs.items():
                if len(docs) < self.mindoc:
                    continue
                docs_info = [
                    {
                        "doc_id": d,
                        "doc_time":docid2doctime[d],
                        "doc":docid2doc[d],
                        "degree": degree.get(d, 0)
                    }
                    for d in sorted(docs, key=lambda x: degree.get(x, 0), reverse=True)
                ]
                record = {
                    "cluster_id": cid,
                    "doc_num": len(docs),
                    "docs": docs_info
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n Finish document clustering!! ^w^")
