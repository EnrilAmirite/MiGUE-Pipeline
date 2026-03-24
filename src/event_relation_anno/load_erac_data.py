import pandas as pd
import json
from src.event_relation_anno.load_erac_config import load_erac_config
from src.event_relation_anno.ev2pair_cd import ev2pair_cd

class record_status(load_erac_config):
    def __init__(self):
        super().__init__()
        with open(self.rule_p,'r',encoding='utf-8') as f:
            self.rule_f=json.load(f)
        #question list(relation)
        self.rel_list=["TEMPORAL","CAUSAL","SUBEVENT","COREFERENCE"]
        #relations have answered
        self.ra_list=[]
        #tags have answered
        self.ta_list=[]
        #rule
        self.must_not={
                "TEMPORAL":[],
                "CAUSAL":[],
                "SUBEVENT":[],
                "COREFERENCE":[]
            }
        self.must={
                "TEMPORAL":[],
                "CAUSAL":[],
                "SUBEVENT":[],
                "COREFERENCE":[]    
            }
        self.ans={
            "TEMPORAL":"",
            "CAUSAL":"",
            "SUBEVENT":"",
            "COREFERENCE":""
        }
        #option list
        self.opt_list=[]

    
    def clean_status(self):
        #question list(relation)
        self.rel_list=["TEMPORAL","CAUSAL","SUBEVENT","COREFERENCE"]
        #relations have answered
        self.ra_list=[]
        #tags have answered
        self.ta_list=[]
        #rule
        self.must_not={
                "TEMPORAL":[],
                "CAUSAL":[],
                "SUBEVENT":[],
                "COREFERENCE":[]
            }
        self.must={
                "TEMPORAL":[],
                "CAUSAL":[],
                "SUBEVENT":[],
                "COREFERENCE":[]    
            }
        self.ans={
            "TEMPORAL":"",
            "CAUSAL":"",
            "SUBEVENT":"",
            "COREFERENCE":""
        }
        #option list
        self.opt_list=[]
        
    def maintain_have_answer(self,rel,tag):
        #relation
        self.ra_list.append(rel)
        #tag
        self.ta_list.append(tag)
        
    def maintain_must_and_must_not(self,rel,tag):
        delete_tag=self.rule_f[rel][tag]["no"]
        add_tag=self.rule_f[rel][tag]["must"]
        #must not
        self.must_not["CAUSAL"].extend(delete_tag["CAUSAL"])
        self.must_not["COREFERENCE"].extend(delete_tag["COREFERENCE"])
        self.must_not["SUBEVENT"].extend(delete_tag["SUBEVENT"])
        self.must_not["TEMPORAL"].extend(delete_tag["TEMPORAL"])
        #must
        self.must["CAUSAL"].extend(add_tag["CAUSAL"])
        self.must["COREFERENCE"].extend(add_tag["COREFERENCE"])
        self.must["SUBEVENT"].extend(add_tag["SUBEVENT"])
        self.must["TEMPORAL"].extend(add_tag["TEMPORAL"])
        return 
    
    def upload_ans(self,rel,table):
        self.ans[rel]=table


class load_erac_data(ev2pair_cd):
    def __init__(self):
        super().__init__() 
        with open(self.opt_p,'r',encoding='utf-8') as f:
            self.opt_f=json.load(f)
        with open(self.pp_p,'r',encoding='utf-8') as f:
            self.pp_f=json.load(f)
        with open(self.rule_p,'r',encoding='utf-8') as f:
            self.rule_f=json.load(f)
        with open(self.rre_p,'r',encoding='utf-8') as f:
            self.relre_f=json.load(f)
        with open(self.des_p,'r',encoding='utf-8') as f:
            self.des_f=json.load(f)
        self.sys_pp=self.pp_f['system']

    def create_intro_pp(self,row):
        doc_a=row['doc_a']
        doc_b=row['doc_b']
        em_a=row['em_a']
        em_b=row['em_b']
        off_a=self.str2list(row['offset_a'])
        off_b=self.str2list(row['offset_b'])
        em_a_h=self.highlight(em_a,off_a)
        em_b_h=self.highlight(em_b,off_b)
        pp=""
        for k,v in self.pp_f['definition'].items():
            pp+=v
        pp+=self.pp_f['give']['task'][4]
        pp+=self.pp_f['give']['task'][5]
        pp+=doc_a
        pp+=self.pp_f['give']['task'][6]
        pp+=doc_b
        pp+=self.pp_f['give']['task'][2]
        pp+=em_a_h
        pp+=self.pp_f['give']['task'][3]
        pp+=em_b_h                
        return pp

    def create_rel_pp(self,status: record_status):
        rel_pp=""
        if len(status.ra_list)!=0:
            rel_pp+=self.pp_f['give']['rel'][0]
            for ra in status.ra_list:
                ta = status.ans[ra]
                rel_pp+=self.des_f[ra][ta]['text']
                rel_pp+='\n'
            rel_pp+=self.pp_f['give']['rel'][1]
        rel_pp+=self.pp_f['give']['rel'][2]
        #delete have answered relation
        filtered_options = {
            key: v for key, v in self.opt_f['RELATION_CHOOSE'].items()
            if v["relation"] not in status.ra_list
        }
        #options
        rel_pp+="\n".join([
            f"{key}: {item['text']}"
            for key, item in filtered_options.items()
        ])
        rel_pp+=self.pp_f['op_ctrl']['rel'][0]
        option_list=",".join([
            f"{key}"
            for key in filtered_options
        ])
        rel_pp+=option_list
        rel_pp+=self.pp_f['op_ctrl']['all'][0]
        status.opt_list = [key for key in filtered_options]
        return rel_pp

    #status is an instant
    def create_label_pp(self,rel,status: record_status):
        labels=self.opt_f[rel]
        lb_pp=self.pp_f['give']['label'][0]
        lb_pp+=rel
        lb_pp+=self.pp_f['give']['label'][1]
        #Delete logically conflicting options
        lb_json={
            key:v for key,v in labels.items()
            if v["name"] not in status.must_not[rel]
        }
        lb_pp+="\n".join([
            f"{key}:{item["text"]}"
            for key,item in lb_json.items()
        ])
        lb_pp+=self.pp_f['op_ctrl']['label'][0]
        opt_list=",".join([
            f"{key}"
            for key in lb_json
        ])
        lb_pp+=opt_list
        lb_pp+=self.pp_f['op_ctrl']['all'][0]
        status.opt_list= [key for key in lb_json]
        return lb_pp
    

    def after_thinking_pp(self,rel,label,status: record_status):
        pp_give=""
        pp_give+=self.pp_f['give']['aftt'][0]
        pp_give+=rel
        pp_give+=self.pp_f['give']['aftt'][1]
        pp_give+=label
        must=self.rule_f[rel][label]['must']
        must_not=self.rule_f[rel][label]["no"]
        pp=""
        must_list=[]
        #must have.......
        #if rules have no 'must have' then pass
        if all(len(v)==0 for key,v in must.items()):
            pp+=""
        else: 
            pp+=self.pp_f['give']['aftt'][2]
            thinking_must={
                key:v for key,v in must.items()
                if len(v)!=0
            }
            for key,v_list in thinking_must.items():#is a list
                must_list.extend(key)
                pp+=self.pp_f['give']['aftt'][3]
                pp+=key
                pp+=self.pp_f['give']['aftt'][4]
                for v in v_list:
                    des=self.des_f[key][v]["text"]
                    pp+=f"{des}\n"
        #must not choose
        if all(len(v)==0 for key,v in must_not.items()):
            pp+=""
        else:
            pp+=self.pp_f['give']['aftt'][5]
            #if 'must' has described, then pass
            thinking_must_not={
                key:v for key,v in must_not.items()
                if len(v)!=0 and key not in must_list
            }
            for key,v_list in thinking_must_not.items():
                pp+=self.pp_f['give']['aftt'][6]
                pp+=key
                pp+=self.pp_f['give']['aftt'][7]
                for v in v_list:
                    des=self.des_f[key][v]["text"]
                    pp+=f"{des}\n"
        if pp!="":
            pp+=self.pp_f["op_ctrl"]["aftt"][0]
            pp+="\n".join([
                f"{k}:{v["text"]}"
                for k,v in self.opt_f["ACTION_CHOOSE"].items()
            ])
            #output control
            pp+=self.pp_f["op_ctrl"]["aftt"][1]
            pp+=",".join([
                f"{k}"
                for k,v in self.opt_f["ACTION_CHOOSE"].items()
            ])
            pp_give+=pp
            pp_give+=self.pp_f["op_ctrl"]["all"][0]
        else: return ""
        return pp_give
    
    #relation option to real relation
    #eg. REL_1-->TEMPORAL
    def opt2rel(self,opt):
        mapping=self.opt_f["RELATION_CHOOSE"]
        relation=mapping[opt]["relation"]
        return relation

    #label option to real label
    #eg. TEMP_1--->BEFORE,TEMP_2--->AFTER
    def opt2label(self,rel,opt):  
        mapping=self.opt_f[rel]
        label=mapping[opt]["name"]
        return label
