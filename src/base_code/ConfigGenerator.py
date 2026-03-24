from src.base_code.load_data import load_data
import yaml
import os
from pathlib import Path

class ConfigGenerator(load_data):
    def __init__(self):
        load_data.__init__(self)
        with open('config/user.yaml','r',encoding='utf-8') as f:
            self.user_cfg=yaml.safe_load(f)
        self.rawpath=self.user_cfg['raw_document_path']
        self.base_dir=Path(self.rawpath).parent.parent
        self.rawname=Path(self.rawpath).stem
        self.pipeline=self.user_cfg['pipeline']

    def get_standard_config(self,status):
        match status:
            case 'document_filtering':
                return {
                    "storage": {
                        "input": {
                            "raw": {
                                "folder": f"{self.base_dir}/raw_data",
                                "name": f"{self.rawname}.csv"
                            }
                        },
                        "imd": {
                            "filter": {
                                "folder": f"{self.base_dir}/doc_filtering/imd",
                                "name": f"{self.rawname}_imd.csv"
                            }
                        },
                        "output": {
                            "filtered": {
                                "folder": f"{self.base_dir}/doc_filtering/out",
                                "name": f"{self.rawname}_doc.csv"
                            }
                        }
                    },
                    "prompt": {
                        "zh": {
                            "pp": {
                                "folder": "prompt",
                                "name": "doc_filtering_zh.json"
                            }
                        },
                        "en": {
                            "pp": {
                                "folder": "prompt",
                                "name": ""
                            }
                        }
                    }
            }

            case 'event_anno':
                return {
                    "top_k": 2,
                    "storage": {
                        "input": {
                            "doc": {
                                "folder": f"{self.base_dir}/doc_filtering/out",
                                "name": f"{self.rawname}_doc.csv"
                            }
                        },
                        "imd": {
                            "stc": {
                                "folder": f"{self.base_dir}/event_anno/imd",
                                "name": f"{self.rawname}_sentences.csv"
                            },
                            "docNem": {
                                "folder": f"{self.base_dir}/event_anno/imd",
                                "name": f"{self.rawname}_docNem.csv"
                            },
                            "eva": {
                                "folder": f"{self.base_dir}/event_anno/imd",
                                "name": f"{self.rawname}_eva.csv"
                            }
                        },
                        "output": {
                            "event": {
                                "folder": f"{self.base_dir}/event_anno/out",
                                "name": f"{self.rawname}_event.csv"
                            }
                        },
                        "ebd": {
                            "stc_ebd": {
                                "folder": f"{self.base_dir}/event_anno/ebd",
                                "name": f"{self.rawname}_stc_ebd.pkl"
                            },
                            "case_ebd": {
                                "folder": "case/ebd",
                                "name": "case_ebd.pkl"
                            }
                        }
                    },
                    "prompt": {
                        "zh": {
                            "pp": {
                                "folder": "prompt",
                                "name": "event_anno_zh.json"
                            }
                        },
                        "en": {
                            "pp": {
                                "folder": "prompt",
                                "name": ""
                            }
                        }
                    },
                    "case": {
                        "zh": {
                            "case": {
                                "folder": "case",
                                "name": "event_anno_zh.json"
                            }
                        },
                        "en": {
                            "case": {
                                "folder": "case",
                                "name": ""
                            }
                        }
                    }
                }
            case 'event_relation_anno_cross':
                return {
                    "top_k": 20,
                    "threshold": 0.8,
                    "storage": {
                        "input": {
                            "event": {
                                "folder": f"{self.base_dir}/event_anno/out",
                                "name": f"{self.rawname}_event.csv"
                            },
                            "doc": {
                                "folder": f"{self.base_dir}/doc_filtering/out",
                                "name": f"{self.rawname}_doc.csv"
                            },
                            "eri": {
                                "folder": f"{self.base_dir}/event_relation_anno_intra/output",
                                "name": f"{self.rawname}_eri.jsonl"
                            }
                        },
                        "imd": {
                            "docNem": {
                                "folder": f"{self.base_dir}/event_anno/imd",
                                "name": f"{self.rawname}_docNem.csv"
                            },
                            "ep": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/imd",
                                "name": f"{self.rawname}_epc.csv"
                            },
                            "era": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/imd",
                                "name": f"{self.rawname}_erac_uni.jsonl"
                            },
                            "core": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/imd",
                                "name": f"{self.rawname}_cdcore.jsonl"
                            },
                            "cotrans": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/imd",
                                "name": f"{self.rawname}_cotrans.jsonl"
                            },
                            "reltrans": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/imd",
                                "name": f"{self.rawname}_reltrans.jsonl"
                            }
                        },
                        "ebd": {
                            "em_ebd": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/ebd",
                                "name": f"{self.rawname}_em_ebd.pkl"
                            },
                            "emid2idx": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/ebd",
                                "name": f"{self.rawname}_emid2idx.index"
                            },
                            "emid": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/ebd",
                                "name": f"{self.rawname}_emid.pkl"
                            },
                            "idx2ebd": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/ebd",
                                "name": f"{self.rawname}_idx2ebd.pkl"
                            }
                        },
                        "output": {
                            "erc": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/output",
                                "name": f"{self.rawname}_erc.jsonl"
                            }
                        }
                    },
                    "prompt": {
                        "zh": {
                            "des": {
                                "folder": "prompt",
                                "name": "relation_des_zh.json"
                            },
                            "opt": {
                                "folder": "prompt",
                                "name": "relation_options_zh.json"
                            },
                            "pp": {
                                "folder": "prompt",
                                "name": "relation_anno_zh.json"
                            }
                            },
                        "en": {
                            "des": {
                                "folder": "prompt",
                                "name": "relation_des_en.json"
                            },
                            "opt": {
                                "folder": "prompt",
                                "name": "relation_options_en.json"
                            },
                            "pp": {
                                "folder": "prompt",
                                "name": "relation_anno_en.json"
                            }
                        }
                    },
                    "rule": {
                        "rule": {
                            "folder": "rule",
                            "name": "relation_rules.json"
                        },
                        "rel_rvs": {
                            "folder": "rule",
                            "name": "relation_reverse.json"
                        },
                        "rel_trans": {
                            "folder": "rule",
                            "name": "relation_transmission.json"
                        }
                    }
                }
            case 'event_relation_anno_intra':
                return {
                    "storage": {
                        "input": {
                            "event": {
                                "folder": f"{self.base_dir}/event_anno/out",
                                "name": f"{self.rawname}_event.csv"
                            },
                            "doc": {
                                "folder": f"{self.base_dir}/doc_filtering/out",
                                "name": f"{self.rawname}_doc.csv"
                            }
                        },
                        "imd": {
                            "ep": {
                                "folder": f"{self.base_dir}/event_relation_anno_intra/imd",
                                "name": f"{self.rawname}_evp.csv"
                            },
                            "era": {
                                "folder": f"{self.base_dir}/event_relation_anno_intra/imd",
                                "name": f"{self.rawname}_era_uni.jsonl"
                            }
                        },
                        "output": {
                            "eri": {
                                "folder": f"{self.base_dir}/event_relation_anno_intra/output",
                                "name": f"{self.rawname}_eri.jsonl"
                            }
                        }
                    },
                    "prompt": {
                        "zh": {
                            "des": {
                                "folder": "prompt",
                                "name": "relation_des_zh.json"
                            },
                            "opt": {
                                "folder": "prompt",
                                "name": "relation_options_zh.json"
                            },
                            "pp": {
                                "folder": "prompt",
                                "name": "relation_anno_zh.json"
                            }
                        },
                        "en": {
                            "des": {
                                "folder": "prompt",
                                "name": "relation_des_en.json"
                            },
                            "opt": {
                                "folder": "prompt",
                                "name": "relation_options_en.json"
                            },
                            "pp": {
                                "folder": "prompt",
                                "name": "relation_anno_en.json"
                            }
                        }
                    },
                    "rule": {
                        "rule": {
                            "folder": "rule",
                            "name": "relation_rules.json"
                        },
                        "rel_rvs": {
                            "folder": "rule",
                            "name": "relation_reverse.json"
                        }
                    }
                }
            case 'document_clustering':
                return {
                    "storage": {
                        "input": {
                            "cdcore": {
                                "folder": f"{self.base_dir}/event_relation_anno_cross/imd",
                                "name": f"{self.rawname}_cdcore.jsonl"
                            },
                            "doc": {
                                "folder": f"{self.base_dir}/doc_filtering/out",
                                "name": f"{self.rawname}_doc.csv"
                            }
                        },
                        "imd": {
                            "coreline": {
                                "folder": f"{self.base_dir}/document_filtering/imd",
                                "name": f"{self.rawname}_coreline.jsonl"
                            }
                        },
                        "output": {
                            "dc": {
                                "folder": f"{self.base_dir}/document_filtering/output",
                                "name": f"{self.rawname}_dc.jsonl"
                            }
                        }
                    }
                }
            case 'user':
                return {
                    "language": "zh",
                    "max_workers": 8,
                    "document_path": 'null',
                    "print_prompt": 'null',
                    "print_llm_reply": 'null',
                    "document_filtering": {
                        "llm": "4omini"
                    },
                    "event_annotation": {
                        "llm": "null",
                        "type": "reflection",
                        "embedding_llm": "null"
                    },
                    "event_relation_annotation_intra": {
                        "llm": "null",
                        "type": "reflection"
                    },
                    "event_relation_annotation_cross": {
                        "llm": "null",
                        "type": "reflection"
                    },
                    "document_clustering": {
                        "min_core_num": 2,
                        "min_cluster_doc": 2
                    },
                    "pipeline":[
                            "document_filtering",
                            "event_annotation",
                            "event_relation_annotation_intra",
                            "event_relation_annotation_cross",
                            "document_clustering"
                        ]
                }


    def save_config_as_yaml(self,status,save_path):
        config_data = self.get_standard_config(status)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, sort_keys=False)

    def checkNsave_cfg(self):
        if not self.checkfile('config/document_clustering.yaml'):
            for stts in ["document_filtering","event_anno", "event_relation_anno_intra","event_relation_anno_cross","document_clustering"]:
                save_path=os.path.join("config",f"{stts}.yaml")
                self.save_config_as_yaml(stts,save_path)
            print('\nGenerate all config ~ ^w^')
            return 0
        return 1
    
