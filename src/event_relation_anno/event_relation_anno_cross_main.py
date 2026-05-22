from src.event_relation_anno.ev2pair_cd import ev2pair_cd
from src.event_relation_anno.erac import erac
from src.event_relation_anno.erc_trans import erc_trans



if __name__=='__main__':
    ev2pair_cd=ev2pair_cd()
    ev2pair_cd.create_em_ebd()
    ev2pair_cd.create_cdep()
    erac=erac()
    erac.erac_cc()
    erc_trans=erc_trans()
    erc_trans.filtering_core()
    erc_trans.create_cotrans()
    erc_trans.create_ruletrans()
    erc_trans.final_erc()