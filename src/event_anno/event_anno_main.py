from src.event_anno.eva_ebd import eva_ebd
from src.event_anno.doc2stc import doc2stc
from src.event_anno.eva import eva
from src.event_anno.clean_eva import clean_eva

if __name__=='__main__':
    doc2stc().split_doc2stc()
    eva_ebd().create_case_ebd()
    eva_ebd().create_stc_ebd_cc()
    eva().eva_cc()
    clean_eva().create_event()
    clean_eva().create_docNem_index()


