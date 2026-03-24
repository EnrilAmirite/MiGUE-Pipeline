from src.event_relation_anno.ev2pair import ev2pair
from src.event_relation_anno.erai import erai
        



if __name__=='__main__':
    ev2pair().create_epair()
    erai().erai_cc()
    erai().era_rev()