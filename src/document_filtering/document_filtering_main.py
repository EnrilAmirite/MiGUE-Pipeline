from filtering import doc_filtering



if __name__=='__main__':
    tool=doc_filtering()
    tool.filter_doc_concurrent()
    tool.clean_raw_doc()