


from data_filter_function_design import raw_doc_json2csv_gemini
path_input="DATA/raw_data/MIDEAST_doc_2022_raw.json"
path_output="DATA/raw_data/raw_docs_TCE_2022.csv"




if __name__=='__main__':
    raw_doc_json2csv_gemini(
        path_input=path_input,
        path_output=path_output
    )