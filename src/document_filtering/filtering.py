from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from load_df_data import load_df_data


class doc_filtering(load_df_data):
    def __init__(self):
        super().__init__()

    def process_single_row(self,idx):
        try:
            raw_doc = self.df.loc[idx,'doc']
            vote_results = {}
            vote_num = 0
            for eval_type in ["factual","text","event"]:
                sys_prom,prompt=self.load_filter_prompt(
                    type=eval_type,
                    raw_text=raw_doc
                )
                #instant
                answer=self.llm.call_llm(
                    system_prompt=sys_prom,
                    user_prompt=prompt
                )
                vote_results[f'vote_{eval_type}']=answer
                if answer=='P':
                    vote_num+=1
            vote_final = 'Pass' if vote_num >= 3 else 'Fail'
            with self.df_lock:
                for col, val in vote_results.items():
                    self.df.loc[idx, col] = val
                self.df.loc[idx, "vote_final"] = vote_final
                self.df.loc[idx, "processed"] = 1
                return True
        except Exception as e:
            print(f"\n Something is wrong *ﾟДﾟ* :\n row id: {idx}\n error: {e}")
            return False

    def filter_doc_concurrent(self):
        for col in ["vote_text", "vote_factual", "vote_event", "vote_final"]:
            if col not in self.df.columns:
                self.df[col] = ""
            self.df[col] = self.df[col].astype(object)
        if "processed" not in self.df.columns:
            self.df["processed"] = 0
        todo_indices = self.df[self.df["processed"] == 0].index.tolist()
        total_tasks = len(todo_indices)
        if total_tasks == 0:
            print("All documents have already checked ^o^ ~")
            return
        print(f"Start concurrent checking !\n total number of tasks: {total_tasks}\n number of threads: {self.max_workers}")
        with tqdm(total=total_tasks, desc="Evaluating raw documents...o(^ω^)o") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_row, idx): idx for idx in todo_indices
                }            
                for i, future in enumerate(as_completed(futures)):
                    pbar.update(1)
                    if i%10 == 0:
                        with self.df_lock:
                            self.df.to_csv(self.imd_path, index=False, encoding='utf-8')
                with self.df_lock:
                    self.df.to_csv(self.imd_path, index=False, encoding='utf-8')               
        print(f"\n All documents have already checked ^o^ ~")
    
    def clean_raw_doc(self):
        self.keep_csv_row(
            self.df,
            self.op_path,
            'vote_final',
            'Pass'
        )



