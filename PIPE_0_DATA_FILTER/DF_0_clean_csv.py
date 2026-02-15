import pandas as pd
import csv
import sys
import re

def clean_csv(input_file, output_file):
    # 1. 扩大 Python 能够处理的最大字段限制
    csv.field_size_limit(sys.maxsize)

    print(f"开始处理文件: {input_file}...")

    # 2. 增强版清洗函数
    def remove_illegal_and_newlines(text):
        if not isinstance(text, str):
            return text
        
        # A. 将换行符（\r, \n）替换为中文句号
        # 
        text = re.sub(r'[\r\n]+', '。', text)
        
        # B. 移除其他不可见的非法控制字符 (ASCII 0-31 中除去必要的)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # C. 清洗细节：处理可能出现的连续句号（比如原文末尾有句号又加了换行）
        text = re.sub(r'[。]+', '。', text)
        
        return text.strip()

    try:
        # 3. 使用 Python 引擎读取
        df = pd.read_csv(
            input_file, 
            encoding='utf-8-sig', # 建议用 sig，兼容性更好
            engine='python', 
            on_bad_lines='warn'
        )

        # 4. 执行清洗
        print("正在清理非法字符并将换行符替换为句号...")
        for col in df.columns:
            if df[col].dtype == 'object':  # 只处理文本列
                df[col] = df[col].apply(remove_illegal_and_newlines)

        # 5. 保存结果
        df.to_csv(
            output_file, 
            index=False, 
            encoding='utf-8-sig', 
            quoting=csv.QUOTE_ALL 
        )

        print(f"清洗完成！已生成文件: {output_file}")
        print(f"有效行数: {len(df)}")

    except Exception as e:
        print(f"处理过程中发生严重错误: {e}")


import pandas as pd

def sort_csv_by_id(input_file, output_file, ascending=True):
    """
    将 CSV 文件按 id 列进行排序
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param ascending: True 为升序，False 为降序
    """
    try:
        print(f"正在读取文件: {input_file}")
        # 读取文件，指定 engine='python' 以增加对复杂长文本的兼容性
        df = pd.read_csv(input_file, encoding='utf-8-sig', engine='python')

        if 'id' not in df.columns:
            print(f"错误：文件中未找到 'id' 列。现有列名为: {df.columns.tolist()}")
            return

        # 1. 确保 id 列是数值类型 (防止 1, 10, 2 这种字符串排序错误)
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        
        # 2. 检查是否有非法 id (非数字)
        if df['id'].isnull().any():
            print("警告：部分行的 id 不是有效的数字，这些行将被移动到末尾。")

        # 3. 执行排序
        print(f"正在按 id 进行{'升序' if ascending else '降序'}排序...")
        df = df.sort_values(by='id', ascending=ascending)

        # 4. 保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=1) # quoting=1 即 QUOTE_ALL
        print(f"排序完成！已保存至: {output_file}")

    except Exception as e:
        print(f"排序过程中发生错误: {e}")

# --- 调用示例 ---
if __name__ == "__main__":
    sort_csv_by_id("cleaned_data.csv", "sorted_data.csv")


input_path="DATA/raw_data/raw_docs_TCE_2022_clean.csv"
output_path="DATA/raw_data/raw_docs_TCE_2022.csv"

if __name__ == "__main__":
    #clean_csv(input_path, output_path)
    sort_csv_by_id(input_file=input_path, output_file=output_path, ascending=True)