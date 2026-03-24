import os
from pathlib import Path
from datetime import datetime

class PathManager:
    def __init__(self, task_name, base_dir="data"):
        # 自动生成带时间戳的运行目录，防止数据覆盖
        self.run_id = datetime.now().strftime("%m%d_%H%M")
        self.root = Path(base_dir) / task_name / self.run_id
        
        # 预定义各个阶段的路径
        self.input_path = Path("raw_data/input.csv") # 原始输入通常固定
        self.imd_path = self.root / "intermediate" / "checkpoint.csv"
        self.output_path = self.root / "result" / "final_output.csv"
        self.log_path = self.root / "logs" / "process.log"
        
        # 核心：一键创建所有文件夹
        self._make_dirs()
        print(f"🚀 Project initialized at: {self.root}")

    def _make_dirs(self):
        # 遍历所有路径属性，如果是文件，就创建它的父文件夹
        for attr in self.__dict__.values():
            if isinstance(attr, Path) and not attr.suffix == "":
                attr.parent.mkdir(parents=True, exist_ok=True)

# 使用方式
paths = PathManager("DocFiltering")
print(paths.imd_path) # 输出: data/DocFiltering/0520_1430/intermediate/checkpoint.csv