from src.base_code.ConfigGenerator import ConfigGenerator
from src.base_code.load_config import load_config
import subprocess
import sys
import os

class MiGUE(ConfigGenerator):
    def __init__(self):
        ConfigGenerator.__init__(self)

    def run_scripts(self,script_list):
        for script in script_list:
            print(f"\n Currently executing part: {script} ^w^...")
            result = subprocess.run([sys.executable, script], check=True)
            if result.returncode == 0:
                print(f"\n Successfully completed {script} part !^.^\n")
            else:
                print(f"\nThere is something wrong in {script} part!")
                break

    def MiGUE_Pipeline(self):
        check=self.checkNsave_cfg()
        if check:
            scripts_to_run=[]
            for stts in self.pipeline:
                if stts.startswith("event_relation_anno"):
                    script_path = os.path.join("src","event_relation_anno",f"{stts}_main.py")
                else:
                    script_path = os.path.join("src",stts,f"{stts}_main.py")
                    
                scripts_to_run.append(script_path)
            self.run_scripts(scripts_to_run)