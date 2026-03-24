import yaml
import json

def yaml_to_json(yaml_file, json_file):
    # 1. 读取 YAML
    with open(yaml_file, 'r', encoding='utf-8') as f:
        # Loader=yaml.FullLoader 是为了安全考虑
        data = yaml.load(f, Loader=yaml.FullLoader)

    # 2. 写入 JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        # indent=4 让生成的 JSON 易读，ensure_ascii=False 保证中文不乱码
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"转换成功: {yaml_file} -> {json_file}")

# 使用示例
yaml_to_json('config/user_sample.yaml', 'config/user.json')