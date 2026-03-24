
import spacy

# 加载中文模型（需要先 pip install spacy 并下载模型）
nlp = spacy.load("zh_core_web_sm")

def get_shared_entities(sent1, sent2):
    # 1. 提取实体
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)
    
    # 2. 存入集合（取实体的文本内容）
    ents1 = {ent.text for ent in doc1.ents}
    ents2 = {ent.text for ent in doc2.ents}
    
    # 3. 求交集
    shared = ents1.intersection(ents2)
    
    # 4. 判断逻辑：是否有两个以上相同元素
    if len(shared) >= 2:
        return True, shared
    return False, shared

# 测试
s1 = "2023年，小明在北京遇见了张三。"
s2 = "张三昨天也去了北京，但他没看到小明。"
found, entities = get_shared_entities(s1, s2)
print(f"匹配结果: {found}, 共同元素: {entities}")