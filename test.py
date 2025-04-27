import json

# 原始 JSON 格式的字符串
json_string = '''
{
    "resource_allocation": {
        "action": 0,
        "reason": "当前食物、水和医疗资源充足，优先满足受灾群众的基本生存需求（食物和水），以稳定局势并为后续行动创造条件。",
        "quantity": {
            "food": 126,
            "water": 126,
            "medical": 0
        }
    }
}
'''

# 使用 json.loads() 解析字符串
try:
    data_dict = json.loads(json_string)
    print("解析后的字典：")
    print(data_dict)
except json.JSONDecodeError as e:
    print(f"解析 JSON 时出错：{e}")