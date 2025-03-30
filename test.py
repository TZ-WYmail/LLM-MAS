import json



# 提取 content 字段
content = """```json
{
    "resource_allocation": {
        "action": 0,
        "quantity": {
            "food": 139,
            "water": 139,
            "medical": 0
        },
        "reason": "当前食物和水的需求量均为139，且资源充足（食物247，水332），优先满足基本生存需求可以确保受灾群众的生命安全。医疗需求虽然同样重要，但在食物和水需求未满足的情况下，优先级略低。"
    }
}
```"""
print(content)

# 去除多余的 Markdown 格式
start_marker = "```json"
end_marker = "```"
start_index = content.find(start_marker) + len(start_marker)
end_index = content.find(end_marker, start_index)
json_content = content[start_index:end_index].strip()

# 解析 JSON 格式的内容


parsed_content = json.loads(json_content)
resource_allocation = parsed_content['resource_allocation']
print("提取的 resource_allocation 字典：")
print(resource_allocation)
print("Action:", resource_allocation['action'])
print("Reason:", resource_allocation['reason'])
