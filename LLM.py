import json
import os
import re

from openai import OpenAI


class LLMClient:
    """
    封装语言模型客户端，用于生成文本响应。
    """

    def __init__(self):
        """
        初始化语言模型客户端。

        :param api_key: OpenAI API Key
        :param base_url: OpenAI API 的基础 URL
        :param model_name: 使用的语言模型名称，默认为 'qwen-plus'
        """
        self.api_key = "sk-2505ce4643044eaab9a653a1f58752bb"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = "qwen-plus"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_response(self, messages):
        """
        使用指定的语言模型生成响应。

        :param messages: 与 LLM 交互的消息列表，格式为 [{"role": "system", "content": "..."}, ...]
        :return: LLM 生成的响应（JSON 格式）
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return completion.model_dump_json()  # 返回生成的响应（JSON 格式）
        except Exception as e:
            return json.dumps({"error": str(e)})

    def __str__(self):
        return f"LLMClient(model={self.model_name}, base_url={self.base_url})"

    def parse_json_string(self, json_content):
        """
        使用正则表达式和字符串操作解析 JSON 格式的字符串。
        """
        try:
            # 提取 action
            action_match = re.search(r'"action":\s*(\d+)', json_content)
            action = int(action_match.group(1)) if action_match else None

            # 提取 quantity
            quantity_match = re.search(r'"quantity":\s*({[^}]+})', json_content)
            quantity_str = quantity_match.group(1) if quantity_match else "{}"
            quantity = {}
            for item in quantity_str.split(","):
                key, value = item.split(":")
                quantity[key.strip().strip('"')] = int(value.strip())

            # 提取 reason
            reason_match = re.search(r'"reason":\s*"([^"]+)"', json_content)
            reason = reason_match.group(1) if reason_match else ""

            # 构造字典
            return {
                "action": action,
                "quantity": quantity,
                "reason": reason
            }
        except Exception as e:
            print(f"字符串解析错误：{e}")
            return {}

    def extract_resource_allocation(self, content):
        """
        从字符串中提取 resource_allocation 的内容并转换为字典。
        """
        try:
            # 提取 action
            action_start_marker = "action\": \""
            action_end_marker = "\""
            action_start_index = content.find(action_start_marker) + len(action_start_marker)
            action_end_index = content.find(action_end_marker, action_start_index)
            action = int(content[action_start_index:action_end_index].strip())


            # 提取 quantity
            quantity_start_marker = "quantity\": \""
            quantity_end_marker = "\""
            quantity_start_index = content.find(quantity_start_marker) + len(quantity_start_marker)
            quantity_end_index = content.find(quantity_end_marker, quantity_start_index)
            quantity_str = content[quantity_start_index:quantity_end_index].strip()
            quantity_dict = {}
            for item in quantity_str.split(","):
                key, value = item.split(":")
                quantity_dict[key.strip().strip('"')] = int(value.strip())

            # 提取 reason
            reason_start_marker = "reason\": \""
            reason_end_marker = "\""
            reason_start_index = content.find(reason_start_marker) + len(reason_start_marker)
            reason_end_index = content.find(reason_end_marker, reason_start_index)
            reason = content[reason_start_index:reason_end_index].strip()

            # 组合成字典
            resource_allocation_dict = {
                "action": action,
                "quantity": quantity_dict,
                "reason": reason
            }

            return resource_allocation_dict
        except Exception as e:
            print(f"解析错误：{e}")
            return None

# 示例：使用封装后的 LLMClient 类
if __name__ == "__main__":
    api_key = "sk-2505ce4643044eaab9a653a1f58752bb"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name = "qwen-plus"

    # 初始化 LLM 客户端
    llm_client = LLMClient()

    # 定义消息列表
    messages = [
        {"role": "system", "content": "你是一个专业的政策制定者，负责为政府制定灾害响应政策。"},
        {"role": "user", "content": "你是谁？"}
    ]

    # 调用 LLM 生成响应
    response = llm_client.generate_response(messages)

    # 将 JSON 字符串解析为 Python 字典
    response_dict = json.loads(response)

    # 提取并打印 content 中的内容
    content = response_dict.get("choices", [])[0].get("message", {}).get("content", "No content found.")
    print(content)