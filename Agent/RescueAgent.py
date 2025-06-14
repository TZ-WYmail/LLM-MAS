import json
import requests
import ast
from Agent.Memory import Memory


class RescueAgent:
    """救援代理，负责协调救援行动。"""

    def __init__(self, env, llm):
        """
        初始化救援代理。
        :param env: 灾害响应环境实例。
        :param llm: 语言模型实例，用于生成救援建议。
        """
        self.default_resource_action = None
        self.env = env
        self.llm = llm
        self.next_actions = None
        self.Agent_name = 'RescueAgent'
        self.memory = Memory(decay_rate=0.1, long_memory_threshold=3)
        self.gain = 0
        self.memory.file_name = 'memory/RescueAgent.csv'
        self.memory.clean_memory()
        self.default_rescue_action = {
            "action": 0,  # 默认行动：派出救援队搜救未被救出人员
            "quantity": {
                "members": 1,  # 默认出动1名救援队员
                "equipment": 1,  # 默认配备1单位救援装备
                "medical": 0  # 医疗资源仅在医疗行动中使用
            },
            "reason": "默认行动：执行搜救任务提升生存机会。"
        }
        self.use_memory = True
        self.actions_dict = {
            0: "派出救援队搜救未被救出人员（每出动一人消耗1单位救援人员和1单位装备，成功率受天气影响，可能找到活人也可能找到死人）",
            1: "提供紧急医疗援助（每名医疗队成员配合1单位医疗资源，治疗伤员，减少医疗需求）"
        }

    def next_action(self):
        """
        获取下一步行动。
        """
        background = """
                你是一名经验丰富的救援指挥官，负责协调救援行动。你的任务是根据灾害场景制定高效的救援计划，确保最大限度地减少人员伤亡和财产损失。你需要考虑救援队伍的部署、紧急物资分配和现场安全措施。
                """
        prompt = f"""
                你是一名救援行动指挥官，当前面临以下灾害场景：
                - **灾害强度**：{self.env.state[0]}（影响救援行动成功率）
                - **未安置群众数量**：{self.env.state[1]}
                - **已安置群众数量**：{self.env.state[2]}
                - **可用医疗资源**：{self.env.state[6]}
                - **可用救援装备数量**：{self.env.state[7]}
                - **可用救援人员数量**：{self.env.state[8]}
                - **避难所可容纳人数量**：{self.env.state[9]}
                - **可用食物资源**：{self.env.state[4]}
                - **可用水资源**：{self.env.state[5]}
                - **食物需求**：{self.env.state[10]}
                - **水需求**：{self.env.state[11]}
                - **污染度**：{self.env.state[14]}（影响医疗援助成功率，污染度过高会影响食物资源和水资源）
                - **天气条件**：{self.env.state[13]}（影响搜救成功率）

                ### 你的任务是选择最佳的救援行动：
                0. **执行搜救行动**（每出动一人消耗1单位救援人员和1单位装备，成功率受天气影响，可能找到活人也可能找到死人）。
                1. **医疗救助行动**（每执行一个单位消耗医疗资源1单位，优先救治伤员，减少医疗需求）。

                ### 你需要考虑：
                - **是否要搜救** 搜救会提高人员生存率，如果不搜救，有可能仍存活的人最终因为没有得到救助而死去。
                - **可用救援人员数量和可用救援装备数量** 是否足够执行搜救等任务，避免资源不足。
                - **天气条件** 是否适合搜救行动，避免高风险任务失败，尤其在恶劣天气下。
                - **食物和水资源、食物需求和水需求、污染度** 是否要选择进行医疗援助，以免污染度恶化导致可用的食物和水资源下降
                - **医疗资源，未安置群众数量** 是否要优先救助伤员。

                请根据当前状态选择 **最优的救援行动**，最大化救援效率和成功率。

                你的返回内容要是一个 JSON，格式如下（不要在开头加上```json，末尾也不要加上```），数值填充区域不要有多余的文字,也不要写分析过程：
                {{
                    'resource_allocation': {{
                        'action': <选择的行动编号>,
                        'reason': '选择该行动的原因',
                        "quantity": {{
                                "members": <安排的救援人员数量>
                                "equipment": <安排的救援装备数量>
                                "medical": <提供的医疗资源数量>
                                }}
                    }}
                }}
                """

        memory_prompt = self.get_memory_prompt()
        print("memory_prompt:" + memory_prompt)
        if self.use_memory:
            prompt += f"\n\n历史经验：{memory_prompt}"
        messages = [
            {"role": "system", "content": background},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.generate_response(messages)
        response_dict = json.loads(response)
        content = response_dict["choices"][0]["message"]["content"]

        # 将单引号替换为双引号
        json_string = content.replace("'", '"')
        file_path = 'RescueAgent.json'
        # 将 response_dict 保存到文件
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(json_string, file, ensure_ascii=False, indent=4)

        print(f"数据已保存到 {file_path}")

        # 从文件加载数据
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_dict = json.load(file)
        resource_allocation = None
        # 使用 json.loads() 解析字符串
        try:
            data_dict = json.loads(loaded_dict)
            resource_allocation = data_dict["resource_allocation"]
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错：{e}")
        # 打印提取的结果
        print("提取的 RescueAgent 字典：")
        print('action:',resource_allocation.get('action'))
        print('quantity:',resource_allocation.get('quantity'))
        print('reason:',resource_allocation.get('reason'))
        self.next_actions = resource_allocation.get('action', {})
        return resource_allocation

    def update_memory(self):
        """
        更新记忆，记录当前行动的结果并自动处理短期记忆和长期记忆。
        :param next_actions: 采取的行动（字典格式，包含 action 和 quantity）。
        :param gain: 行动的收益（正或负）。
        """
        evi = tuple(self.env.state)  # 将当前环境状态作为记忆的证据
        # 将当前行动和收益记录到短期记忆中
        self.memory.update_memory(self.next_actions, evi, self.gain)

    def get_memory_prompt(self):
        """
        根据当前环境状态，从记忆中获取参考行动，并返回一个记忆提示。
        :return: 包含推荐行动和原因的记忆提示字符串。
        """
        current_evi = tuple(self.env.state)  # 当前环境状态
        reference_action = self.memory.get_memory_prompt(current_evi)

        if reference_action is not None:
            # 如果找到了参考行动，生成一个记忆提示
            memory_prompt = (
                f"根据历史经验行动编号：{reference_action['action']}时。\n"
                f"环境状态：{reference_action['evi']}\n"
                f"历史收益：{reference_action['gain']:.2f}\n"
                f"验证次数：{reference_action['count']}"
            )
        else:
            # 如果没有找到合适的参考行动，提供一个默认提示
            memory_prompt = (
                "没有找到合适的参考行动。建议根据当前环境需求和资源情况，"
                "优先考虑满足最紧急的需求（如医疗需求），并确保资源分配的合理性。"
            )
        return memory_prompt
