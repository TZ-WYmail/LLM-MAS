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
        self.default_rescue_action = {
            "action": 0,  # 默认行动：执行搜救任务
            "quantity": 1,  # 默认数量：使用1个救援队伍
            "reason": "默认行动：执行搜救任务以减少受灾区域待救援人数。"
        }
        self.use_memory = True
        self.actions_dict = {
            0: "执行搜救行动（每执行一个单位消耗救援队伍1单位，根据天气和灾害强度决定成功率，搜救受灾群众并安置）",
            1: "生产救援队伍（每生产一个单位消耗食物20单位、水20单位，增加可用救援队伍数量）",
            2: "医疗救助行动（每执行一个单位消耗医疗资源1单位，优先救治伤员，减少医疗需求）",
            3: "组织撤离（每执行一个单位消耗救援队伍1单位，减少受灾群众数量，降低死亡风险）",
            4: "搭建临时医疗点（每搭建一个单位消耗医疗资源1单位、基础设施1单位，提高医疗救治效率）"
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
                - **受灾区域待救援人数**：{self.env.state[1]}
                - **可用救援队伍数量**：{self.env.state[12]}
                - **医疗资源**：{self.env.state[4]}
                - **食物资源**：{self.env.state[2]}
                - **水资源**：{self.env.state[3]}
                - **庇护所容量**：{self.env.state[13]}（影响安置能力）
                - **天气条件**：{self.env.state[11]}（影响搜救成功率）
                
                ### 你的任务是选择最佳的救援行动：
                0. **执行搜救行动**（每执行一个单位消耗救援队伍1单位，根据天气和灾害强度决定成功率，搜救受灾群众并安置）。
                1. **生产救援队伍**（每生产一个单位消耗食物20单位、水20单位，增加可用救援队伍数量）。
                2. **医疗救助行动**（每执行一个单位消耗医疗资源1单位，优先救治伤员，减少医疗需求）。
                3. **组织撤离**（每执行一个单位消耗救援队伍1单位，减少受灾群众数量，降低死亡风险）。
                4. **搭建临时医疗点**（每搭建一个单位消耗医疗资源1单位、基础设施1单位，提高医疗救治效率）。
                
                ### 你需要考虑：
                - **救援队伍的数量** 是否足够执行搜救、撤离等任务，避免资源不足。
                - **天气条件** 是否适合搜救行动，避免高风险任务失败，尤其在恶劣天气下。
                - **食物和水资源** 是否足够支持生产新的救援队伍，以便继续救援。
                - **医疗资源** 是否充足，是否应优先用于救治伤员或提供其他医疗服务。
                - **庇护所容量** 是否足够安置新搜救到的受灾群众，避免过多群众无法安置。
                
                请根据当前状态选择 **最优的救援行动**，最大化救援效率和成功率。
                
                你的返回内容要是一个 JSON，格式如下（不要在开头加上```json，末尾也不要加上```），数值填充区域不要有多余的文字,也不要写分析过程：
                {{
                    'resource_allocation': {{
                        'action': <选择的行动编号>,
                        'quantity': <具体数量>,
                        'reason': '选择该行动的原因'
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
            print(loaded_dict)
        resource_allocation = None
        # 使用 json.loads() 解析字符串
        try:
            data_dict = json.loads(loaded_dict)
            resource_allocation = data_dict["resource_allocation"]
            print("解析后的字典：")
            print(resource_allocation)
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错：{e}")
        # 打印提取的结果
        print("提取的 resource_allocation 字典：")
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
