import json
import requests

from Agent.Memory import Memory


class GovernmentAgent:
    """政府代理，负责制定灾害响应政策。"""

    def __init__(self, env, llm):
        """
        初始化政府代理。
        :param env: 灾害响应环境实例。
        :param llm: 语言模型实例，用于生成政策文本。
        """
        self.env = env
        self.llm = llm
        self.next_actions = None
        self.RebuildingAgent = None
        self.ResourceManagementAgent = None
        self.RescueAgent = None
        self.Agent_name = 'GovernmentAgent'
        self.memory = Memory(decay_rate=0.1, long_memory_threshold=3)
        self.gain = 0
        self.memory.file_name = 'memory/GovernmentAgent.csv'
        self.agent_action = None
        self.use_memory = True

    def next_action(self):
        """
        获取下一步行动。
        """
        global reason
        background = """
                你是一个专业的政策制定者，负责为政府制定灾害响应政策。你的任务是根据灾害场景生成有效的应急响应政策，确保政策内容涵盖救援行动、资源分配、重建计划和公众沟通等方面。
                你的政策需要清晰、具体，并且能够指导实际操作。
                """
        prompt = f"""
                你是一名政策制定者，当前面临以下灾害场景：
                - 灾害强度：{self.env.state[0]}（影响救援和重建的难度）
                - 受灾区域待救援人数：{self.env.state[1]}（需要优先救援的人数）
                - 可用食物资源：{self.env.state[2]}（单位：份，用于满足受灾群众的食物需求）
                - 可用水资源：{self.env.state[3]}（单位：升，用于满足受灾群众的水需求）
                - 可用医疗资源：{self.env.state[4]}（单位：份，用于满足受灾群众的医疗需求）
                - 基础设施损坏数量：{self.env.state[5]}（影响资源生产和分配效率）
                - 受灾群众总数量：{self.env.state[6]}（包括待救援和已安置的群众）
                - 医疗需求：{self.env.state[7]}（大于于受灾群众数量的50%会出现减员，需要医疗援助的人数）
                - 食物需求：{self.env.state[8]}（大于于受灾群众数量的30%会出现减员，需要食物的人数）
                - 水需求：{self.env.state[9]}（大于于受灾群众数量的30%会出现减员，需要水的人数）
                - 被安置群众数量：{self.env.state[10]}（已安置到避难所的人数）
                - 天气条件：{self.env.state[11]}（影响救援和重建效率，数值越低越恶劣）
                - 可用救援队伍数量：{self.env.state[12]}（可用于搜救的队伍数量）
                - 可用避难所数量：{self.env.state[13]}（可用于安置受灾群众的避难所数量）
                - 死亡人数：{self.env.state[14]}（当前已死亡的人数，作为政策制定的参考）

                你的任务是综合考虑当前灾害情况，制定一个全面的灾害响应政策，明确调用以下智能体的行动：
                - **救援行动**：根据受灾区域待救援人数和可用救援队伍数量，决定是否调用救援代理（RescueAgent）。
                - **资源分配**：根据受灾群众的需求（食物、水、医疗）和现有资源，决定是否调用资源管理代理（ResourceManagementAgent）。
                - **重建计划**：根据基础设施损坏数量和可用避难所数量，决定是否调用重建代理（RebuildingAgent）。

                请根据当前状态制定一个全面且可行的灾害响应政策，并以 JSON 格式返回，格式如下，数值填充区域不要有多余的文字：
                {{
                    "actions": {{
                        "rescue_action": <调用救援代理的行动建议>,
                        "Agent_name": <调用智能体的名称>(可以有多个，逗号隔开，不要有多余的文字),
                        "Agent_action": <调用智能体的原因>,
                    }}
                }}
                """

        memory_prompt = self.get_memory_prompt()
        print("memory_prompt:" + memory_prompt)
        messages = [
            {"role": "system", "content": background},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.generate_response(messages)
        response_dict = json.loads(response)
        print("Generated Policy:", response_dict)

        # 提取 Agent_name
        content = response_dict['choices'][0]['message']['content']
        start_marker = "Agent_name\": \""
        end_marker = "\","

        # 找到标记位置
        start_index = content.find(start_marker) + len(start_marker)
        end_index = content.find(end_marker, start_index)

        # 提取 Agent_name
        agent_names = content[start_index:end_index].strip()
        agent_list = [agent.strip() for agent in agent_names.split(",")]

        print("需要调用的智能体：", agent_list)
        self.next_actions = agent_list
        #提取Agent_action
        content = response_dict['choices'][0]['message']['content']
        start_marker = "Agent_action\": \""
        end_marker = "\""
        # 找到标记位置
        start_index = content.find(start_marker) + len(start_marker)
        end_index = content.find(end_marker, start_index)
        # 提取 Agent_action
        agent_actions = content[start_index:end_index].strip()
        agent_action_list = [agent.strip() for agent in agent_actions.split(",")]
        print("调用智能体的原因：", agent_action_list)

        return agent_list

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

    def action_get(self, agent_name, agent_action):
        if agent_name == 'RescueAgent':
            self.agent_action['RescueAgent'] = agent_action
        elif agent_name == 'ResourceManagementAgent':
            self.agent_action['ResourceManagementAgent'] = agent_action
        elif agent_name == 'RebuildingAgent':
            self.agent_action['RebuildingAgent'] = agent_action
        else:
            print('智能体名称错误')
