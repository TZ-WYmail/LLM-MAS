import json
import requests

from Agent.Memory import Memory


class RebuildingAgent:
    '''重建代理，负责制定灾后重建计划。'''

    def __init__(self, env, llm):
        '''
        初始化重建代理。
        :param env: 灾害响应环境实例。
        :param llm: 语言模型实例，用于生成重建建议。
        '''
        self.default_resource_action = None
        self.env = env
        self.llm = llm
        self.next_actions = None
        self.Agent_name = 'RebuildingAgent'
        self.memory = Memory(decay_rate=0.1, long_memory_threshold=3)
        self.gain = 0
        self.memory.file_name = 'memory/RebuildingAgent.csv'
        self.memory.clean_memory()
        self.default_rebuild_action = {
            'action': 0,  # 默认行动：建造避难所
            'quantity': {  # 默认资源配置
                'members': 1,  # 默认需要1个建设成员
                'equipment': 1  # 默认需要1个建设工具
            },
            'reason': '默认行动：建造避难所以增加安置能力。'
        }

        self.use_memory = True

        self.actions_dict = {
            0: '建造避难所（救援装备按数量消耗，概率决定建造成功数量，增加避难所容量）',
            1: '修复基础设施（每修复一个单位消耗救援人员1单位、救援装备1单位，概率决定修复成功数量，减少基础设施损坏）',
            2: '污染治理（每治理一个单位消耗工作成员1单位、装备1单位，概率决定治理成功数量，减少污染）'
        }

    def next_action(self):
        '''
        获取下一步行动。
        '''
        background = '''
                你是一个专业的重建规划师，负责制定灾后重建计划。你的任务是根据灾害场景和现有资源，设计一个高效、可持续的重建方案，包括基础设施修复、住房重建、经济复苏和社会服务恢复等方面。
                '''
        prompt = f'''
                你是一名专业的重建规划师，当前面临以下灾害场景：
                - **灾害强度**：{self.env.state[0]}（影响救援行动成功率）
                - **未安置群众数量**：{self.env.state[1]}
                - **可用救援装备数量**：{self.env.state[7]}
                - **可用救援人员数量**：{self.env.state[8]}
                - **避难所可容纳人数量**：{self.env.state[9]}
                - **可用食物资源**：{self.env.state[4]}
                - **可用水资源**：{self.env.state[5]}
                - **食物需求**：{self.env.state[10]}
                - **水需求**：{self.env.state[11]}
                - **污染度**：{self.env.state[14]}（影响医疗援助成功率，污染度过高会影响食物资源和水资源）
                - **天气条件**：{self.env.state[13]}（影响搜救成功率）

                你的任务是选择最佳的重建行动：
                0. **建造避难所**（救援装备按数量消耗，增加避难所容量，如果未安置群众数量较多而避难所可容纳人数不足，请考虑是否要建造避难所）。
                1. **修复基础设施**（需要足够的救援人员，每修复1个单位消耗救援装备1单位，成功率与天气条件相关，减少基础设施损坏，每点基础设施可以稳定地提供1单位水和1单位食物）。
                2. **污染治理**（需要足够的救援人员，每治理1个单位消耗装备1单位，成功率与天气条件、污染度和灾害影响度相关，治理无成效可能会导致污染度加重，污染度过高会使可用水资源与食物资源数量下降）。

                你需要考虑：
                - **当前资源是否足够支持重建行动**，避免不必要的资源浪费。
                - **天气条件是否适合进行重建**
                - **优先修复基础设施还是建造避难所**，以最大程度减少受灾群众的损失，并提高灾后恢复效率。
                - **污染治理是否迫在眉睫**，污染度大于60的时候食物资源与水资源会被影响，如果污染度并不高，可以暂时先置之不理。

                请根据当前状态选择 **最优的重建行动**，并最大化重建效率。

                你的返回内容要是一个 JSON，格式如下（不要在开头加上```json，末尾也不要加上```），数值填充区域不要有多余的文字,也不要写分析过程：
                {{
                    'resource_allocation': {{
                        'action': <选择的行动编号>,
                        'quantity': {{
                            'members': <使用的成员数量>,
                            'equipment': <使用的装备数量>
                        }},
                        'reason': '选择该行动的原因'
                    }}
                }}
        '''

        memory_prompt = self.get_memory_prompt()
        if self.use_memory:
            prompt += f'\n\n历史经验：{memory_prompt}'
        messages = [
            {'role': 'system', 'content': background},
            {'role': 'user', 'content': prompt}
        ]
        response = self.llm.generate_response(messages)
        response_dict = json.loads(response)
        content = response_dict["choices"][0]["message"]["content"]
        # 将单引号替换为双引号
        json_string = content.replace("'", '"')
        file_path = 'RebuildingAgent.json'
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
        print("提取的 RebuildingAgent 字典：")
        print('action:', resource_allocation.get('action'))
        print('quantity:', resource_allocation.get('quantity'))
        print('reason:', resource_allocation.get('reason'))
        self.next_actions = resource_allocation.get('action', {})
        return resource_allocation

    def update_memory(self):
        '''
        更新记忆，记录当前行动的结果并自动处理短期记忆和长期记忆。
        :param next_actions: 采取的行动（字典格式，包含 action 和 quantity）。
        :param gain: 行动的收益（正或负）。
        '''
        evi = tuple(self.env.state)  # 将当前环境状态作为记忆的证据
        # 将当前行动和收益记录到短期记忆中
        self.memory.update_memory(self.next_actions, evi, self.gain)

    def get_memory_prompt(self):
        '''
        根据当前环境状态，从记忆中获取参考行动，并返回一个记忆提示。
        :return: 包含推荐行动和原因的记忆提示字符串。
        '''
        current_evi = tuple(self.env.state)  # 当前环境状态
        reference_action = self.memory.get_memory_prompt(current_evi)

        if reference_action is not None:
            # 如果找到了参考行动，生成一个记忆提示
            memory_prompt = (
                f'根据历史经验行动编号：{reference_action["action"]}时。\n'
                f'环境状态：{reference_action["evi"]}\n'
                f'历史收益：{reference_action["gain"]:.2f}\n'
                f'验证次数：{reference_action["count"]}'
            )
        else:
            # 如果没有找到合适的参考行动，提供一个默认提示
            memory_prompt = (
                '没有找到合适的参考行动。建议根据当前环境需求和资源情况，'
                '优先考虑满足最紧急的需求（如医疗需求），并确保资源分配的合理性。'
            )
        return memory_prompt
