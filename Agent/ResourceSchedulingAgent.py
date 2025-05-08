import json
import requests

from Agent.Memory import Memory


class ResourceSchedulingAgent:
    '''重建代理，负责制定资源外部调度计划。'''

    def __init__(self, env, llm):
        '''
        初始化重建代理。
        :param env: 灾害响应环境实例。
        :param llm: 语言模型实例，用于生成调度建议。
        '''
        self.default_resource_action = None
        self.env = env
        self.llm = llm
        self.next_actions = None
        self.Agent_name = 'ResourceSchedulingAgent'
        self.memory = Memory(decay_rate=0.1, long_memory_threshold=3)
        self.gain = 0
        self.memory.file_name = 'memory/ResourceSchedulingAgent.csv'
        self.memory.clean_memory()
        self.default_rebuild_action = {
            'action': 0,  # 默认行动：从外面调动食物和水
            'quantity': {  # 默认资源配置
                'food': 10,  # 默认提供10单位食物
                'water': 10,  # 默认提供10单位水
                'member': 0,
                'medicine': 0,
                'equipment': 0
            },
            'reason': '默认行动：从外面调度食物和水以供灾区使用'
        }

        self.use_memory = True

        self.actions_dict = {
            0: '从外面调度食物和水（供应灾区基础生活资源）',
            1: '从外面调度救援人员、医疗资源和救援装备（供应救灾设备）'
        }

    def next_action(self):
        '''
        获取下一步行动。
        '''
        background = '''
                你是一个专业的资源调度师，负责制定外部资源的调配计划。你的任务是根据灾区需求和现有资源，设计一个高效、合理的资源调度方案。
                '''
        prompt = f'''
                你是一名专业的资源调度师，当前面临以下灾害场景：
                - **灾害强度**：{self.env.state[0]}（数字越大影响越大）
                - **需要医疗救助的群众数量**：{self.env.state[1]}
                - **可用救援装备数量**：{self.env.state[7]}
                - **可用救援人员数量**：{self.env.state[8]}
                - **可用食物资源**：{self.env.state[4]}
                - **可用水资源**：{self.env.state[5]}
                - **食物需求**：{self.env.state[10]}
                - **水需求**：{self.env.state[11]}
                - **医疗需求**：{self.env.state[12]}

                你的任务是选择最佳的资源调配方案：
                0. 从外部调度食物和水（补充灾区基础生活物资，提升食物与水资源储备，满足居民基本生存需求）。
                1.从外部调度救援人员、医疗资源和救援装备（提升救援能力与医疗水平，可支持重建、救援与医疗援助行动）。
                你需要考虑：
                当前食物和水的库存是否足以满足受灾群众的基本需求，如出现短缺，可能导致死亡人数上升。
                医疗资源是否紧缺，特别是当灾区伤病人数较多时，医疗资源直接影响生存率。
                重建与救援行动是否因缺乏人员或装备而无法顺利进行，如有瓶颈，应优先补足相关资源。
                灾区总需求的紧急程度，合理评估生活保障与救援能力之间的平衡，以提高整体灾后响应效率。
                请根据当前状态选择 最优的资源调配方案，最大化资源使用效率并保障灾后应急响应能力。

                你的返回内容要是一个 JSON，格式如下（不要在开头加上```json，末尾也不要加上```），数值填充区域不要有多余的文字,也不要写分析过程：
                {{
                    'resource_allocation': {{
                        'action': <选择的行动编号>,
                        'quantity': {{
                            'food': <要调来的食物数量>,
                            'water': <要调来的水数量>,
                            'medicine': <要调来的成员数量>,
                            'members': <要调来的成员数量>,
                            'equipment': <要调来的装备数量>
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
        file_path = 'ResourceSchedulingAgent.json'
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
        print("提取的 ResourceSchedulingAgent 字典：")
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
