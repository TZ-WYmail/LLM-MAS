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
        self.default_rebuild_action = {
            'action': 0,  # 默认行动：建造避难所
            'quantity': 1,  # 默认数量：建造1个避难所
            'reason': '默认行动：建造避难所以增加安置能力。'
        }
        self.use_memory = True
        self.actions_dict = {
            0: '建造避难所（每建造一个单位消耗食物10单位、水5单位、医疗资源1单位，概率决定建造成功数量，增加避难所容量）',
            1: '修复医疗站（每修复一个单位消耗医疗资源1单位，概率决定修复成功数量，减少基础设施损坏，增加被安置群众数量）',
            2: '强化避难所（每强化一个单位消耗食物1单位、水1单位，概率决定强化成功数量，提高避难所容量和天气适应能力）',
            3: '修建道路（每修建一个单位消耗食物1单位，概率决定修建成功数量，减少基础设施损坏，提高救援队行动能力）',
            4: '维修供水设施（每维修一个单位消耗水1单位，概率决定维修成功数量，修复基础设施，增加可用水资源）'
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
                - 基础设施损坏数量：{self.env.state[5]}
                - 可用避难所数量：{self.env.state[13]}
                - 可用食物资源：{self.env.state[2]}
                - 可用水资源：{self.env.state[3]}
                - 可用医疗资源：{self.env.state[4]}
                - 天气条件：{self.env.state[11]}（影响重建效率）
                
                你的任务是选择最佳的重建行动：
                0. **建造避难所**（每建造一个单位消耗食物10单位、水5单位、医疗资源1单位，概率决定建造成功数量，增加避难所容量）。
                1. **修复医疗站**（每修复一个单位消耗医疗资源1单位，概率决定修复成功数量，减少基础设施损坏，增加被安置群众数量）。
                2. **强化避难所**（每强化一个单位消耗食物1单位、水1单位，概率决定强化成功数量，提高避难所容量和天气适应能力）。
                3. **修建道路**（每修建一个单位消耗食物1单位，概率决定修建成功数量，减少基础设施损坏，提高救援队行动能力）。
                4. **维修供水设施**（每维修一个单位消耗水1单位，概率决定维修成功数量，修复基础设施，增加可用水资源）。  
                
                你需要考虑：
                - **当前资源是否足够支持重建行动**，避免不必要的资源浪费。
                - **天气条件是否适合进行重建**，比如天气适应能力（`self.env.state[11]`）对重建效率的影响。
                - **优先修复基础设施还是建造避难所**，以最大程度减少受灾群众的损失，并提高灾后恢复效率。
                
                请根据当前状态选择 **最优的重建行动**，并最大化重建效率。
                
                你的返回内容要是一个 JSON，格式如下（不要在开头加上```json，末尾也不要加上```），数值填充区域不要有多余的文字,也不要写分析过程：
                {{
                    'resource_allocation': {{
                        'action': <选择的行动编号>,
                        'quantity': <建造的具体数量>,
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
