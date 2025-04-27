import json
import requests

from Agent.Memory import Memory


class ResourceManagementAgent:
    """资源管理代理，负责优化资源分配。"""

    def __init__(self, env, llm):
        """
        初始化资源管理代理。
        :param env: 灾害响应环境实例。
        :param llm: 语言模型实例，用于生成资源分配建议。
        """
        self.env = env
        self.llm = llm
        self.next_actions = None
        self.Agent_name = 'ResourceManagementAgent'
        self.memory = Memory(decay_rate=0.1, long_memory_threshold=3)
        self.memory_prompt = ""
        self.gain = 0
        self.memory.file_name = 'memory/ResourceManagementAgent.csv'
        self.memory.clean_memory()
        self.default_resource_action = {
            "action": 0,  # 默认行动：发放食物和水
            "quantity": {  # 默认资源分配数量
                "food": 10,  # 发放10单位食物
                "water": 10,  # 发放10单位水
                "medical": 0  # 不分配医疗资源
            },
            "reason": "默认行动：优先满足食物和水需求，以维持基本生存。"
        }
        self.use_memory = True
        self.actions_dict = {
            0: "发放食物和水（每发放一个单位消耗食物1单位、水1单位，减少受灾群众的食物和水需求）",
            1: "提供医疗援助（每提供一个单位消耗医疗资源1单位，减少医疗需求，提高生存率）",
            2: "组织物资采集（每组织一个单位消耗劳动力1单位，增加食物和水资源，受灾害强度影响成功率）",
            3: "建立物资储存点（每建立一个单位消耗食物10单位、水10单位、基础设施2单位，提高资源存储能力，减少浪费）",
            4: "搭建发电设施（每搭建一个单位消耗食物20单位、水10单位，促进基础设施恢复，增强天气适应能力）"
        }

    def next_action(self):
        """
        获取下一步行动。
        """
        self.memory_prompt = self.get_memory_prompt()
        background = """
                你是一名资源管理专家，负责在灾害场景中优化资源分配。你的任务是根据政策文本中的建议，制定具体的资源分配计划，确保资源能够高效利用，满足救援和重建的需求。
                """
        prompt = f"""
                你是一名资源管理专家，当前面临以下灾害场景：
                - **可用食物资源**：{self.env.state[2]}
                - **可用水资源**：{self.env.state[3]}
                - **可用医疗资源**：{self.env.state[4]}
                - **食物需求**：{self.env.state[8]}
                - **水需求**：{self.env.state[9]}
                - **医疗需求**：{self.env.state[7]}
                - **可用劳动力**：{self.env.state[6]}（用于采集资源）
                - **基础设施状况**：{self.env.state[5]}（影响资源储备和能源生产）
                - **天气适应能力**：{self.env.state[11]}（影响能源生产效率）
                
                ### 你的任务是选择最佳的资源分配行动：
                0. **发放食物和水**（每发放一个单位消耗食物1单位、水1单位，减少受灾群众的食物和水需求）。
                1. **提供医疗援助**（每提供一个单位消耗医疗资源1单位，减少医疗需求，提高生存率）。
                2. **组织物资采集**（每组织一个单位消耗劳动力1单位，增加食物和水资源，受灾害强度影响成功率）。
                3. **建立物资储存点**（每建立一个单位消耗食物10单位、水10单位、基础设施2单位，提高资源存储能力，减少浪费）。
                4. **搭建发电设施**（每搭建一个单位消耗食物20单位、水10单位，促进基础设施恢复，增强天气适应能力）。  
                
                ### 你需要考虑：
                - **当前资源是否足够满足需求**，避免不必要的消耗。
                - **优先满足哪些需求可以最大程度减少受灾群众的损失**，比如医疗、食物和水需求。
                - **是否需要采集更多资源**，以确保未来的物资供给，特别是增加食物和水的库存。
                - **是否应当建立储存点**，以减少资源浪费并提高长期供应能力。
                - **是否需要搭建发电设施**，以促进灾区恢复并增强天气适应能力。
                
                请根据当前状态选择 **最优的资源分配行动**，最大化资源利用效率，并确保灾后恢复工作顺利进行。

                你的返回内容要是一个 JSON，格式如下（不要在开头加上```json，末尾也不要加上```），数值填充区域不要有多余的文字,也不要写分析过程：
                {{
                    'resource_allocation': {{
                        'action': <选择的行动编号>,
                        'reason': <选择该行动的原因>,
                        'quantity': {{
                            'food': <发放的食物数量>,
                            'water': <发放的水数量>,
                            'medical': <提供的医疗资源数量>
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
        file_path='ResourceManagementAgent.json'
        # 将 response_dict 保存到文件
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(json_string, file, ensure_ascii=False, indent=4)

        print(f"数据已保存到 {file_path}")

        # 从文件加载数据
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_dict = json.load(file)
        data_dict = json.loads(loaded_dict)
        resource_allocation = data_dict["resource_allocation"]
        # 打印提取的结果
        print("提取的 ResourceManagementAgent 字典：")
        print('action:', resource_allocation.get('action'))
        print('quantity:', resource_allocation.get('quantity'))
        print('reason:', resource_allocation.get('reason'))
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
