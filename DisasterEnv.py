import gym
import numpy as np
from gym import spaces


class DisasterResponseEnv(gym.Env):
    """
    灾害响应环境类，模拟灾害场景下的救援和资源分配决策过程。
    """

    def __init__(self, duration):
        """
        初始化灾害响应环境。
        :param duration: 模拟的总时间步长。
        """
        super().__init__()
        self.duration = duration
        self.loop = 0  # 当前时间步
        self.action_space_rescue = spaces.Discrete(3)
        self.action_space_resource = spaces.Discrete(5)
        self.action_space_rebuild = spaces.Discrete(3)
        # 定义状态空间：15维状态向量
        self.observation_space = spaces.Box(low=0, high=100, shape=(17,), dtype=np.float32)
        # 灾害强度的动态变化参数
        self.amplitude = 40  # 振幅
        self.frequency = 0.05  # 角频率
        self.offset = 50  # 偏移量

        #灾害临界点，超过了则灾害影响度随机(?)加重
        self.tipping_point = 200 #3.19新增

        #灾害相关的参数
        self.total_number = 1000  # 受灾区域总人数
        self.total_infrastructure = 500
        self.unaffected_number = 0
        #资源上限
        self.max_food = 1000
        self.max_water = 1000
        self.max_medical = 1000
        # 初始化状态
        self.state = self._initialize_state()
        #全部的值均设置为0-1000
        # state = [
        #0     disaster_intensity,       # 灾害影响度
        #1     affected_number,          # 受灾区域待救援人数
        #2     available_food,           # 可用食物资源
        #3     available_water,          # 可用水资源
        #4     available_medical,        # 可用医疗资源
        #5     infrastructure_damage,    # 基础设施损坏数量（决定下一轮的可用资源数，主动）
        #6     population_affected,      # 受灾群众总数量（基础设施修复，避难所数量增加均让人数下降，反之上升）
        #7     medical_needs,            # 医疗需求（待治疗人数，占受灾群众数量30%出现减员）
        #8     food_needs,               # 食物需求（待治疗人数，占受灾群众数量30%出现减员）
        #9     water_needs,              # 水需求 （待治疗人数，占受灾群众数量30%出现减员）
        #10     resettled_residents_number,# 被安置群众数量(不属于受灾群众了)
        #11     weather_conditions,       # 天气条件 （影响基础设施修复数量，避难所增加数量）
        #12     rescue_teams,             # 可用救援队伍数量 （增加会带动三个需求增加，反正减少）
        #13     evacuation_centers,       # 可用避难所数量(天气恶劣时，避难所数量减少)
        #14     number_of_deaths          # 死亡总人数（作为惩罚的一项）
        #15     economic_recovery         # 经济恢复？
        #16     tipping_point_accumulation #灾害临界点累计，如果到达设定的tipping_point数值，灾难将进一步加重

        # ]

    def _initialize_state(self):
        """
        初始化状态向量。
        :return: 初始化后的状态向量。
        """
        # 初始化状态向量，确保每个状态的初始值合理
        state = np.random.randint(200, 600, size=(17,))
        state[5] = np.random.randint(300, 500)  # 基础设施损坏数量
        state[6] = np.random.randint(300, 500)  # 受灾群众总数量
        state[1] = state[6]  # 受灾区域待救援人数初始化为受灾群众总数量
        self.unaffected_number = self.total_number - state[6]
        state[7] = int(state[6] * 0.3)  # 医疗需求
        state[8] = int(state[6] * 0.3)  # 食物需求
        state[9] = int(state[6] * 0.3)  # 水需求
        state[10] = 0  # 被安置群众数量初始化为0
        state[11] = 50  # 天气条件初始化为50
        state[12] = 10  # 可用救援队伍数量初始化为10
        state[13] = 20  # 可用避难所数量初始化为20
        state[14] = 0  # 死亡人数初始化为0
        state[15] = 0 # 灾后相对经济水平初始化为0
        state[16] = 0
        #打印基本信息
        print("初始化状态：")
        print("灾害强度：", state[0])
        print("受灾区域待救援人数：", state[1])
        print("可用食物资源：", state[2])
        print("可用水资源：", state[3])
        print("可用医疗资源：", state[4])
        print("基础设施损坏数量：", state[5])
        print("受灾群众总数量：", state[6])
        print("医疗需求：", state[7])
        print("食物需求：", state[8])
        print("水需求：", state[9])
        print("被安置群众数量：", state[10])
        print("天气条件：", state[11])
        return state

    def step(self, action_rebuild, action_rescue, action_resource):
        """
        执行一步操作，更新环境状态。
        :param action_rebuild: 灾后重建智能体的行动。
        :param action_rescue: 救援智能体的行动。
        :param action_resource: 资源分配智能体的行动。
        :return: 更新后的状态、救援智能体的奖励、资源分配智能体的奖励、是否结束、附加信息。
        """
        reward_rescue = 0
        reward_resource = 0
        reward_rebuild = 0
        done = False

        # 更新灾害强度
        self.state[0] = self._calculate_disaster_intensity(self.loop)
        # 执行灾后重建智能体的行动
        reward_rebuild += self._execute_rebuild_action(action_rebuild)
        # 执行救援智能体的行动
        reward_rescue += self._execute_rescue_action(action_rescue)
        # 执行资源分配智能体的行动
        reward_resource += self._execute_resource_action(action_resource)

        # 动态更新状态
        self._update_state_dynamics()

        # 检查是否结束
        done = self.loop >= self.duration

        return self.state, reward_rescue, reward_resource, reward_rebuild, done, {}

    def _execute_rebuild_action(self, de_action):
        """
        执行灾后重建智能体的行动。
        :param action: 灾后重建智能体的行动。
        :return: 灾后重建智能体的奖励。
        """
        action = de_action['action']
        num = de_action['quantity']
        reward = 0

        if action == 0:  # 建造避难所，消耗食物、水和医疗资源
            if self.state[2] > num and self.state[3] > num and self.state[4] > num:
                self.state[2] -= num
                self.state[3] -= num
                self.state[4] -= num
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)
                self.state[13] += number  # 增加避难所数量
                self.state[6] -= number  # 减少受灾群众数量
                self.state[1] -= number  # 受灾区待救援人数减少
                reward += 10
            else:
                reward -= 5

        elif action == 1:  # 修复医疗站，消耗医疗资源
            if self.state[4] > num and self.state[5] > 0:
                self.state[4] -= num
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)
                self.state[5] -= number
                self.state[7] -= number  # 医疗需求减少
                self.state[10] += number  # 增加被安置群众数量
                self.state[1] -= number  # 受灾区待救援人数减少
                reward += 15
            else:
                reward -= 5

        elif action == 2:  # 强化避难所，提高恶劣天气下的生存能力
            if self.state[2] > num and self.state[3] > num:
                self.state[2] -= num
                self.state[3] -= num
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)
                self.state[13] += number
                self.state[11] += number * 2  # 提高天气适应能力
                reward += 12
            else:
                reward -= 5

        elif action == 3:  # 修建道路，减少基础设施损坏，提高救援能力
            if self.state[5] > 0 and self.state[2] > num:
                self.state[2] -= num
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)
                self.state[5] -= number  # 基础设施损坏减少
                self.state[12] += number  # 增加可用救援队伍数量
                reward += 10
            else:
                reward -= 5

        elif action == 4:  # 维修供水设施
            if self.state[3] > num and self.state[5] > 0:
                self.state[3] -= num
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)
                self.state[5] -= number  # 基础设施损坏减少
                self.state[3] += number * 5  # 增加可用水资源
                self.state[9] -= number * 2  # 降低水需求
                reward += 8
            else:
                reward -= 5

        return reward

    def _execute_rescue_action(self, de_action):
        """
        执行救援智能体的行动。
        :param de_action: 救援智能体的行动。
        :return: 救援智能体的奖励。
        """
        action = de_action['action']
        num = de_action['quantity']
        reward = 0

        if action == 0:  # 搜救行动
            if np.random.rand() > self.state[11] / 500:  # 天气影响搜救成功率
                if self.state[12] >= num and self.state[1] > 0:  # 检查救援队伍和受灾人数
                    self.state[12] -= num  # 消耗救援队伍

                    # 计算成功搜救人数
                    number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)

                    # 避难所容量计算
                    available_shelter = self.state[13] * 4 - self.state[10]
                    if available_shelter > 0:
                        number = min(number, available_shelter // 4)
                    else:
                        number = 0  # 没有空余避难所，无法安置

                    # 限制搜救人数不能超过受灾区待救援人数
                    number = min(number * 4, self.state[1])

                    self.state[1] -= number  # 受灾区待救援人数减少
                    self.state[10] += number  # 增加安置人数

                    # 需求调整（恢复期的需求减少）
                    self.state[7] = max(0, self.state[7] - number * 0.2)  # 医疗需求
                    self.state[8] = max(0, self.state[8] - number * 0.2)  # 食物需求
                    self.state[9] = max(0, self.state[9] - number * 0.2)  # 水需求

                    reward += 10  # 成功搜救奖励
                else:
                    reward -= 20  # 资源不足惩罚
            else:
                reward -= 20  # 天气恶劣导致搜救失败

        elif action == 1:  # 生产救援队伍
            if self.state[2] >= 20 * num and self.state[3] >= 20 * num:  # 资源充足
                self.state[2] -= 20 * num  # 食物消耗
                self.state[3] -= 20 * num  # 水资源消耗
                self.state[12] += num  # 增加救援队伍
                reward += 10  # 生产救援队伍奖励
            else:
                reward -= 10  # 资源不足惩罚

        elif action == 2:  # 医疗救助
            if self.state[4] >= num and self.state[7] > 0:  # 需要医疗资源和待治疗人数
                self.state[4] -= num  # 消耗医疗资源
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)

                # 限制医疗救治人数不超过实际需求
                number = min(number, self.state[7])
                self.state[7] -= number  # 医疗需求减少

                # 如果医疗需求减少，降低死亡率
                self.state[14] = max(0, self.state[14] - number * 0.1)

                reward += 15  # 成功救治奖励
            else:
                reward -= 10  # 资源不足惩罚

        elif action == 3:  # 组织撤离
            if self.state[12] >= num and self.state[1] > 0:  # 需要救援队伍
                self.state[12] -= num  # 消耗救援队伍
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)

                # 限制撤离人数不超过受灾群众
                number = min(number, self.state[1])
                self.state[1] -= number  # 受灾群众减少
                self.state[10] += number  # 安置群众增加

                # 撤离成功降低死亡风险
                self.state[14] = max(0, self.state[14] - number * 0.2)

                reward += 20  # 成功撤离奖励
            else:
                reward -= 15  # 资源不足惩罚

        elif action == 4:  # 搭建临时医疗点
            if self.state[4] >= num and self.state[5] > 0:  # 需要医疗资源和基础设施
                self.state[4] -= num  # 消耗医疗资源
                number = sum(1 for _ in range(num) if np.random.rand() > self.state[0] / 100)
                self.state[5] -= number  # 设施损坏减少
                self.state[7] = max(0, self.state[7] - number * 2)  # 医疗需求减少

                reward += 10  # 额外奖励
            else:
                reward -= 10  # 资源不足惩罚

        return reward

    def _execute_resource_action(self, de_action):
        """
        执行资源分配智能体的行动。
        :param de_action: 资源分配智能体的行动。
        :return: 资源分配智能体的奖励。
        """
        reward = 0
        action = de_action['action']
        resource = de_action['quantity']

        if action == 0:  # 发放食物和水
            food = resource['food']
            water = resource['water']
            if self.state[2] > food and self.state[3] > water:  # 检查是否有足够的食物和水
                if self.state[8] > 0 and self.state[9] > 0:  # 如果还有需求
                    self.state[2] -= food  # 减少食物资源
                    self.state[3] -= water  # 减少水资源
                    self.state[8] -= food * 1.5  # 减少食物需求
                    self.state[9] -= water * 1.5  # 减少水需求
                    reward += 10  # 成功发放资源奖励
                    print(f"食物需求：{self.state[8]}, 水需求：{self.state[9]}")
                else:
                    reward -= 10  # 没有需求的惩罚
            else:
                reward -= 5  # 资源不足惩罚

        elif action == 1:  # 医疗援助
            medical = resource['medical']
            if self.state[4] > medical:  # 检查是否有足够的医疗资源
                if self.state[7] > 0:  # 如果还有医疗需求
                    self.state[4] -= medical  # 减少医疗资源
                    self.state[7] -= medical * 1.5  # 减少医疗需求
                    reward += 15  # 成功提供医疗援助奖励
                    print(f"医疗需求：{self.state[7]}")
                else:
                    reward -= 10  # 没有需求的惩罚
            else:
                reward -= 10  # 资源不足的惩罚

        elif action == 2:  # 资源生产：组织物资采集（增加食物和水）
            workers = resource['workers']
            if self.state[6] > workers:  # 检查是否有足够的受灾群众参与采集
                self.state[6] -= workers  # 投入受灾群众进行采集
                number = sum(1 for _ in range(workers) if np.random.rand() > self.state[0] / 100)  # 受灾强度影响采集成功率
                self.state[2] += number * 3  # 增加食物资源
                self.state[3] += number * 3  # 增加水资源
                self.state[6] += workers  # 完成采集后归还受灾群众
                reward += 10  # 成功采集的奖励
            else:
                reward -= 5  # 资源不足惩罚

        elif action == 3:  # 资源储备：建立物资储存点，减少资源浪费
            if self.state[2] >= 10 and self.state[3] >= 10:  # 检查是否有足够的食物和水资源
                self.state[2] -= 10  # 消耗食物
                self.state[3] -= 10  # 消耗水
                self.state[5] -= 2  # 消耗基础设施
                self.state[2] += 20  # 增加食物储存
                self.state[3] += 20  # 增加水储存
                reward += 15  # 建立储存点奖励
            else:
                reward -= 10  # 资源不足的惩罚

        elif action == 4:  # 能源生产：搭建发电设施，增强灾后恢复能力
            if self.state[2] >= 20 and self.state[3] >= 10:  # 需要食物和水支持建设
                self.state[2] -= 20  # 消耗食物
                self.state[3] -= 10  # 消耗水
                number = sum(1 for _ in range(10) if np.random.rand() > self.state[0] / 100)  # 受灾强度影响建设成功率
                self.state[5] -= number  # 基础设施恢复
                self.state[11] += number * 2  # 提升天气适应能力
                reward += 12  # 成功搭建发电设施奖励
            else:
                reward -= 8  # 资源不足的惩罚

        return reward

    def _calculate_disaster_intensity(self, t):
        """
        计算灾害强度的动态变化。
        :param t: 当前时间步。
        :return: 计算后的灾害强度。
        """
        return self.amplitude * np.sin(self.frequency * t) + self.offset

    def _update_state_dynamics(self):
        """
        动态更新状态，模拟环境的自然变化。
        """
        #本轮人数跟新
        #安置情况判断
        if self.state[10] > self.state[13] * 4:
            self.state[1] += self.state[10] - self.state[13] * 4
            self.state[10] = self.state[13] * 4
        #死亡人数跟新
        level = 0
        if self.state[7] >= self.state[10] * 0.25:
            level += 1
        if self.state[8] >= self.state[10] * 0.3:
            level += 1
        if self.state[9] >= self.state[10] * 0.35:
            level += 1
        #根据灾害等级， 受灾群众总数量占区域总人数概率的决定死亡人数
        if level == 1:
            self.state[14] += int(level * self.state[10] * 0.05 * np.random.rand())
        elif level == 2:
            self.state[14] += int(level * self.state[10] * 0.1 * np.random.rand())
        elif level == 3:
            self.state[14] += int(level * self.state[10] * 0.15 * np.random.rand())

        """
              根据当前状态统计减员情况。
              减员规则：
              - 医疗需求超过受灾群众数量的50%时，减员人数为超出部分的50%。
              - 食物需求或水需求超过受灾群众数量的30%时，减员人数为超出部分的30%。
              """
        population_affected = self.state[6]  # 受灾群众总数量
        medical_needs = self.state[7]  # 医疗需求
        food_needs = self.state[8]  # 食物需求
        water_needs = self.state[9]  # 水需求

        # 初始化减员人数
        casualties = 0

        # 医疗需求导致的减员
        if medical_needs > population_affected * 0.5:
            excess_medical = medical_needs - population_affected * 0.5
            casualties += excess_medical * 0.05  # 超出部分的50%作为减员

        # 食物需求导致的减员
        if food_needs > population_affected * 0.3:
            excess_food = food_needs - population_affected * 0.3
            casualties += excess_food * 0.03  # 超出部分的30%作为减员

        # 水需求导致的减员
        if water_needs > population_affected * 0.3:
            excess_water = water_needs - population_affected * 0.3
            casualties += excess_water * 0.03  # 超出部分的30%作为减员

        self.state[14] += casualties
        self.state[6] -= casualties  # 更新受灾群众总数量

        #跟新灾害强度
        self.loop += 1
        self.state[0] = self._calculate_disaster_intensity(self.loop)

        # 灾害强度影响受灾群众数量和需求
        number = int(self.state[0] * 0.1)
        if self.total_number - number < 0:
            number = self.total_number
        self.state[6] += number  # 灾害强度增加受灾群众数量
        self.total_number -= number
        # 灾害强度影响基础设施损坏
        self.state[5] += int(self.state[0] * 0.1) * np.random.rand()  # 灾害强度增加基础设施损坏数量

        # 安置情况判断
        if self.state[10] > self.state[13] * 4:
            self.state[1] += self.state[10] - self.state[13] * 4
            self.state[10] = self.state[13] * 4

        #根据基础设施损坏数量，决定资源生产情况
        self.state[2] += int(self.total_infrastructure - self.state[5])
        self.state[3] += int(self.total_infrastructure - self.state[5])
        self.state[4] += int(self.total_infrastructure - self.state[5])

        #跟新需求
        self.state[7] += self.state[10] * 0.1 + self.state[1] * np.random.rand() * 0.8
        self.state[8] += self.state[10] * 0.1 + self.state[1] * np.random.rand() * 0.8
        self.state[9] += self.state[10] * 0.1 + self.state[1] * np.random.rand() * 0.8
        #跟新天气
        self.state[11] += int(np.random.rand() * 10)

        # 经济恢复提升影响可用资源
        if self.state[15] > 50:
            self.state[2] += 5  # 增加食物
            self.state[3] += 5  # 增加水
            self.state[4] += 2  # 增加医疗资源
            self.state[12] += 1  # 增加救援队伍

        # 如果灾害临界点过高，灾害影响度增加
        if self.state[16] > self.tipping_point:
            self.state[0] += 10
            self.state[5] += 5
            self.state[6] += 50
            self.state[15] -= 5

        #打印输出本轮结果
        print("第", self.loop, "轮结果：")
        print("灾害强度：", self.state[0])
        print("受灾区域待救援人数：", self.state[1])
        print("可用食物资源：", self.state[2])
        print("可用水资源：", self.state[3])
        print("可用医疗资源：", self.state[4])
        print("基础设施损坏数量：", self.state[5])
        print("受灾群众总数量：", self.state[6])
        print("医疗需求：", self.state[7])
        print("食物需求：", self.state[8])
        print("水需求：", self.state[9])
        print("被安置群众数量：", self.state[10])
        print("天气条件：", self.state[11])
        print("可用救援队伍数量：", self.state[12])
        print("可用避难所数量：", self.state[13])
        print("死亡人数：", self.state[14])
        print("----------------------------------")

    def reset(self):
        """
        重置环境，返回初始状态。
        """
        self.state = self._initialize_state()
        self.loop = 0
        return self.state
