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
        # 定义状态空间：16维状态向量
        self.observation_space = spaces.Box(low=0, high=100, shape=(16,), dtype=np.float32)
        # 灾害强度的动态变化参数
        self.amplitude = 40  # 振幅
        self.frequency = 0.05  # 角频率
        self.offset = 50  # 偏移量
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

        # state = [
        # 0	disaster_intensity,	#灾害影响度
        # 1	unrescued people,	#还没救出来的人
        # 2	resettled_residents_number,	#已被安置群众数量（累计）
        # 3	infrastructure_damage,	#基础设施损坏数量（修好了会定量提供食物和水资源）

        # 4	available_food,	#可用食物资源
        # 5	available_water,	#可用水资源
        # 6	available_medical,	#可用医疗资源
        # 7	available_rescue_resource,	#可用救援装备
        # 8	available_rescue_member,	#可用救援人员数量
        # 9	available_center,	    #避难所可容纳人数量

        #需求
        # 10	food_needs,
        # 11	water_needs,
        # 12	medical_needs,

        #客观条件（0-100，对于天气数字越高则越好，对于污染数字越低则越好）
        # 13	weather_conditions,	#天气条件（影响救援成功率）
        # 14	pollution,          #污染（从死亡人数计算，死亡人数过多则污染概率更大，每轮随机）

        #影响量
        # 15	number_of_deaths    #死亡总人数（作为惩罚的一项）
        # ]

    def _initialize_state(self):
        """
        初始化状态向量。
        :return: 初始化后的状态向量。
        """
        # 初始化状态向量，确保每个状态的初始值合理
        state = np.random.randint(200, 600, size=(16,))
        state[2] = 0
        if self.total_infrastructure < state[3]:
            state[3] = self.total_infrastructure
        state[10] = state[1] * 1.5
        state[11] = state[1] * 1.5
        state[13] = (state[13] - 200) / (600 - 200) * 100
        state[14] = (state[14] - 200) / (600 - 200) * 100
        state[15] = 0
        #打印基本信息
        print("初始化状态：")
        print("灾害强度：", state[0])
        print("受灾区域待救援人数：", state[1])
        print("已被安置群众数量（累计）：", state[2])
        print("基础设施损坏数量：", state[3])

        print("可用食物资源：", state[4])
        print("可用水资源：", state[5])
        print("可用医疗资源：", state[6])
        print("可用救援装备：", state[7])
        print("可用救援人员数量：", state[8])
        print("可用避难所数量：", state[9])
        print("食物需求：", state[10])
        print("水需求：", state[11])
        print("医疗需求：", state[12])

        print("天气条件：", state[13])
        print("污染状况：", state[14])

        print("死亡人数：", state[15])
        return state

    def step(self, action_rebuild, action_rescue, action_resource, action_schedule):
        """
        执行一步操作，更新环境状态。
        :param action_rebuild: 灾后重建智能体的行动。
        :param action_rescue: 救援智能体的行动。
        :param action_resource: 资源分配智能体的行动。
        :param action_schedule: 调度智能体的行动。
        :return: 更新后的状态、救援智能体的奖励、资源分配智能体的奖励、调度智能体的奖励是否结束、附加信息。
        """
        reward_rescue = 0
        reward_resource = 0
        reward_rebuild = 0
        reward_schedule = 0
        done = False

        # 更新灾害强度
        self.state[0] = self._calculate_disaster_intensity(self.loop)
        # 执行灾后重建智能体的行动
        reward_rebuild += self._execute_rebuild_action(action_rebuild)
        # 执行救援智能体的行动
        reward_rescue += self._execute_rescue_action(action_rescue)
        # 执行资源分配智能体的行动
        reward_resource += self._execute_resource_action(action_resource)
        reward_schedule += self._execute_schedule_action(action_schedule)
        # 动态更新状态
        self._update_state_dynamics()

        self._update_pollution()
        self._apply_pollution_effect()
        # 检查是否结束
        done = self.loop >= self.duration

        return self.state, reward_rescue, reward_resource, reward_rebuild, reward_schedule, done, {}

    def _execute_rebuild_action(self, de_action):
        """
        执行重建智能体的行动。
        :param de_action: 重建智能体的行动。
        :return: 重建智能体的奖励。
        """
        reward = 0
        action = de_action['action']
        resource = de_action['quantity']

        if action == 0:  # 扩建避难所容量
            builders = resource['members']
            equipment = resource['equipment']
            if self.state[8] >= builders and self.state[7] >= equipment:
                built = sum(1 for _ in range(builders) if np.random.rand() < self.state[13] / 100)
                self.state[9] += built * 5  # 每单位工人扩建5人容量
                self.state[7] -= equipment
                reward += built * 2  # 每成功一次给2分
            else:
                reward -= 10

        elif action == 1:  # 修复基础设施（用救援人员 + 装备）
            members = resource['members']
            equipment = resource['equipment']
            if self.state[8]+self.state[2] >= members and self.state[7] >= equipment:
                success = sum(1 for _ in range(members) if np.random.rand() > (1 - self.state[13]) / 100)
                self.state[3] = max(0, self.state[3] - success)  # 减少损坏的基础设施
                self.state[7] -= equipment
                reward += success * 3
            else:
                reward -= 10


        elif action == 2:  # 污染治理（减轻污染值）
            workers = resource['members']  # 所需人力（救援人员 + 已安置群众）
            equipment = resource['equipment']  # 所需救援装备
            total_manpower = self.state[8] + self.state[2]  # 可用救援人员 + 已安置群众
            if total_manpower >= workers and self.state[7] >= equipment:
                success_rate = (self.state[13] / 100) * 0.6 + (1 - self.state[0] / 1000) * 0.4
                cleaned = sum(1 for _ in range(workers) if np.random.rand() < success_rate)
                self.state[7] -= equipment  # 消耗装备
                if cleaned / workers < 0.2:  # 成功率低于20%算失败
                    pollution_risk = (1 - self.state[13] / 100) * 0.5 + (self.state[0] / 1000) * 0.5
                    pollution_increase = int(np.random.rand() * 5 * pollution_risk)
                    self.state[14] += min(100, self.state[14] + pollution_increase)
                    reward -= 5
                else:
                    self.state[14] = max(0, self.state[14] - cleaned * 2)  # 成功治理，污染下降
                    reward += cleaned * 2
            else:
                reward -= 10

        return reward

    def _execute_rescue_action(self, de_action):
        """
        执行救援智能体的行动。
        :param de_action: 救援智能体的行动。
        :return: 救援智能体的奖励。
        """
        reward = 0
        action = de_action['action']
        resource = de_action['quantity']

        if action == 0:  # 派出救援队搜救未被救出人员
            members = resource['members']
            equipment = resource['equipment']
            if self.state[8] >= members and self.state[7] >= equipment:
                success_rescue = 0
                death_found = 0
                self.state[7] -= equipment
                for _ in range(members):
                    base_prob = self.state[13] / 100  # 天气影响
                    difficulty_penalty = self.state[0] / 200  # 灾害强度影响
                    rescue_chance = max(0, base_prob - difficulty_penalty)
                    if np.random.rand() < rescue_chance:
                        if np.random.rand() < 0.85:
                            success_rescue += 1
                            reward += 5
                        else:
                            death_found += 1
                            reward -= 3
                    else:
                        reward -= 5
                self.state[1] += success_rescue
                self.state[15] += death_found
            else:
                reward -= 10  # 资源不足惩罚

        elif action == 1:  # 提供紧急医疗援助
            members = resource['members']
            medical = resource['medical']
            equipment = resource['equipment']
            if self.state[8] >= members and self.state[6] >= medical and self.state[7] >= equipment:
                treated = 0
                for _ in range(members):
                    # 成功率受天气、污染、灾难影响影响
                    base_chance = self.state[13] / 100
                    pollution_penalty = self.state[14] / 200
                    disaster_penalty = self.state[0] / 200
                    success_chance = max(0, base_chance - pollution_penalty - disaster_penalty)
                    if np.random.rand() < success_chance:
                        treated += 1
                actual_treated = min(treated, self.state[12])  # 不超过需求
                self.state[12] = max(0, self.state[12] - actual_treated)
                self.state[6] -= medical
                self.state[7] -= equipment
                self.state[14] = max(0, self.state[14] - actual_treated * 0.5)  # 每成功1人减少0.5污染
                reward += actual_treated * 3
            else:
                reward -= 10  # 资源不足

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
            if self.state[4] >= food and self.state[5] >= water:  # 有足够资源
                if self.state[10] > 0 or self.state[11] > 0:  # 有需求
                    self.state[4] -= food
                    self.state[5] -= water
                    self.state[10] = max(0, self.state[10] - food * 1.5)
                    self.state[11] = max(0, self.state[11] - water * 1.5)
                    reward += 10
                    print(f"食物需求：{self.state[10]}, 水需求：{self.state[11]}")
                else:
                    reward -= 10  # 没有需求仍发放
            else:
                reward -= 10  # 资源不足


        elif action == 1:  # 安置受灾群众
            settle_count = resource['settle']# 决定安置多少人
            available_people = self.state[1]  # 仍未安置的人数
            if self.state[9] >= settle_count and available_people > 0:
                self.state[2] += settle_count
                self.state[1] -= settle_count
                self.state[9] -= settle_count
                reward += settle_count * 0.5
            else:
                reward -= 5


        elif action == 2:  # 基础资源采集（低效率获取食物/水）
            workers = resource['workers']
            if self.state[2] >= workers:  # 由已被安置群众进行采集
                gathered = 0
                for _ in range(workers):
                    # 成功概率受天气、污染、灾难强度共同影响
                    base_chance = 0.6
                    weather_bonus = self.state[13] / 200  # 天气越好，概率越高
                    pollution_penalty = self.state[14] / 300  # 污染越高，概率越低
                    disaster_penalty = self.state[0] / 300  # 灾难越强，越难采集
                    success_chance = min(1.0,
                                         max(0.1, base_chance + weather_bonus - pollution_penalty - disaster_penalty))
                    if np.random.rand() < success_chance:
                        gathered += 1
                self.state[4] += gathered * 2  # 每单位食物
                self.state[5] += gathered * 2  # 每单位水
                reward += gathered
            else:
                reward -= 10  # 人力不足

        return reward

    def _execute_schedule_action(self, de_action):
        """
        执行调度智能体的行动。
        :param de_action: 调度智能体的行动（包含action和quantity字典）。
        :return: 奖励值。
        """
        reward = 0
        action = de_action['action']
        quantity = de_action['quantity']

        if action == 0:
            # 调度食物和水
            food = quantity.get('food', 0)
            water = quantity.get('water', 0)

            # 超量调度惩罚（只要某一项本次调度 > 200 就惩罚）
            if food > 200:
                reward -= 5
            if water > 200:
                reward -= 5

            self.state[4] += food  # 食物
            self.state[5] += water  # 水

            reward += 5

        elif action == 1:
            # 调度救援成员、医疗资源、装备
            member = quantity.get('member', 0)
            medicine = quantity.get('medicine', 0)
            equipment = quantity.get('equipment', 0)

            if member > 200:
                reward -= 5
            if medicine > 200:
                reward -= 5
            if equipment > 200:
                reward -= 5

            self.state[8] += member  # 救援人员
            self.state[6] += medicine  # 医疗资源
            self.state[7] += equipment  # 救援装备

            reward += 10

        else:
            reward -= 10 # 非法行为惩罚

        return reward

    def _calculate_disaster_intensity(self, t):
        """
        计算灾害强度的动态变化。
        :param t: 当前时间步。
        :return: 计算后的灾害强度。
        """
        return self.amplitude * np.sin(self.frequency * t) + self.offset

    def _apply_pollution_effect(self):
        """
        污染对食物和水资源造成损耗。
        """
        pollution = self.state[14]  # 0-100，越高越污染
        decay_rate = pollution / 100 # 最大每轮损失10%

        food_loss = int(self.state[4] * decay_rate)
        water_loss = int(self.state[5] * decay_rate)
        if (food_loss > 0):
            self.state[4] = max(0, self.state[4] - food_loss)
        else:
            food_loss = 0
        if (water_loss > 0):
            self.state[5] = max(0, self.state[5] - water_loss)
        else:
            water_loss = 0



        print(f"污染影响资源：食物减少 {food_loss}，水减少 {water_loss}")

    def _update_pollution(self):
        """
        根据死亡人数和天气条件计算污染值。
        """
        deaths = self.state[15]
        weather = self.state[13]  # 0-100，越高越好（清凉）

        # 死亡比例归一化（设最大1000人为极限）
        death_factor = min(deaths / 1000, 1.0)

        # 天气因子，天气越差（越热），污染因子越高
        temperature_factor = 1 - (weather / 100)

        # 计算污染值，最大污染为100
        pollution = death_factor * temperature_factor * 100

        self.state[14] = min(100, int(pollution))  # 限制在 0~100
        self.state[14] = max(0,self.state[14])
        print(f"更新污染值为：{self.state[14]}")

    def _update_state_dynamics(self):
        """
        动态更新状态，模拟环境的自然变化。
        """
        #基础设施提供的
        # self.state[4] += max(0,500 - self.state[3])
        # self.state[5] += max(0,500 - self.state[3])
        # #外来补给
        # self.state[6] += 100
        # self.state[7] += 100
        # self.state[8] += 100

        # 减员规则：
        population_affected = self.state[1] + self.state[2]  # 受灾群众总数量
        medical_needs = self.state[12]  # 医疗需求
        food_needs = self.state[10]  # 食物需求
        water_needs = self.state[11]  # 水需求

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

        self.state[15] += casualties
        self.state[1] -= casualties  # 更新受灾群众总数量
        if self.state[1] < 0: self.state[1] = 0
        # 更新灾害强度
        self.loop += 1
        self.state[0] = self._calculate_disaster_intensity(self.loop)
        # 灾害强度影响基础设施损坏
        self.state[3] += int(self.state[0] * 0.1) * np.random.rand()  # 灾害强度增加基础设施损坏数量

        # 更新需求
        self.state[12] += self.state[2] * 0.1
        self.state[10] += self.state[2] * 0.1
        self.state[11] += self.state[2] * 0.1

        # 更新天气
        self.state[13] += int(np.random.rand() * 10)

        # 打印输出本轮结果
        print("第", self.loop, "轮结果：")
        print("灾害强度：", self.state[0])
        print("受灾区域待救援人数：", self.state[1])
        print("已被安置群众数量（累计）：", self.state[2])
        print("基础设施损坏数量：", self.state[3])
        print("可用食物资源：", self.state[4])
        print("可用水资源：", self.state[5])
        print("可用医疗资源：", self.state[6])
        print("可用救援装备：", self.state[7])
        print("可用救援人员数量：", self.state[8])
        print("可用避难所数量：", self.state[9])
        print("食物需求：", self.state[10])
        print("水需求：", self.state[11])
        print("医疗需求：", self.state[12])
        print("天气条件：", self.state[13])
        print("污染状况：", self.state[14])
        print("死亡人数：", self.state[15])

    def reset(self):
        """
        重置环境，返回初始状态。
        """
        self.state = self._initialize_state()
        self.loop = 0
        return self.state

    def return_state(self):
        return self.state