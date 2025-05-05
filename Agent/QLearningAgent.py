import numpy as np

from DisasterEnv import DisasterResponseEnv


class QLearningAgent:
    def __init__(self, state_space, action_space_rescue, action_space_resource, action_space_rebuild, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, num_bins=3):
        self.state_space = state_space
        self.action_space_rescue = action_space_rescue
        self.action_space_resource = action_space_resource
        self.action_space_rebuild = action_space_rebuild
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_bins = num_bins

        # 初始化状态离散化的边界
        self.state_bins = [np.linspace(state_space.low[i], state_space.high[i], num_bins) for i in range(state_space.shape[0])]

        # 计算离散化后的状态数量
        num_discrete_states = num_bins ** state_space.shape[0]

        self.q_table_rescue = np.zeros((num_discrete_states, self.action_space_rescue.n))
        self.q_table_resource = np.zeros((num_discrete_states, self.action_space_resource.n))
        self.q_table_rebuild = np.zeros((num_discrete_states, self.action_space_rebuild.n))

    def discretize_state(self, state):
        """
        将连续状态离散化
        """
        discrete_state = []
        for i in range(len(state)):
            discrete_state.append(np.digitize(state[i], self.state_bins[i]))
        # 将离散化后的状态转换为单个索引
        index = 0
        for i, val in enumerate(discrete_state):
            index += val * (self.num_bins ** i)
        return index

    def choose_action(self, state, agent_type):
        discrete_state = self.discretize_state(state)
        if np.random.uniform(0, 1) < self.epsilon:
            if agent_type == 'rescue':
                return self.action_space_rescue.sample()
            elif agent_type == 'resource':
                return self.action_space_resource.sample()
            elif agent_type == 'rebuild':
                return self.action_space_rebuild.sample()
        else:
            if agent_type == 'rescue':
                return np.argmax(self.q_table_rescue[discrete_state, :])
            elif agent_type == 'resource':
                return np.argmax(self.q_table_resource[discrete_state, :])
            elif agent_type == 'rebuild':
                return np.argmax(self.q_table_rebuild[discrete_state, :])

    def update_q_table(self, state, action, reward, next_state, agent_type):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        if agent_type == 'rescue':
            predict = self.q_table_rescue[discrete_state, action]
            target = reward + self.discount_factor * np.max(self.q_table_rescue[discrete_next_state, :])
            self.q_table_rescue[discrete_state, action] = (1 - self.learning_rate) * predict + self.learning_rate * target
        elif agent_type == 'resource':
            predict = self.q_table_resource[discrete_state, action]
            target = reward + self.discount_factor * np.max(self.q_table_resource[discrete_next_state, :])
            self.q_table_resource[discrete_state, action] = (1 - self.learning_rate) * predict + self.learning_rate * target
        elif agent_type == 'rebuild':
            predict = self.q_table_rebuild[discrete_state, action]
            target = reward + self.discount_factor * np.max(self.q_table_rebuild[discrete_next_state, :])
            self.q_table_rebuild[discrete_state, action] = (1 - self.learning_rate) * predict + self.learning_rate * target


# 主程序
if __name__ == "__main__":
    duration = 20
    env = DisasterResponseEnv(duration)
    agent = QLearningAgent(env.observation_space, env.action_space_rescue, env.action_space_resource, env.action_space_rebuild)

    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action_rescue = agent.choose_action(state, 'rescue')
            action_resource = agent.choose_action(state, 'resource')
            action_rebuild = agent.choose_action(state, 'rebuild')

            # 这里需要构造符合环境要求的动作格式
            de_action_rescue = {'action': action_rescue, 'quantity': 1}
            de_action_resource = {'action': action_resource, 'quantity': {'food': 1, 'water': 1, 'medical': 1, 'workers': 1}}
            de_action_rebuild = {'action': action_rebuild, 'quantity': 1}

            next_state, reward_rescue, reward_resource, reward_rebuild, done, _ = env.step(de_action_rebuild, de_action_rescue, de_action_resource)

            agent.update_q_table(state, action_rescue, reward_rescue, next_state, 'rescue')
            agent.update_q_table(state, action_resource, reward_resource, next_state, 'resource')
            agent.update_q_table(state, action_rebuild, reward_rebuild, next_state, 'rebuild')

            state = next_state

        print(f"Episode {episode + 1} finished")