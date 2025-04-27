from Single_agent import SingleAgent
from DisasterEnv import DisasterResponseEnv
from LLM import LLMClient


def main():
    # 初始化环境
    duration = 20  # 模拟总时间步长
    env = DisasterResponseEnv(duration)

    llm = LLMClient()

    # 初始化单一智能体
    single_agent = SingleAgent(env, llm)

    # 模拟循环
    for t in range(duration):
        print(f"\n--- 第 {t + 1} 轮模拟 ---")

        # 单一智能体决策
        print("单一智能体正在制定政策...")
        rebuild_action, rescue_action, resource_action = single_agent.next_action()
        print(rebuild_action)
        print(rescue_action)
        print(resource_action)
        print('..............')

        # 执行智能体的行动并更新环境状态
        print("执行智能体的行动并更新环境状态...")
        state, reward_rescue, reward_resource, reward_rebuild, done, info = env.step(rebuild_action, rescue_action,
                                                                                     resource_action)
        # 打印奖励和环境状态
        print(f"救援行动奖励: {reward_rescue}")
        print(f"资源管理行动奖励: {reward_resource}")
        print(f"重建行动奖励: {reward_rebuild}")
        print(f"环境状态: {state}")

        # 更新单一智能体的收益
        single_agent.gain = reward_rebuild + reward_rescue + reward_resource

        # 更新单一智能体的记忆
        print("更新单一智能体的记忆...")
        single_agent.update_memory()

        # 存储记忆
        single_agent.memory.save_to_csv()
        print("记忆存储成功！")

        if done:
            print("模拟结束！")
            break


if __name__ == "__main__":
    main()