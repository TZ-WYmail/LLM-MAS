from Agent.GovernmentAgent import GovernmentAgent
from Agent.RebuildingAgent import RebuildingAgent
from Agent.RescueAgent import RescueAgent
from Agent.ResourceManagementAgent import ResourceManagementAgent
from newEnv import DisasterResponseEnv
from LLM import LLMClient


def main():
    # 初始化环境
    duration = 20  # 模拟总时间步长
    env = DisasterResponseEnv(duration)

    llm = LLMClient()

    # 初始化智能体
    government_agent = GovernmentAgent(env, llm)
    rebuilding_agent = RebuildingAgent(env, llm)
    rescue_agent = RescueAgent(env, llm)
    resource_management_agent = ResourceManagementAgent(env, llm)

    # 将其他智能体实例绑定到政府智能体
    government_agent.RebuildingAgent = rebuilding_agent
    government_agent.RescueAgent = rescue_agent
    government_agent.ResourceManagementAgent = resource_management_agent
    RebuildingAgent.GovernmentAgent = government_agent
    RescueAgent.GovernmentAgent = government_agent
    ResourceManagementAgent.GovernmentAgent = government_agent

    # 模拟循环
    for t in range(duration):
        print(f"\n--- 第 {t + 1} 轮模拟 ---")

        # 政府智能体决策
        print("政府智能体正在制定政策...")
        agent_names = government_agent.next_action()
        print('..............')

        # 根据政府智能体的决策调用其他智能体
        actions = {}
        for agent_name in agent_names:
            if agent_name == "RescueAgent":
                print("救援智能体正在制定救援计划...")
                actions["rescue_action"] = rescue_agent.next_action()
                print('..............')
            elif agent_name == "ResourceManagementAgent":
                print("资源管理智能体正在制定资源分配计划...")
                actions["resource_action"] = resource_management_agent.next_action()
                print('..............')
            elif agent_name == "RebuildingAgent":
                print("重建智能体正在制定重建计划...")
                actions["rebuild_action"] = rebuilding_agent.next_action()
                print('..............')
            else:
                print(f"未知智能体名称：{agent_name}")

        # 执行智能体的行动并更新环境状态
        print("执行智能体的行动并更新环境状态...")
        # 提取动作字典中的具体行动，确保每个行动都有默认值
        rebuild_action = actions.get("rebuild_action", {"action": 0, "quantity": 0})
        rescue_action = actions.get("rescue_action", {"action": 0, "quantity": 0})
        resource_action = actions.get("resource_action", {"action": 0, "resource": [0, 0]})

        RebuildingAgent.GovernmentAgent.action_get("RebuildingAgent", rebuild_action)
        RescueAgent.GovernmentAgent.action_get("RescueAgent", rescue_action)
        ResourceManagementAgent.GovernmentAgent.action_get("ResourceManagementAgent", resource_action)
        rebuild_action, rescue_action, resource_action = government_agent.re_decision()

        print("rebuild_action:" + str(rebuild_action))
        print("rescue_action:" + str(rescue_action))
        print("resource_action:" + str(resource_action))

        # 调用环境的 step 方法，执行一步操作
        state, reward_rescue, reward_resource, reward_rebuild, done, info = env.step(rebuild_action, rescue_action,
                                                                                     resource_action)
        # 打印奖励和环境状态
        print(f"救援智能体奖励: {reward_rescue}")
        print(f"资源管理智能体奖励: {reward_resource}")
        print(f"重建智能体奖励: {reward_resource}")
        print(f"环境状态: {state}")

        rebuilding_agent.gain = reward_rebuild
        rescue_agent.gain = reward_rescue
        resource_management_agent.gain = reward_resource
        GovernmentAgent.gain = reward_rebuild + reward_rescue + reward_resource
        # 更新智能体的记忆
        print("更新智能体的记忆...")
        government_agent.update_memory()
        rebuilding_agent.update_memory()
        rescue_agent.update_memory()
        resource_management_agent.update_memory()

        # 存储记忆
        government_agent.memory.save_to_csv()
        rebuilding_agent.memory.save_to_csv()
        rescue_agent.memory.save_to_csv()
        resource_management_agent.memory.save_to_csv()
        print(rescue_agent.memory.short_memory.__len__())
        print("记忆存储成功！")

        if done:
            print("模拟结束！")

            break


if __name__ == "__main__":
    main()
