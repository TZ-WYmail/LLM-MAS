from flask import Flask, request, jsonify
import threading
import time
import threading

import requests

from Agent.GovernmentAgent import GovernmentAgent
from Agent.RebuildingAgent import RebuildingAgent
from Agent.RescueAgent import RescueAgent
from Agent.ResourceManagementAgent import ResourceManagementAgent
from Agent.ResourceSchedulingAgent import ResourceSchedulingAgent
from newEnv import DisasterResponseEnv
from LLM import LLMClient
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
app = Flask(__name__)


def main():
    # 初始化环境
    global simulation_states, simulation_running,state_dict
    simulation_running = True
    simulation_states = np.zeros(16, dtype=float)
    state_dict = build_state_dict(simulation_states)
    duration = 20  # 模拟总时间步长
    env = DisasterResponseEnv(duration)

    llm = LLMClient()

    # 初始化智能体
    government_agent = GovernmentAgent(env, llm)
    rebuilding_agent = RebuildingAgent(env, llm)
    rescue_agent = RescueAgent(env, llm)
    resource_management_agent = ResourceManagementAgent(env, llm)
    resource_scheduling_agent = ResourceSchedulingAgent(env, llm)
    # 将其他智能体实例绑定到政府智能体
    government_agent.RebuildingAgent = rebuilding_agent
    government_agent.RescueAgent = rescue_agent
    government_agent.ResourceManagementAgent = resource_management_agent
    government_agent.ResourceSchedulingAgent = resource_scheduling_agent

    RebuildingAgent.GovernmentAgent = government_agent
    RescueAgent.GovernmentAgent = government_agent
    ResourceManagementAgent.GovernmentAgent = government_agent
    ResourceSchedulingAgent.GovernmentAgent = government_agent

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
            elif agent_name == "ResourceSchedulingAgent":
                print("调度智能体正在制定调度计划...")
                actions["schedule_action"] = resource_scheduling_agent.next_action()
                print('..............')
            else:
                print(f"未知智能体名称：{agent_name}")

        # 执行智能体的行动并更新环境状态
        print("执行智能体的行动并更新环境状态...")
        # 提取动作字典中的具体行动，确保每个行动都有默认值
        rebuild_action = actions.get("rebuild_action", {"action": 0, "quantity": [0, 0]})
        rescue_action = actions.get("rescue_action", {"action": 0, "quantity": [0, 0, 0]})
        resource_action = actions.get("resource_action", {"action": 0, "quantity": [0, 0, 0, 0, 0]})
        schedule_action = actions.get("schedule_action", {"action": 0, "quantity": [0, 0, 0, 0, 0]})

        RebuildingAgent.GovernmentAgent.action_get("RebuildingAgent", rebuild_action)
        RescueAgent.GovernmentAgent.action_get("RescueAgent", rescue_action)
        ResourceManagementAgent.GovernmentAgent.action_get("ResourceManagementAgent", resource_action)
        ResourceSchedulingAgent.GovernmentAgent.action_get("ResourceSchedulingAgent",schedule_action)
        rebuild_action, rescue_action, resource_action, schedule_action = government_agent.re_decision()

        print("rebuild_action:" + str(rebuild_action))
        print("rescue_action:" + str(rescue_action))
        print("resource_action:" + str(resource_action))
        print("schedule_action:" + str(schedule_action))
        # 调用环境的 step 方法，执行一步操作
        state, reward_rescue, reward_resource, reward_rebuild, reward_schedule, done, info = env.step(rebuild_action, rescue_action,resource_action, schedule_action)
        # 打印奖励和环境状态
        print(f"救援智能体奖励: {reward_rescue}")
        print(f"资源管理智能体奖励: {reward_resource}")
        print(f"重建智能体奖励: {reward_resource}")
        print(f"调度智能体奖励: {reward_schedule}")
        print(f"环境状态: {state}")

        rebuilding_agent.gain = reward_rebuild
        rescue_agent.gain = reward_rescue
        resource_management_agent.gain = reward_resource
        resource_scheduling_agent.gain = reward_schedule
        GovernmentAgent.gain = reward_rebuild + reward_rescue + reward_resource + reward_schedule
        # 更新智能体的记忆
        print("更新智能体的记忆...")
        government_agent.update_memory()
        rebuilding_agent.update_memory()
        rescue_agent.update_memory()
        resource_management_agent.update_memory()
        resource_scheduling_agent.update_memory()

        # 存储记忆
        government_agent.memory.save_to_csv()
        rebuilding_agent.memory.save_to_csv()
        rescue_agent.memory.save_to_csv()
        resource_management_agent.memory.save_to_csv()
        resource_scheduling_agent.memory.save_to_csv()
        print(rescue_agent.memory.short_memory.__len__())
        print("记忆存储成功！")
        simulation_states = env.return_state()
        print(simulation_states)
        state_dict = build_state_dict(simulation_states)
        print(state_dict)
        if done:
            print("模拟结束！")
            simulation_running = False
            break

# 模拟函数（你已有的环境模拟函数应写在这里）
def run_simulation():
    global simulation_states, simulation_running
    simulation_running = True
    simulation_states = []

    for _ in range(5):  # 假设模拟 5 轮（你可以自定义）
        time.sleep(1)  # 模拟每轮耗时（你已有逻辑代替）
        fake_state = [i for i in range(16)]  # 示例，替换为你的 state
        simulation_states.append(fake_state)

    simulation_running = False


@app.route('/start_disaster', methods=['POST'])
def start_disaster():
    print("收到 Unity 启动请求")
    t = threading.Thread(target=main)
    t.start()
    return jsonify({"status": "started"})


@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({
        "states": state_dict,
        "running": simulation_running
    })

def send_state_to_unity(state_dict):
    url = "http://localhost:8888/state"
    try:
        response = requests.post(url, json=state_dict)
        print("发送成功:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print("请求失败:", e)

def build_state_dict(state):
    return {
        "disaster_intensity": int(state[0]),
        "unrescued_people": int(state[1]),
        "resettled_residents_number": int(state[2]),
        "infrastructure_damage": int(state[3]),
        "available_food": int(state[4]),
        "available_water": int(state[5]),
        "available_medical": int(state[6]),
        "available_rescue_resource": int(state[7]),
        "available_rescue_member": int(state[8]),
        "available_center": int(state[9]),
        "food_needs": int(state[10]),
        "water_needs": int(state[11]),
        "medical_needs": int(state[12]),
        "weather_conditions": int(state[13]),
        "pollution": int(state[14]),
        "number_of_deaths": int(state[15])
    }

if __name__ == '__main__':
    app.run(host="localhost", port=8080)
