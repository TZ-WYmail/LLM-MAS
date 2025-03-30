import csv
import os
import random
import time
import json

import numpy as np


class Memory:
    def __init__(self, decay_rate=0.1, long_memory_threshold=3):
        """
        初始化记忆类。
        :param decay_rate: 遗忘曲线的衰减率。
        :param long_memory_threshold: 转换为长期记忆的阈值（验证次数）。
        """
        self.short_memory = []  # 短期记忆
        self.long_memory = []  # 长期记忆
        self.decay_rate = decay_rate
        self.long_memory_threshold = long_memory_threshold
        self.file_name=None

    def add_to_short_memory(self, timestamp, action, evi, gain):
        """
        将结果（无论是好是坏）添加到短期记忆中。
        :param timestamp: 时间戳。
        :param action: 行动。
        :param evi: 当前环境。
        :param gain: 增益（正或负）。
        """
        self.short_memory.append({
            "timestamp": timestamp,
            "action": action,
            "evi": evi,
            "gain": gain,
            "count": 1  # 初始化验证次数
        })

    def apply_forget_curve(self):
        """
        应用遗忘曲线，对短期记忆中的内容进行动态管理。
        好的记忆衰减慢，坏的记忆衰减快。
        """
        current_time = time.time()
        self.short_memory = [
            mem for mem in self.short_memory
            if (current_time - mem["timestamp"]) < (1 / self.decay_rate)
        ]

    def promote_to_long_memory(self):
        """
        将多次验证的结果转移到长期记忆。
        好的记忆和坏的记忆都可以进入长期记忆，但权重不同。
        """
        for mem in self.short_memory:
            if mem["count"] >= self.long_memory_threshold:
                self.long_memory.append(mem)
                self.short_memory.remove(mem)


    def update_memory(self, action, evi, gain):
        """
        更新记忆内容。如果记忆不存在，则自动添加到短期记忆中。
        自动处理短期记忆的更新、遗忘曲线应用和长期记忆的提升。
        :param action: 行动。
        :param evi: 当前环境（15维数组）。
        :param gain: 增益（正或负）。
        """
        # 定义相似性阈值
        similarity_threshold = 400.0  # 可根据实际情况调整

        # 检查短期记忆中是否存在相似的记忆
        found = False
        for mem in self.short_memory:
            if mem["action"] == action:
                # 计算 evi 和 mem["evi"] 之间的欧几里得距离
                distance = np.linalg.norm(np.array(evi) - np.array(mem["evi"]))
                if distance <= similarity_threshold:
                    # 更新验证次数和增益值
                    mem["count"] += 1
                    mem["gain"] = (mem["gain"] * (mem["count"] - 1) + gain) / mem["count"]
                    found = True
                    break

        # 如果未找到相似的记忆，则添加到短期记忆
        if not found:
            self.add_to_short_memory(time.time(), action, evi, gain)
            print("add to short memory")

        # 应用遗忘曲线，清理过期的短期记忆
        self.apply_forget_curve()

        # 检查是否需要将短期记忆提升到长期记忆
        self.promote_to_long_memory()

    def get_memory_prompt(self, current_evi):
        """
        根据短期记忆和长期记忆的内容分配权重，提供相似情况的行动。
        好的记忆权重高，坏的记忆权重低。
        :param current_evi: 当前环境。
        :return: 包含推荐行动编号、原因、历史收益等信息的字典。
        """
        reference_actions = []
        similarity_threshold = 400.0  # 相似性阈值

        # 从短期记忆中获取相似行动
        for mem in self.short_memory:
            distance = np.linalg.norm(np.array(current_evi) - np.array(mem["evi"]))
            if distance <= similarity_threshold:
                reference_actions.append(mem)

        # 从长期记忆中获取相似行动
        for mem in self.long_memory:
            distance = np.linalg.norm(np.array(current_evi) - np.array(mem["evi"]))
            if distance <= similarity_threshold:
                reference_actions.append(mem)

        # 根据权重（增益）选择参考行动
        if reference_actions:
            # 权重选择：增益高的行动被选中的概率更高
            total_gain = sum(abs(mem["gain"]) for mem in reference_actions)
            if total_gain == 0:
                return None
            weights = [abs(mem["gain"]) / total_gain for mem in reference_actions]
            selected_action = random.choices(reference_actions, weights=weights, k=1)[0]

            # 构造返回的提示字典
            memory_prompt = {
                "action": selected_action["action"],
                "evi": selected_action["evi"],
                "reason": f"相似环境下的历史收益为 {selected_action['gain']:.2f}，验证次数为 {selected_action['count']}",
                "gain": selected_action["gain"],
                "count": selected_action["count"]
            }
            return memory_prompt
        else:
            return None

    def save_to_csv(self):
        """
        将数据保存到 JSON 文件中。如果文件或目录不存在，则创建文件和目录。
        :param file_name: 文件路径。
        """
        # 确保文件所在的目录存在
        """
           将短期记忆和长期记忆保存到 CSV 文件中。
           :param file_name: 文件路径。
           """
        file_name = self.file_name
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        try:
            with open(file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 写入标题行
                writer.writerow(["memory_type", "action", "evi", "gain", "count"])
                # 写入短期记忆
                for mem in self.short_memory:
                    writer.writerow(["short", mem["action"], mem["evi"], mem["gain"], mem["count"]])
                # 写入长期记忆
                for mem in self.long_memory:
                    writer.writerow(["long", mem["action"], mem["evi"], mem["gain"], mem["count"]])
            print(f"数据已成功保存到 {file_name}")
        except Exception as e:
            print(f"保存文件时发生错误：{e}")

    def load_from_json(self):
        """
        从 JSON 文件中加载短期记忆和长期记忆。
        :param filename: JSON 文件的路径。
        """
        with open(self.file_name, "r") as f:
            data = json.load(f)
        self.short_memory = data.get("short_memory", [])
        self.long_memory = data.get("long_memory", [])
        self.decay_rate = data.get("decay_rate", 0.1)
        self.long_memory_threshold = data.get("long_memory_threshold", 3)

    def __str__(self):
        return f"短期记忆：{self.short_memory}\n长期记忆：{self.long_memory}"

