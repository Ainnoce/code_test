import numpy as np
import matplotlib.pyplot as plt

# 1. 设定真实机器人的隐藏参数 (Sim-to-Real 差异的来源)
J_real = 1.5  # 真实转动惯量 (Sim里以为是 1.0)
B_real = 0.8  # 真实粘性摩擦系数
dt = 0.01  # 控制周期 10ms

# 2. 生成傅里叶激励轨迹 (让机器人按这个位置轨迹运动)
t = np.arange(0, 4, dt)  # 4秒辨识时间
q_des = 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.cos(2 * np.pi * 1.0 * t)
# 通过差分计算速度和加速度 (工程上通常用观测器，这里为演示直接用差分)
q_dot_des = np.gradient(q_des, dt)
q_ddot_des = np.gradient(q_dot_des, dt)

# 3. 模拟真实机器人的响应并加入传感器噪声
tau_cmd = np.zeros_like(t)  # 记录发给机器人的扭矩
q_real = np.zeros_like(t)
q_real[0] = q_des[0]
# 简单的欧拉积分模拟真实物理: tau = J*q_ddot + B*q_dot
for i in range(1, len(t)):
    # 这里的 q_ddot_des[i] 是期望加速度，作为系统的输入激励
    tau = J_real * q_ddot_des[i] + B_real * q_dot_des[i]
    # 加入过程噪声模拟真实电机电流环的误差
    tau_cmd[i] = tau + np.random.normal(0, 0.05)

# 4. 递归最小二乘法 (RLS) 在线辨识
n_params = 2  # [J, B]
theta_hat = np.zeros(n_params)  # 初始猜测 [0, 0]
P = np.eye(n_params) * 1000  # 初始极大不确定度
lambda_factor = 0.995  # 轻微遗忘因子，应对非线性未建模动态

# 存储历史估计值用于画图
history_J = []
history_B = []

for i in range(len(t)):
    if i < 2:
        continue  # 跳过前两点，因为差分需要历史数据

    # 构造回归矩阵 Y = [加速度, 速度]
    Y = np.array([q_ddot_des[i], q_dot_des[i]])

    # 获取当前实际扭矩 (这里用模拟的真实扭矩 tau_cmd)
    tau_measured = tau_cmd[i]

    # --- RLS 核心递推公式 ---
    epsilon = tau_measured - np.dot(Y, theta_hat)  # 计算残差
    K = P @ Y / (lambda_factor + Y.T @ P @ Y)  # 计算卡尔曼增益
    theta_hat = theta_hat + K * epsilon  # 更新参数估计
    P = (np.eye(n_params) - np.outer(K, Y)) @ P / lambda_factor  # 更新协方差

    history_J.append(theta_hat[0])
    history_B.append(theta_hat[1])

# 5. 可视化结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t[2:], history_J, label="Estimated J")
plt.axhline(y=J_real, color="r", linestyle="--", label="True J (1.5)")
plt.title("Inertia Estimation Convergence")
plt.xlabel("Time (s)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t[2:], history_B, label="Estimated B")
plt.axhline(y=B_real, color="r", linestyle="--", label="True B (0.8)")
plt.title("Damping Estimation Convergence")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.show()

print(f"辨识完成的参数 -> 惯量 J: {history_J[-1]:.4f}, 阻尼 B: {history_B[-1]:.4f}")
