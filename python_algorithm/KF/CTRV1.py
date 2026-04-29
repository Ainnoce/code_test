import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from scipy.linalg import block_diag
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class CTRVKalmanSmoother:
    def __init__(self, dt, process_noise_pos=0.1, process_noise_vel=0.1, process_noise_yaw=0.01,
                 measurement_noise_pos=1.0, measurement_noise_vel=0.5):
        """
        使用CTRV模型的卡尔曼滤波器和平滑器

        参数:
        dt: 时间步长(秒)
        process_noise_*: 过程噪声参数
        measurement_noise_*: 测量噪声参数
        """
        self.dt = dt

        # 状态向量维度: [x, y, v, psi, psi_dot]
        self.state_dim = 5
        # 测量向量维度: [x, y, vx, vy]
        self.measure_dim = 4

        # 初始化状态转移矩阵 (非线性函数)
        self.F = self._compute_transition

        # 测量矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0],  # x测量
            [0, 1, 0, 0, 0],  # y测量
            [0, 0, 0, 0, 0],  # 占位符 - 将替换为计算值
            [0, 0, 0, 0, 0]  # 占位符 - 将替换为计算值
        ])

        # 过程噪声协方差
        self.Q = np.diag([
            process_noise_pos * dt ** 4,  # x位置噪声
            process_noise_pos * dt ** 4,  # y位置噪声
            process_noise_vel * dt ** 2,  # 速度噪声
            process_noise_yaw * dt ** 2,  # 航向噪声
            process_noise_yaw * dt ** 2  # 转向率噪声
        ])

        # 测量噪声协方差
        self.R = np.diag([
            measurement_noise_pos,  # x位置测量噪声
            measurement_noise_pos,  # y位置测量噪声
            measurement_noise_vel,  # x速度测量噪声
            measurement_noise_vel  # y速度测量噪声
        ])

        # 存储历史数据
        self.filtered_states = []
        self.filtered_covs = []
        self.predicted_states = []
        self.predicted_covs = []
        self.transition_jacobians = []

    def _compute_transition(self, state, dt=None):
        """
        CTRV 非线性状态转移函数
        :param state: [x, y, v, psi, psi_dot]
        :param dt: 时间步长(可选，默认使用类实例的dt)
        :return: 下一状态
        """
        if dt is None:
            dt = self.dt

        x, y, v, psi, psi_dot = state

        # 避免除以0
        if abs(psi_dot) < 1e-5:
            # 直线运动
            delta_x = v * np.cos(psi) * dt
            delta_y = v * np.sin(psi) * dt
            new_psi = psi
        else:
            # 曲线运动
            delta_theta = psi_dot * dt
            delta_x = (v / psi_dot) * (np.sin(psi + delta_theta) - np.sin(psi))
            delta_y = (v / psi_dot) * (np.cos(psi) - np.cos(psi + delta_theta))
            new_psi = psi + delta_theta

        return np.array([
            x + delta_x,
            y + delta_y,
            v,  # 速度保持不变（恒速模型）
            new_psi,
            psi_dot  # 转向率保持不变
        ])

    def _compute_jacobian(self, state, dt=None):
        """
        计算状态转移函数的雅可比矩阵
        :param state: [x, y, v, psi, psi_dot]
        :return: 雅可比矩阵 (5x5)
        """
        if dt is None:
            dt = self.dt

        _, _, v, psi, psi_dot = state

        # 处理直线运动（近似）
        if abs(psi_dot) < 1e-5:
            jac = np.array([
                [1, 0, np.cos(psi) * dt, -v * np.sin(psi) * dt, 0],
                [0, 1, np.sin(psi) * dt, v * np.cos(psi) * dt, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, dt],
                [0, 0, 0, 0, 1]
            ])
            return jac

        # 曲线运动情况
        delta_psi = psi_dot * dt
        v_div_psi = v / psi_dot

        # 偏导数计算
        j13 = (np.sin(psi + delta_psi) - np.sin(psi)) / psi_dot
        j14 = v_div_psi * (np.cos(psi + delta_psi) - np.cos(psi))
        j15 = v_div_psi * (dt * np.cos(psi + delta_psi) -
                           (np.sin(psi + delta_psi) - np.sin(psi)) / psi_dot)

        j23 = (np.cos(psi) - np.cos(psi + delta_psi)) / psi_dot
        j24 = v_div_psi * (np.sin(psi + delta_psi) - np.sin(psi))
        j25 = v_div_psi * (dt * np.sin(psi + delta_psi) -
                           (np.cos(psi + delta_psi) - np.cos(psi)) / psi_dot)

        # 构造雅可比矩阵
        jac = np.array([
            [1, 0, j13, j14, j15],
            [0, 1, j23, j24, j25],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, dt],
            [0, 0, 0, 0, 1]
        ])

        return jac

    def _compute_measurement_jacobian(self, state):
        """
        计算测量函数的雅可比矩阵
        :param state: [x, y, v, psi, psi_dot]
        :return: 测量雅可比矩阵 (4x5)
        """
        _, _, v, psi, _ = state
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, np.cos(psi), -v * np.sin(psi), 0],
            [0, 0, np.sin(psi), v * np.cos(psi), 0]
        ])
        return H

    def _init_from_measurement(self, measurement):
        """
        根据测量值初始化状态
        :param measurement: [x, y, vx, vy]
        """
        x, y, vx, vy = measurement

        # 计算速度和航向角
        v = np.sqrt(vx ** 2 + vy ** 2)
        psi = np.arctan2(vy, vx)

        # 初始转向率为0 (后续会更新)
        return np.array([x, y, v, psi, 0.0])

    def predict(self, state, cov):
        """
        预测步骤
        :param state: 当前状态估计
        :param cov: 当前协方差
        :return: 预测状态和协方差
        """
        # 非线性状态预测
        predicted_state = self.F(state)

        # 计算雅可比矩阵
        F_j = self._compute_jacobian(state)
        self.transition_jacobians.append(F_j)

        # 预测协方差
        predicted_cov = F_j @ cov @ F_j.T + self.Q

        return predicted_state, predicted_cov

    def update(self, measurement, predicted_state, predicted_cov):
        """
        更新步骤
        :param measurement: 新测量值 [x, y, vx, vy]
        :param predicted_state: 预测状态
        :param predicted_cov: 预测协方差
        :return: 更新后的状态和协方差
        """
        # 计算测量雅可比矩阵
        H_j = self._compute_measurement_jacobian(predicted_state)

        # 计算预测测量值
        _, _, v, psi, _ = predicted_state
        predicted_measurement = np.array([
            predicted_state[0],
            predicted_state[1],
            v * np.cos(psi),
            v * np.sin(psi)
        ])

        # 测量残差
        residual = measurement - predicted_measurement

        # 确保航向角残差在(-π, π]范围内
        # (位置和速度更新中不含角度，因此不需要特殊处理)

        # 卡尔曼增益
        S = H_j @ predicted_cov @ H_j.T + self.R
        K = predicted_cov @ H_j.T @ np.linalg.inv(S)

        # 状态更新
        updated_state = predicted_state + K @ residual

        # 归一化航向角到(-π, π]
        updated_state[3] = np.arctan2(np.sin(updated_state[3]), np.cos(updated_state[3]))

        # 协方差更新
        I = np.eye(self.state_dim)
        updated_cov = (I - K @ H_j) @ predicted_cov

        return updated_state, updated_cov

    def filter(self, measurements):
        """
        执行卡尔曼滤波
        :param measurements: 测量值序列 [[x, y, vx, vy], ...]
        :return: 滤波后的状态序列
        """
        # 初始化历史记录
        self.filtered_states = []
        self.filtered_covs = []
        self.predicted_states = []
        self.predicted_covs = []
        self.transition_jacobians = []

        # 初始化状态 (从第一个测量值)
        init_state = self._init_from_measurement(measurements[0])

        # 初始化协方差 (较大的初始不确定性)
        init_cov = np.diag([
            10.0,  # x位置
            10.0,  # y位置
            5.0,  # 速度
            np.pi,  # 航向角 (180°不确定性)
            1.0  # 转向率 (1rad/s不确定性)
        ])

        current_state = init_state
        current_cov = init_cov

        # 存储初始状态
        self.filtered_states.append(current_state)
        self.filtered_covs.append(current_cov)

        for i, meas in enumerate(measurements[1:], 1):
            # 预测步骤
            predicted_state, predicted_cov = self.predict(current_state, current_cov)
            self.predicted_states.append(predicted_state)
            self.predicted_covs.append(predicted_cov)

            # 更新步骤
            updated_state, updated_cov = self.update(meas, predicted_state, predicted_cov)

            current_state = updated_state
            current_cov = updated_cov

            self.filtered_states.append(current_state)
            self.filtered_covs.append(current_cov)

        return np.array(self.filtered_states)

    def rts_smooth(self):
        """
        RTS平滑算法 (后向处理)
        :return: 平滑后的状态序列
        """
        n = len(self.filtered_states)

        # 初始化平滑结果
        smoothed_states = [self.filtered_states[-1]]
        smoothed_covs = [self.filtered_covs[-1]]

        # 反向遍历 (从倒数第二个到第一个)
        for k in range(n - 2, -1, -1):
            # 获取前向估计
            filtered_state = self.filtered_states[k]
            filtered_cov = self.filtered_covs[k]

            # 获取预测结果
            predicted_state_next = self.predicted_states[k]
            predicted_cov_next = self.predicted_covs[k]

            # 获取状态转移雅可比矩阵
            F_j = self.transition_jacobians[k]

            # 平滑增益
            C = filtered_cov @ F_j.T @ np.linalg.inv(predicted_cov_next)

            # 平滑状态更新
            smooth_state = filtered_state + C @ (smoothed_states[0] - predicted_state_next)

            # 归一化航向角
            smooth_state[3] = np.arctan2(np.sin(smooth_state[3]), np.cos(smooth_state[3]))

            # 平滑协方差更新
            smooth_cov = filtered_cov + C @ (smoothed_covs[0] - predicted_cov_next) @ C.T

            # 添加到结果开头
            smoothed_states.insert(0, smooth_state)
            smoothed_covs.insert(0, smooth_cov)

        return np.array(smoothed_states)

    def process(self, measurements):
        """
        完整处理流程：滤波 + 平滑
        :param measurements: 测量值序列
        :return: 滤波和平滑后的状态
        """
        filtered = self.filter(measurements)
        smoothed = self.rts_smooth()
        return filtered, smoothed


def generate_test_data(dt=0.1, total_time=20):
    """生成测试数据：车辆进行转弯并恢复直线运动"""
    t = np.arange(0, total_time, dt)
    n = len(t)

    # 位置
    x = np.zeros(n)
    y = np.zeros(n)

    # 速度
    v = 5.0  # 恒定速度 5 m/s
    vx = np.zeros(n)
    vy = np.zeros(n)

    # 航向角
    psi = np.zeros(n)
    psi_dot = np.zeros(n)

    # 初始航向 (东方向)
    psi[0] = 0

    # 在第5-15秒进行转弯 (增加航向角)
    for i in range(0, n):
        if 5 <= t[i] <= 15:
            # 恒定转向率
            psi_dot[i - 1] = 0.1  # rad/s
        else:
            psi_dot[i - 1] = 0

        # 更新航向角
        psi[i] = psi[i - 1] + psi_dot[i - 1] * dt

        # 更新速度分量
        vx[i] = v * np.cos(psi[i])
        vy[i] = v * np.sin(psi[i])

        # 更新位置
        x[i] = x[i - 1] + vx[i] * dt
        y[i] = y[i - 1] + vy[i] * dt

    # 添加噪声
    pos_noise = 0.5  # 位置噪声
    vel_noise = 0.3  # 速度噪声

    noisy_x = x + np.random.normal(0, pos_noise, n)
    noisy_y = y + np.random.normal(0, pos_noise, n)
    noisy_vx = vx + np.random.normal(0, vel_noise, n)
    noisy_vy = vy + np.random.normal(0, vel_noise, n)

    # 返回真实值和带噪声的测量值
    measurements = np.column_stack([noisy_x, noisy_y, noisy_vx, noisy_vy])
    ground_truth = np.column_stack([x, y, vx, vy])

    return measurements, ground_truth


def analyze_results(measurements, ground_truth, filtered_states, smoothed_states):
    """分析和可视化结果"""
    # 提取数据
    meas_x, meas_y = measurements[:, 0], measurements[:, 1]
    true_x, true_y = ground_truth[:, 0], ground_truth[:, 1]
    filt_x, filt_y = filtered_states[:, 0], filtered_states[:, 1]
    smooth_x, smooth_y = smoothed_states[:, 0], smoothed_states[:, 1]

    # 计算误差
    filt_pos_error = np.sqrt((filt_x - true_x) ** 2 + (filt_y - true_y) ** 2)
    smooth_pos_error = np.sqrt((smooth_x - true_x) ** 2 + (smooth_y - true_y) ** 2)

    print("平均位置误差:")
    print(f"滤波结果: {np.mean(filt_pos_error):.4f} m")
    print(f"平滑结果: {np.mean(smooth_pos_error):.4f} m")

    # 计算速度误差
    true_vx, true_vy = ground_truth[:, 2], ground_truth[:, 3]
    true_speed = np.sqrt(true_vx ** 2 + true_vy ** 2)

    filt_v = filtered_states[:, 2]
    smooth_v = smoothed_states[:, 2]

    filt_vel_error = np.abs(filt_v - true_speed)
    smooth_vel_error = np.abs(smooth_v - true_speed)

    print("\n平均速度误差:")
    print(f"滤波结果: {np.mean(filt_vel_error):.4f} m/s")
    print(f"平滑结果: {np.mean(smooth_vel_error):.4f} m/s")

    # 可视化轨迹
    plt.figure(figsize=(14, 10))

    # 轨迹图
    plt.subplot(2, 1, 1)
    plt.plot(true_x, true_y, 'g-', linewidth=3, label='真实轨迹')
    plt.plot(meas_x, meas_y, 'r.', markersize=4, alpha=0.5, label='测量值')
    plt.plot(filt_x, filt_y, 'b--', linewidth=1.5, label='滤波轨迹')
    plt.plot(smooth_x, smooth_y, 'm-', linewidth=2, label='平滑轨迹')
    plt.title('车辆轨迹估计对比 (CTRV模型)')
    plt.xlabel('X位置 (m)')
    plt.ylabel('Y位置 (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 速度估计
    plt.subplot(2, 2, 3)
    plt.plot(true_speed, 'g-', linewidth=2, label='真实速度')
    plt.plot(filt_v, 'b--', alpha=0.7, label='滤波估计')
    plt.plot(smooth_v, 'm-', alpha=0.9, label='平滑估计')
    plt.title('速度估计')
    plt.xlabel('时间步')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)

    # 位置误差
    plt.subplot(2, 2, 4)
    plt.plot(filt_pos_error, 'b-', label='滤波误差')
    plt.plot(smooth_pos_error, 'm-', label='平滑误差')
    plt.title('位置估计误差')
    plt.xlabel('时间步')
    plt.ylabel('欧氏距离误差 (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ctrv_smoothing_results.png', dpi=150)
    plt.show()

    # 返回误差指标
    return {
        'mean_filtered_pos_error': np.mean(filt_pos_error),
        'mean_smoothed_pos_error': np.mean(smooth_pos_error),
        'mean_filtered_vel_error': np.mean(filt_vel_error),
        'mean_smoothed_vel_error': np.mean(smooth_vel_error)
    }


# 主程序
if __name__ == "__main__":
    # 生成测试数据
    dt = 0.1  # 时间步长 (10Hz)
    measurements, ground_truth = generate_test_data(dt)

    # 创建卡尔曼平滑器
    smoother = CTRVKalmanSmoother(
        dt=dt,
        process_noise_pos=0.05,
        process_noise_vel=0.1,
        process_noise_yaw=0.05,
        measurement_noise_pos=0.5,
        measurement_noise_vel=0.3
    )

    # 处理数据
    filtered_states, smoothed_states = smoother.process(measurements)

    # 分析结果
    results = analyze_results(measurements, ground_truth, filtered_states, smoothed_states)
    print("\n结果摘要:")
    print(f"平滑减少的位置误差: {(1 - results['mean_smoothed_pos_error'] / results['mean_filtered_pos_error']) * 100:.2f}%")
    print(f"平滑减少的速度误差: {(1 - results['mean_smoothed_vel_error'] / results['mean_filtered_vel_error']) * 100:.2f}%")