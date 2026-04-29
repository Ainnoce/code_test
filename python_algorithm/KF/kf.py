import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from scipy.linalg import block_diag

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
class CTRVKalmanSmoother:
    def __init__(self, dt, max_speed=35.0, max_yaw_rate=1.0, min_speed=0.0,
                 process_noise_pos=0.1, process_noise_vel=0.1,
                 process_noise_yaw=0.01, measurement_noise_pos=0.5,
                 measurement_noise_vel=0.3):
        """
        带约束的CTRV模型卡尔曼平滑器

        参数:
        dt: 时间步长(秒)
        max_speed: 最大允许速度(m/s)
        max_yaw_rate: 最大允许转向率(rad/s)
        min_speed: 最小允许速度(m/s)
        process_noise_*: 过程噪声参数
        measurement_noise_*: 测量噪声参数
        """
        self.dt = dt

        # 状态向量维度: [x, y, vx, vy, psi_dot]
        self.state_dim = 5
        # 测量向量维度: [x, y, vx, vy]
        self.measure_dim = 4

        # 车辆动力学约束
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.min_speed = min_speed

        # 加速度限制
        self.max_lon_acc = 6.0  # 纵向加速度限制 (m/s²)
        self.max_lat_acc = 8.0  # 横向加速度限制 (m/s²)

        # 过程噪声协方差
        self.Q = np.diag([
            process_noise_pos * dt ** 4,  # x位置噪声
            process_noise_pos * dt ** 4,  # y位置噪声
            process_noise_vel * dt ** 2,  # vx速度噪声
            process_noise_vel * dt ** 2,  # vy速度噪声
            process_noise_yaw * dt ** 2  # 转向率噪声
        ])

        # 测量噪声协方差
        self.R = np.diag([
            measurement_noise_pos,  # x位置测量噪声
            measurement_noise_pos,  # y位置测量噪声
            measurement_noise_vel,  # vx速度测量噪声
            measurement_noise_vel  # vy速度测量噪声
        ])

        # 存储历史数据
        self.filtered_states = []
        self.filtered_covs = []
        self.predicted_states = []
        self.predicted_covs = []
        self.transition_jacobians = []
        self.previous_state = None

    def normalize_angle(self, angle):
        """将角度规范到 [0, 2π] 范围内"""
        return angle % (2 * np.pi)

    def angle_difference(self, angle1, angle2):
        """计算两个角度的最小差值，考虑圆周特性"""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def apply_constraints(self, state):
        """
        应用车辆动力学约束到状态向量

        参数:
        state: [x, y, vx, vy, psi_dot]

        返回:
        约束后的状态
        """
        constrained_state = state.copy()

        # 计算速度和航向角
        vx = state[2]
        vy = state[3]
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # 1. 速度约束
        if speed > self.max_speed:
            scale = self.max_speed / speed
            constrained_state[2] *= scale
            constrained_state[3] *= scale
        elif speed < self.min_speed and speed > 0.001:
            scale = self.min_speed / speed
            constrained_state[2] *= scale
            constrained_state[3] *= scale

        # 2. 转向角速度约束
        psi_dot = state[4]
        constrained_state[4] = np.clip(psi_dot, -self.max_yaw_rate, self.max_yaw_rate)

        # 3. 加速度约束
        if self.previous_state is not None:
            dt = self.dt
            prev_vx = self.previous_state[2]
            prev_vy = self.previous_state[3]

            # 计算当前加速度
            ax = (vx - prev_vx) / dt
            ay = (vy - prev_vy) / dt
            acc = np.sqrt(ax ** 2 + ay ** 2)

            # 应用纵向加速度约束
            if acc > self.max_lon_acc:
                scale = self.max_lon_acc / acc
                constrained_state[2] = prev_vx + ax * scale * dt
                constrained_state[3] = prev_vy + ay * scale * dt

            # 横向加速度约束
            psi_dot = constrained_state[4]
            speed = np.sqrt(constrained_state[2] ** 2 + constrained_state[3] ** 2)
            lat_acc = speed * abs(psi_dot)
            if lat_acc > self.max_lat_acc:
                max_yaw_rate = min(self.max_lat_acc / speed, self.max_yaw_rate)
                constrained_state[4] = np.sign(psi_dot) * max_yaw_rate

        return constrained_state

    def _compute_transition(self, state, dt=None):
        """
        CTRV 非线性状态转移函数

        参数:
        state: [x, y, vx, vy, psi_dot]

        返回:
        预测状态
        """
        if dt is None:
            dt = self.dt

        x, y, vx, vy, psi_dot = state

        # 计算航向角和速度
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed > 0.001:
            psi = np.arctan2(vy, vx)
        else:
            psi = 0.0

        # 处理零转向率的情况
        if abs(psi_dot) < 1e-5:
            # 直线运动
            new_x = x + vx * dt
            new_y = y + vy * dt
            new_vx = vx
            new_vy = vy
        else:
            # 曲线运动
            delta_theta = psi_dot * dt
            delta_x = (vx / psi_dot) * (np.sin(psi + delta_theta) - np.sin(psi))
            delta_y = (vy / psi_dot) * (np.cos(psi) - np.cos(psi + delta_theta))

            # 更新位置
            new_x = x + delta_x
            new_y = y + delta_y

            # 更新速度分量
            new_vx = speed * np.cos(psi + delta_theta)
            new_vy = speed * np.sin(psi + delta_theta)

        return np.array([new_x, new_y, new_vx, new_vy, psi_dot])

    def _compute_jacobian(self, state, dt=None):
        """
        计算状态转移函数的雅可比矩阵

        参数:
        state: [x, y, vx, vy, psi_dot]

        返回:
        雅可比矩阵 (5x5)
        """
        if dt is None:
            dt = self.dt

        x, y, vx, vy, psi_dot = state

        # 计算航向角和速度
        speed = np.sqrt(vx ** 2 + vy ** 2)
        if speed > 0.001:
            psi = np.arctan2(vy, vx)
        else:
            psi = 0.0

        # 处理直线运动
        if abs(psi_dot) < 1e-5:
            jac = np.array([
                [1, 0, dt, 0, 0],
                [0, 1, 0, dt, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ])
            return jac

        # 曲线运动情况
        delta_theta = psi_dot * dt

        # 计算偏导数
        # 对x的偏导
        dx_dvx = (np.sin(psi + delta_theta) - np.sin(psi)) / psi_dot
        dx_dvy = 0
        dx_dpsidot = (vx / psi_dot) * (dt * np.cos(psi + delta_theta) -
                                       (np.sin(psi + delta_theta) - np.sin(psi)) / psi_dot)

        # 对y的偏导
        dy_dvx = 0
        dy_dvy = (np.cos(psi) - np.cos(psi + delta_theta)) / psi_dot
        dy_dpsidot = (vy / psi_dot) * (dt * np.sin(psi + delta_theta) -
                                       (np.cos(psi + delta_theta) - np.cos(psi)) / psi_dot)

        # 对vx的偏导
        dvx_dpsi = -speed * np.sin(psi + delta_theta) * dt
        dvx_dpsidot = -speed * np.sin(psi + delta_theta) * dt

        # 对vy的偏导
        dvy_dpsi = speed * np.cos(psi + delta_theta) * dt
        dvy_dpsidot = speed * np.cos(psi + delta_theta) * dt

        # 雅可比矩阵
        jac = np.array([
            [1, 0, dx_dvx, dx_dvy, dx_dpsidot],
            [0, 1, dy_dvx, dy_dvy, dy_dpsidot],
            [0, 0, np.cos(psi + delta_theta), 0, dvx_dpsidot],
            [0, 0, 0, np.sin(psi + delta_theta), dvy_dpsidot],
            [0, 0, 0, 0, 1]
        ])

        return jac

    def predict(self, state, cov):
        """预测步骤"""
        # 应用约束
        constrained_state = self.apply_constraints(state)

        # 非线性状态预测
        predicted_state = self._compute_transition(constrained_state)

        # 计算雅可比矩阵
        F_j = self._compute_jacobian(constrained_state)

        # 预测协方差
        predicted_cov = F_j @ cov @ F_j.T + self.Q

        return predicted_state, predicted_cov, F_j

    def update(self, measurement, predicted_state, predicted_cov):
        """更新步骤"""
        # 测量函数为恒等函数减去最后一维
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])

        # 测量残差
        residual = measurement - H @ predicted_state

        # 卡尔曼增益
        S = H @ predicted_cov @ H.T + self.R
        K = predicted_cov @ H.T @ np.linalg.inv(S)

        # 状态更新
        updated_state = predicted_state + K @ residual

        # 应用约束
        updated_state = self.apply_constraints(updated_state)

        # 协方差更新
        I = np.eye(self.state_dim)
        updated_cov = (I - K @ H) @ predicted_cov

        return updated_state, updated_cov

    def filter(self, measurements):
        """卡尔曼滤波前向处理"""
        self.filtered_states = []
        self.filtered_covs = []
        self.predicted_states = []
        self.predicted_covs = []
        self.transition_jacobians = []
        self.previous_state = None

        # 初始化状态 (使用输入格式)
        init_state = np.array([
            measurements[0, 0],  # x
            measurements[0, 1],  # y
            measurements[0, 2],  # vx
            measurements[0, 3],  # vy
            0.0  # psi_dot (初始化为0)
        ])

        # 初始化协方差
        init_cov = np.diag([
            10.0,  # x位置
            10.0,  # y位置
            5.0,  # vx速度
            5.0,  # vy速度
            1.0  # 转向率
        ])

        current_state = init_state
        current_cov = init_cov

        self.filtered_states.append(current_state)
        self.filtered_covs.append(current_cov)

        for i in range(1, len(measurements)):
            self.previous_state = current_state.copy()

            # 预测步骤
            predicted_state, predicted_cov, F_j = self.predict(current_state, current_cov)
            self.predicted_states.append(predicted_state)
            self.predicted_covs.append(predicted_cov)
            self.transition_jacobians.append(F_j)

            # 更新步骤
            measurement_i = measurements[i]
            updated_state, updated_cov = self.update(measurement_i, predicted_state, predicted_cov)

            # 应用约束
            current_state = self.apply_constraints(updated_state)
            current_cov = updated_cov

            self.filtered_states.append(current_state)
            self.filtered_covs.append(current_cov)

        return np.array(self.filtered_states)

    def rts_smooth(self):
        """RTS平滑后向处理"""
        n = len(self.filtered_states)

        # 初始化平滑结果
        smoothed_states = [self.filtered_states[-1]]

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
            state_diff = smoothed_states[0] - predicted_state_next
            smooth_state = filtered_state + C @ state_diff

            # 应用约束
            smooth_state = self.apply_constraints(smooth_state)

            smoothed_states.insert(0, smooth_state)

        return np.array(smoothed_states)

    def smooth_trajectory(self, measurements):
        """完整的平滑流程"""
        # 先执行前向滤波
        filtered_states = self.filter(measurements)

        # 后向平滑
        smoothed_states = self.rts_smooth()

        # 提取我们需要的x,y,vx,vy（忽略psi_dot）
        filtered_output = filtered_states[:, :4]
        smoothed_output = smoothed_states[:, :4]

        return filtered_output, smoothed_output


# 测试函数
def test_ctrv_smoothing():
    # 设置随机种子以确保可重现性
    np.random.seed(42)

    # 生成测试数据
    dt = 0.1
    total_time = 30
    t = np.arange(0, total_time, dt)
    n = len(t)

    # 创建变化的速度和方向
    speeds = 15 + 5 * np.sin(t * 0.2)  # 速度在10-20 m/s之间变化
    headings = np.cumsum(np.random.normal(0, 0.05, n))  # 航向连续变化

    # 位置
    x = np.zeros(n)
    y = np.zeros(n)

    # 速度分量（包含负值）
    vx = np.zeros(n)
    vy = np.zeros(n)

    # 初始位置
    for i in range(1, n):
        # 更新速度分量
        vx[i] = speeds[i] * np.cos(headings[i])
        vy[i] = speeds[i] * np.sin(headings[i])

        # 更新位置
        x[i] = x[i - 1] + vx[i] * dt
        y[i] = y[i - 1] + vy[i] * dt

    # 添加噪声
    pos_noise = 0.5
    vel_noise = 0.3

    noisy_x = x + np.random.normal(0, pos_noise, n)
    noisy_y = y + np.random.normal(0, pos_noise, n)
    noisy_vx = vx + np.random.normal(0, vel_noise, n)
    noisy_vy = vy + np.random.normal(0, vel_noise, n)

    # 构造测量数据 (x,y,vx,vy)
    measurements = np.column_stack([noisy_x, noisy_y, noisy_vx, noisy_vy])
    ground_truth = np.column_stack([x, y, vx, vy])

    # 创建带约束的平滑器
    smoother = CTRVKalmanSmoother(
        dt=dt,
        max_speed=25.0,
        max_yaw_rate=1.0,
        min_speed=0.0,
        process_noise_pos=0.05,
        process_noise_vel=0.1,
        process_noise_yaw=0.02,
        measurement_noise_pos=0.5,
        measurement_noise_vel=0.3
    )

    # 处理数据
    filtered_output, smoothed_output = smoother.smooth_trajectory(measurements)

    # 可视化结果
    plot_results(measurements, ground_truth, filtered_output, smoothed_output, dt)

    return measurements, ground_truth, filtered_output, smoothed_output


def plot_results(measurements, ground_truth, filtered_output, smoothed_output, dt):
    # 提取数据
    meas_x, meas_y, meas_vx, meas_vy = measurements.T
    true_x, true_y, true_vx, true_vy = ground_truth.T
    filt_x, filt_y, filt_vx, filt_vy = filtered_output.T
    smooth_x, smooth_y, smooth_vx, smooth_vy = smoothed_output.T

    # 计算真实和估计的速度大小
    true_speed = np.sqrt(true_vx ** 2 + true_vy ** 2)
    filt_speed = np.sqrt(filt_vx ** 2 + filt_vy ** 2)
    smooth_speed = np.sqrt(smooth_vx ** 2 + smooth_vy ** 2)

    # 位置误差
    filt_pos_error = np.sqrt((filt_x - true_x) ** 2 + (filt_y - true_y) ** 2)
    smooth_pos_error = np.sqrt((smooth_x - true_x) ** 2 + (smooth_y - true_y) ** 2)

    # 速度误差
    filt_speed_error = np.abs(filt_speed - true_speed)
    smooth_speed_error = np.abs(smooth_speed - true_speed)

    # 创建时间轴
    t = np.arange(0, len(meas_x) * dt, dt)

    # 创建图表
    plt.figure(figsize=(16, 12))

    # 1. 轨迹比较
    plt.subplot(3, 2, 1)
    plt.plot(true_x, true_y, 'g-', linewidth=2, label='真实轨迹')
    plt.plot(meas_x, meas_y, 'r.', alpha=0.4, markersize=3, label='测量值')
    plt.plot(filt_x, filt_y, 'b--', linewidth=1.5, label='滤波轨迹')
    plt.plot(smooth_x, smooth_y, 'm-', linewidth=1.8, label='平滑轨迹')
    plt.title('车辆轨迹估计')
    plt.xlabel('X位置 (m)')
    plt.ylabel('Y位置 (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # 2. 位置误差
    plt.subplot(3, 2, 2)
    plt.plot(t, filt_pos_error, 'b-', label='滤波误差')
    plt.plot(t, smooth_pos_error, 'm-', label='平滑误差')
    plt.title('位置估计误差')
    plt.xlabel('时间 (秒)')
    plt.ylabel('欧氏距离误差 (m)')
    plt.legend()
    plt.grid(True)

    # 3. 速度分量X
    plt.subplot(3, 2, 3)
    plt.plot(t, true_vx, 'g-', label='真实vx')
    plt.plot(t, meas_vx, 'r.', alpha=0.3, markersize=3, label='测量vx')
    plt.plot(t, filt_vx, 'b--', alpha=0.7, label='滤波vx')
    plt.plot(t, smooth_vx, 'm-', alpha=0.9, label='平滑vx')
    plt.title('X方向速度估计')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)

    # 4. 速度分量Y
    plt.subplot(3, 2, 4)
    plt.plot(t, true_vy, 'g-', label='真实vy')
    plt.plot(t, meas_vy, 'r.', alpha=0.3, markersize=3, label='测量vy')
    plt.plot(t, filt_vy, 'b--', alpha=0.7, label='滤波vy')
    plt.plot(t, smooth_vy, 'm-', alpha=0.9, label='平滑vy')
    plt.title('Y方向速度估计')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)

    # 5. 速度大小
    plt.subplot(3, 2, 5)
    plt.plot(t, true_speed, 'g-', label='真实速度')
    plt.plot(t, filt_speed, 'b--', alpha=0.7, label='滤波速度')
    plt.plot(t, smooth_speed, 'm-', alpha=0.9, label='平滑速度')
    plt.axhline(y=25.0, color='r', linestyle='--', label='速度约束')
    plt.title('速度大小估计')
    plt.xlabel('时间 (秒)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)

    # 6. 速度误差
    plt.subplot(3, 2, 6)
    plt.plot(t, filt_speed_error, 'b-', label='滤波速度误差')
    plt.plot(t, smooth_speed_error, 'm-', label='平滑速度误差')
    plt.title('速度估计误差')
    plt.xlabel('时间 (秒)')
    plt.ylabel('绝对误差 (m/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ctrv_smoothing_results.png', dpi=150)
    plt.show()

    # 性能统计
    print("== 性能统计 ==")
    print(f"平均位置误差(滤波): {np.mean(filt_pos_error):.4f} m")
    print(f"平均位置误差(平滑): {np.mean(smooth_pos_error):.4f} m")
    print(f"平均速度误差(滤波): {np.mean(filt_speed_error):.4f} m/s")
    print(f"平均速度误差(平滑): {np.mean(smooth_speed_error):.4f} m/s")
    print(f"平滑减少的位置误差: {(1 - np.mean(smooth_pos_error) / np.mean(filt_pos_error)) * 100:.1f}%")
    print(f"平滑减少的速度误差: {(1 - np.mean(smooth_speed_error) / np.mean(filt_speed_error)) * 100:.1f}%")


if __name__ == "__main__":
    # 运行测试
    measurements, ground_truth, filtered_output, smoothed_output = test_ctrv_smoothing()

    # 保存结果
    np.savetxt('input_measurements.csv', measurements, delimiter=',',
               header='x,y,vx,vy', comments='')
    np.savetxt('smoothed_output.csv', smoothed_output, delimiter=',',
               header='x,y,vx,vy', comments='')