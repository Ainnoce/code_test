import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def plot_3d_gaussian_primitive(mu, s, rot):
    print("Plotting 3D Gaussian primitive...")

    R_mat = rot.as_matrix()

    # 构建协方差矩阵: Sigma = R S S^T R^T
    S = np.diag(s)
    Sigma = R_mat @ S @ S @ R_mat.T
    Sigma_inv = np.linalg.inv(Sigma)

    # ---------- 生成椭球表面网格 ----------
    # 在球坐标系采样
    theta = np.linspace(0, np.pi, 50)  # 极角
    phi = np.linspace(0, 2 * np.pi, 100)  # 方位角
    theta, phi = np.meshgrid(theta, phi)

    # 单位球面上的点
    x_sphere = np.sin(theta) * np.cos(phi)
    y_sphere = np.sin(theta) * np.sin(phi)
    z_sphere = np.cos(theta)

    # 将球面点通过缩放和旋转变换到椭球面
    # 椭球表面点 = mu + R * S * 单位球面点 (这里的等值面是对应马氏距离=1的椭球)
    points_sphere = np.stack(
        [x_sphere, y_sphere, z_sphere], axis=-1
    )  # (n_theta, n_phi, 3)
    ellipsoid = points_sphere @ S @ R_mat.T + mu  # 注意乘法顺序

    # 提取变换后的坐标
    x_ell, y_ell, z_ell = ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]

    # ---------- 计算每个点的高斯函数值 (用于着色) ----------
    diff = ellipsoid - mu  # (n_theta, n_phi, 3)
    # 向量化的马氏距离平方: (x-mu)^T Sigma^{-1} (x-mu)
    # 对于每个点，计算 d^2 = sum(diff * (Sigma_inv @ diff.T).T, axis=-1)
    # 更高效的方式：利用广播和einsum
    diff_flat = diff.reshape(-1, 3)
    d2 = np.einsum("ni,ij,nj->n", diff_flat, Sigma_inv, diff_flat)
    d2 = d2.reshape(theta.shape)

    # 非归一化高斯值
    g = np.exp(-0.5 * d2)

    # ---------- 绘制 ----------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制椭球表面，颜色映射到高斯值
    surf = ax.plot_surface(
        x_ell,
        y_ell,
        z_ell,
        facecolors=plt.cm.viridis(g),
        rstride=1,
        cstride=1,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    # 标记中心点
    ax.scatter(*mu, color="red", s=80, label="Center μ")

    # 坐标轴标签和标题
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Gaussian Primitive (Ellipsoid)")
    ax.legend()

    # 设置等比例坐标轴（可选，让椭球不变形）
    max_range = np.max([np.ptp(x_ell), np.ptp(y_ell), np.ptp(z_ell)]) * 0.6
    mid_x, mid_y, mid_z = mu
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(g)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="Gaussian value")

    plt.show()


if __name__ == "__main__":
    mu = np.array([0.0, 0.0, 0.0])
    s = np.array([0.5, 0.8, 1.2])
    rot = R.from_euler("xyz", [0, 30, 15], degrees=True)
    print(f"Rotation matrix:\n{rot.as_matrix()}")
    print(f"quaternion: {rot.as_quat()}")
    print(f"Euler angles (degrees): {rot.as_euler('xyz', degrees=True)}")
    print(f"vetor form: {rot.as_rotvec()}")
    print(f"Applied rotation to vector [1, 0, 0]: {rot.apply([1, 0, 0])}")
    plot_3d_gaussian_primitive(mu, s, rot)
