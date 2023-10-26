import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan


class DroneControlSim:
    def __init__(self):
        self.sim_time = 20
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time / self.sim_step), 12))  # 实际状态量
        self.time = np.zeros((int(self.sim_time / self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time / self.sim_step), 3))  # 目标角速率值
        self.attitude_cmd = np.zeros((int(self.sim_time / self.sim_step), 3))  # 目标姿态角值
        self.velocity_cmd = np.zeros((int(self.sim_time / self.sim_step), 3))  # 目标速度值
        self.position_cmd = np.zeros((int(self.sim_time / self.sim_step), 3))  # 目标位置值
        self.pointer = 0

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 4.00e-3
        self.m = 0.5
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0, .0], [.0, self.I_yy, .0], [.0, .0, self.I_zz]])

    def run(self):
        for self.pointer in range(self.drone_states.shape[0] - 1):
            self.time[self.pointer] = self.pointer * self.sim_step

            # 指定一条飞行路线
            self.position_cmd[self.pointer, 0] = self.pointer * 0.001
            self.position_cmd[self.pointer, 1] = 20 * sin((self.pointer + 1) / 2000)
            self.position_cmd[self.pointer, 2] = 20 * sin((self.pointer + 1) / 2000)

            # 根据规划的路径生成要求的输入T和M
            v_cmd = self.position_controller(self.position_cmd[self.pointer, :])
            _, T_cmd = self.velocity_controller(v_cmd)
            r_cmd = self.attitude_controller(self.attitude_cmd[self.pointer, :])
            M_cmd = self.rate_controller(r_cmd)

            # 根据T和M来更新无人机当前的状态
            dx = self.drone_dynamics(T_cmd, M_cmd)
            self.drone_states[self.pointer + 1, :] = self.drone_states[self.pointer, :] + self.sim_step * dx

        self.time[-1] = self.sim_time

    def drone_dynamics(self, T, M):
        # Input:
        # T: float Thrust
        # M: np.array (3,)  Moments in three axes
        # Output: np.array (12,) the derivative (dx) of the drone 

        x = self.drone_states[self.pointer, 0]
        y = self.drone_states[self.pointer, 1]
        z = self.drone_states[self.pointer, 2]
        vx = self.drone_states[self.pointer, 3]
        vy = self.drone_states[self.pointer, 4]
        vz = self.drone_states[self.pointer, 5]
        phi = self.drone_states[self.pointer, 6]  # x
        theta = self.drone_states[self.pointer, 7]  # y
        psi = self.drone_states[self.pointer, 8]  # z
        p = self.drone_states[self.pointer, 9]
        q = self.drone_states[self.pointer, 10]
        r = self.drone_states[self.pointer, 11]

        # 姿态角变化率与绕机体轴角速度的转换矩阵
        R_d_angle = np.array([[1, tan(theta) * sin(phi), tan(theta) * cos(phi)],
                              [0, cos(phi), -sin(phi)],
                              [0, sin(phi) / cos(theta), cos(phi) / cos(theta)]])
        # Earth -> Body 的转换矩阵
        R_E_B = np.array([[cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
                          [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
                           sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), sin(phi) * cos(theta)],
                          [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
                           cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), cos(phi) * cos(theta)]])

        d_position = np.array([vx, vy, vz])
        d_velocity = np.array([.0, .0, self.g]) + R_E_B.transpose() @ np.array([.0, .0, T]) / self.m
        d_angle = R_d_angle @ np.array([p, q, r])
        d_q = np.linalg.inv(self.I) @ (M - np.cross(np.array([p, q, r]), self.I @ np.array([p, q, r])))

        dx = np.concatenate((d_position, d_velocity, d_angle, d_q))
        return dx

    def rate_controller(self, cmd):
        # Input: cmd np.array (3,) rate commands
        # Output: M np.array (3,) moments
        k_ra = 1
        m_cmd = (cmd - self.drone_states[self.pointer, [9, 10, 11]]) * k_ra
        return m_cmd

    def attitude_controller(self, cmd):
        # Input: cmd np.array (3,) attitude commands
        # Output: M np.array (3,) rate commands
        k_at = 5.5
        r_cmd = (cmd - self.drone_states[self.pointer, [6, 7, 8]]) * k_at
        self.rate_cmd[self.pointer, :] = r_cmd
        return r_cmd

    def velocity_controller(self, cmd):
        # Input: cmd np.array (3,) velocity commands
        # Output: M np.array (2,) phi and theta commands and thrust cmd
        k_t = 7
        k_angle = 3.2
        err = (cmd - self.drone_states[self.pointer, [3, 4, 5]])

        t_cmd = (err[2] * k_t - self.g) * self.m / (cos(self.drone_states[self.pointer, 6]) *
                                                    cos(self.drone_states[self.pointer, 7]))

        theta_cmd = err[0] * k_angle / t_cmd
        phi_cmd = -err[1] * k_angle / t_cmd
        att_cmd = np.array([phi_cmd, theta_cmd])
        self.attitude_cmd[self.pointer, 0:2] = att_cmd
        return att_cmd, t_cmd

    def position_controller(self, cmd):
        # Input: cmd np.array (3,) position commands
        # Output: M np.array (3,) velocity commands
        k_po = 1.8
        v_cmd = (cmd - self.drone_states[self.pointer, [0, 1, 2]]) * k_po
        self.velocity_cmd[self.pointer, :] = v_cmd
        return v_cmd

    def plot_states(self):
        fig1, ax1 = plt.subplots(4, 3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0, 0].plot(self.time, self.drone_states[:, 0], label='real')
        ax1[0, 0].plot(self.time, self.position_cmd[:, 0], label='cmd')
        ax1[0, 0].set_ylabel('x[m]')
        ax1[0, 1].plot(self.time, self.drone_states[:, 1])
        ax1[0, 1].plot(self.time, self.position_cmd[:, 1])
        ax1[0, 1].set_ylabel('y[m]')
        ax1[0, 2].plot(self.time, self.drone_states[:, 2])
        ax1[0, 2].plot(self.time, self.position_cmd[:, 2])
        ax1[0, 2].set_ylabel('z[m]')
        ax1[0, 0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1, 0].plot(self.time, self.drone_states[:, 3])
        ax1[1, 0].plot(self.time, self.velocity_cmd[:, 0])
        ax1[1, 0].set_ylabel('vx[m/s]')
        ax1[1, 1].plot(self.time, self.drone_states[:, 4])
        ax1[1, 1].plot(self.time, self.velocity_cmd[:, 1])
        ax1[1, 1].set_ylabel('vy[m/s]')
        ax1[1, 2].plot(self.time, self.drone_states[:, 5])
        ax1[1, 2].plot(self.time, self.velocity_cmd[:, 2])
        ax1[1, 2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2, 0].plot(self.time, self.drone_states[:, 6])
        ax1[2, 0].plot(self.time, self.attitude_cmd[:, 0])
        ax1[2, 0].set_ylabel('phi[rad]')
        ax1[2, 1].plot(self.time, self.drone_states[:, 7])
        ax1[2, 1].plot(self.time, self.attitude_cmd[:, 1])
        ax1[2, 1].set_ylabel('theta[rad]')
        ax1[2, 2].plot(self.time, self.drone_states[:, 8])
        ax1[2, 2].plot(self.time, self.attitude_cmd[:, 2])
        ax1[2, 2].set_ylabel('psi[rad]')

        self.rate_cmd[-1] = self.rate_cmd[-2]
        ax1[3, 0].plot(self.time, self.drone_states[:, 9])
        ax1[3, 0].plot(self.time, self.rate_cmd[:, 0])
        ax1[3, 0].set_ylabel('p[rad/s]')
        ax1[3, 1].plot(self.time, self.drone_states[:, 10])
        ax1[3, 1].plot(self.time, self.rate_cmd[:, 1])
        ax1[3, 0].set_ylabel('q[rad/s]')
        ax1[3, 2].plot(self.time, self.drone_states[:, 11])
        ax1[3, 2].plot(self.time, self.rate_cmd[:, 2])
        ax1[3, 0].set_ylabel('r[rad/s]')


if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.tight_layout()
    plt.show()
