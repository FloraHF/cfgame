#!/usr/bin/env python

import os
import numpy as np
from math import atan2, asin
import rospy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Twist
from crazyflie_game.msg import Mocap


class DataRecorder(object):
    def __init__(self, max_size=1e3):
        self.max_size = max_size
        self.cur_size = 0
        self.data = []
        self.time = []

    def record(self, t, data):
        if self.cur_size >= self.max_size:
            self.data.pop(0)
            self.time.pop(0)
            self.cur_size -= 1
        self.data.append(data)
        self.time.append(t)
        self.cur_size += 1


class Plotter(object):

    def __init__(self, cf_id="/cf2",
                 goal="/cf2/goal",
                 cmdV='/cf2/cmdV',
                 cmd_vel='cf2/cmd_vel',
                 Vtemp='/cf2/cmdVtemp',
                 mocap="cf2/mocap",
                 max_size=1e4,
                 rate=10):

        self._init_time = self._get_time()
        self._save_interval = 50

        self._locs = DataRecorder(max_size=max_size)
        self._goals = DataRecorder(max_size=max_size)
        self._cmdVs = DataRecorder(max_size=max_size)
        self._vels = DataRecorder(max_size=max_size)
        self._eulers = DataRecorder(max_size=max_size)

        self._cmd_vels = DataRecorder(max_size=max_size)

        self.rate = rospy.Rate(rate)
        self._goal_sub = rospy.Subscriber(goal, PoseStamped, self._update_goal)
        self._mocap_sub = rospy.Subscriber(mocap, Mocap, self._update_mocap)
        # self._cmdV_sub = rospy.Subscriber(cmdV, Float32MultiArray, self._update_cmdV)
        self._cmdVtemp_sub = rospy.Subscriber(Vtemp, Twist, self._update_cmdVtemp)
        self._cmd_vel_sub = rospy.Subscriber(cmd_vel, Twist, self._update_cmd_vel)

        self._locs_plot = self._init_locs_plot()
        self._euler_plot = self._init_euler_plot()
        self._vels_plot = self._init_vels_plot()

        script_dir = os.path.dirname(__file__)
        self._results_dir = os.path.join(script_dir, 'Results/')
        if not os.path.isdir(self._results_dir):
            os.makedirs(self._results_dir)

    def _get_time(self):
        t = rospy.Time.now()
        return t.secs + t.nsecs * 1e-9

    def _qt_to_euler(self, mocap):
        quat = np.zeros(4)
        quat[0] = mocap.quaternion[0]
        quat[1] = mocap.quaternion[1]
        quat[2] = mocap.quaternion[2]
        quat[3] = mocap.quaternion[3]
        ans = np.zeros(3)
        ans[0] = atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                       1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
        ans[1] = asin(2.0 * (quat[2] * quat[0] - quat[3] * quat[1]))
        ans[2] = atan2(2.0 * (quat[3] * quat[0] + quat[1] * quat[2]),
                       1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]))
        return ans

    def _init_locs_plot(self):
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(3, 2)
        axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        axs.append(fig.add_subplot(gs[:, 1]))

        y_labels = ['x(m)', 'y(m)', 'z(m)']
        for ax, y_label in zip(axs[:3], y_labels[:3]):
            ax.set_xlabel('t(s)')
            ax.set_ylabel(y_label)
            ax.grid()
        axs[-1].set_xlabel('x(m)')
        axs[-1].set_ylabel('y(m)')
        axs[-1].grid()

        plt.show(block=False)
        return {'fig': fig, 'axs': axs}

    def _init_euler_plot(self):
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(4, 1)
        axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
        # axs.append(fig.add_subplot(gs[:, 1]))

        y_labels = ['cmd_vel_x', 'cmd_vel_y', 'cmd_vel_z', 'yaw']
        for ax, y_label in zip(axs[:3], y_labels[:3]):
            ax.set_xlabel('t(s)')
            ax.set_ylabel(y_label)
            ax.grid()
        axs[-1].set_ylim(40000, 45000)
        # axs[-1].set_xlabel('x(m)')
        # axs[-1].set_ylabel('y(m)')
        # axs[-1].grid()

        plt.show(block=False)
        return {'fig': fig, 'axs': axs}

    def _init_vels_plot(self):
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(3, 1)
        axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        y_labels = ['vx(m/s)', 'vy(m/s)', 'vz(m/s)']
        for ax, y_label in zip(axs, y_labels):
            ax.set_xlabel('t(s)')
            ax.set_ylabel(y_label)
            ax.grid()
        plt.show(block=False)
        return {'fig': fig, 'axs': axs}

    def _update_mocap(self, mocap):
        self._locs.record(self._get_time() - self._init_time, mocap.position)
        self._vels.record(self._get_time() - self._init_time, mocap.velocity)
        self._eulers.record(self._get_time() - self._init_time, self._qt_to_euler(mocap))

    def _update_goal(self, goal):
        # t = goal.header.stamp.secs
        x = goal.pose.position.x
        y = goal.pose.position.y
        z = goal.pose.position.z
        self._goals.record(self._get_time() - self._init_time, np.array([x, y, z]))

    def _update_cmd_vel(self, cmd):
        x = cmd.linear.x
        y = cmd.linear.y
        z = cmd.linear.z
        self._cmd_vels.record(self._get_time() - self._init_time, np.array([x, y, z]))

    def _update_cmdVtemp(self, cmdVtemp):
        x = cmdVtemp.linear.x
        y = cmdVtemp.linear.y
        z = cmdVtemp.linear.z
        self._cmdVs.record(self._get_time() - self._init_time, np.array([x, y, z]))

    def _update_cmdV(self, cmdV):
        self._cmdVs.record(self._get_time() - self._init_time, cmdV.data)

    def plot_locs(self, L):
        t_loc = self._locs.time
        xyz_loc = np.asarray(self._locs.data)
        t_goal = self._goals.time
        xyz_goal = np.asarray(self._goals.data)

        l1 = min(len(t_loc), len(xyz_loc))
        l2 = min(len(t_goal), len(xyz_goal))

        lb1, ub1 = max(0, l1 - L), l1 - 1
        lb2, ub2 = max(0, l2 - L), l2 - 1
        if l1 > 1 and l2 > 1:
            for ax in self._locs_plot['axs']:
                ax.clear()
            self._locs_plot['axs'][0].plot(t_goal[lb2:ub2], xyz_goal[lb2:ub2, 0], 'r--')
            self._locs_plot['axs'][1].plot(t_goal[lb2:ub2], xyz_goal[lb2:ub2, 1], 'b--')
            self._locs_plot['axs'][2].plot(t_goal[lb2:ub2], xyz_goal[lb2:ub2, 2], 'g--')
            self._locs_plot['axs'][0].plot(t_loc[lb1:ub1], xyz_loc[lb1:ub1, 0], 'r-')
            self._locs_plot['axs'][1].plot(t_loc[lb1:ub1], xyz_loc[lb1:ub1, 1], 'b-')
            self._locs_plot['axs'][2].plot(t_loc[lb1:ub1], xyz_loc[lb1:ub1, 2], 'g-')
            # self._locs_plot['axs'][3].plot(xyz_goal[:ub2, 0], xyz_goal[:ub2, 1], 'b--')
            self._locs_plot['axs'][3].plot(xyz_loc[2:ub1, 0], xyz_loc[2:ub1, 1], 'b-')
            self._locs_plot['axs'][3].plot(-0.2, 0.62, 'ro')
            self._locs_plot['axs'][3].plot(-0.2, -1.17, 'ro')
            self._locs_plot['fig'].canvas.draw()
            for ax in self._locs_plot['axs']:
                ax.grid()
            if int(t_loc[-1]) % self._save_interval == 0:
                self._locs_plot['fig'].savefig(self._results_dir + 'hovering_locs.png')

    def plot_eulers(self, L):
        t = self._cmd_vels.time
        cmd = np.asarray(self._cmd_vels.data)
        t_euler = self._eulers.time
        euler = np.asarray(self._eulers.data)

        l1 = min(len(t), len(cmd))
        l2 = min(len(t_euler), len(euler))

        lb1, ub1 = max(0, l1 - L), l1 - 1
        lb2, ub2 = max(0, l2 - L), l2 - 1

        if l1 > 1 and l2 > 1:
            for ax in self._euler_plot['axs']:
                ax.clear()
            self._euler_plot['axs'][0].plot(t[lb1:ub1], 10*cmd[lb1:ub1, 0], 'b--')
            self._euler_plot['axs'][1].plot(t[lb1:ub1], 10*cmd[lb1:ub1, 1], 'r--')
            self._euler_plot['axs'][2].plot(t[lb1:ub1], cmd[lb1:ub1, 2], 'g--')
            self._euler_plot['axs'][0].plot(t_euler[lb2:ub2], 57.3*euler[lb2:ub2, 0], 'b-')
            self._euler_plot['axs'][1].plot(t_euler[lb2:ub2], 57.3*euler[lb2:ub2, 1], 'r-')
            self._euler_plot['axs'][3].plot(t_euler[lb2:ub2], 57.3*euler[lb2:ub2, 2], 'm-')
            # self._euler_plot['axs'][2].plot(t_euler[lb2:ub2], euler[lb2:ub2, 2], 'g-')
            self._euler_plot['fig'].canvas.draw()
            for ax in self._euler_plot['axs']:
                ax.grid()
            if int(t[-1]) % self._save_interval == 0:
                self._euler_plot['fig'].savefig(self._results_dir + 'hovering_euler.png')

    def plot_vels(self, L):
        t_vel = self._vels.time
        vel = np.asarray(self._vels.data)
        t_cmd = self._cmdVs.time
        cmd = np.asarray(self._cmdVs.data)
        l2 = min(len(t_vel), len(vel))
        l1 = min(len(t_cmd), len(cmd))
        # L1 = min(L, l1)
        # L2 = min(L, l2)
        lb1, ub1 = max(0, l1 - L), max(0, l1 - 1)
        lb2, ub2 = max(0, l2 - L), max(0, l2 - 1)

        if l2 > 1 and l1 > 1:
            # print(cmd[:, 0])
            # print(lb1, ub1)
            for ax in self._vels_plot['axs']:
                ax.clear()
            self._vels_plot['axs'][0].plot(t_cmd[lb1:ub1], cmd[lb1:ub1, 0], 'r--')
            self._vels_plot['axs'][1].plot(t_cmd[lb1:ub1], cmd[lb1:ub1, 1], 'b--')
            self._vels_plot['axs'][2].plot(t_cmd[lb1:ub1], cmd[lb1:ub1, 2], 'g--')
            self._vels_plot['axs'][0].plot(t_vel[lb2:ub2], vel[lb2:ub2, 0], 'r')
            self._vels_plot['axs'][1].plot(t_vel[lb2:ub2], vel[lb2:ub2, 1], 'b')
            self._vels_plot['axs'][2].plot(t_vel[lb2:ub2], vel[lb2:ub2, 2], 'g')
            self._vels_plot['fig'].canvas.draw()
            for ax in self._vels_plot['axs']:
                ax.grid()
            if int(t_vel[-1]) % self._save_interval == 0:
                self._vels_plot['fig'].savefig(self._results_dir + 'hovering_vels.png')


if __name__ == '__main__':
    rospy.init_node('player_recorder', anonymous=True)


    cf_id = rospy.get_param("~cf_frame", "/cf3")

    plotter = Plotter(cf_id=cf_id,
                      goal=cf_id+'/goal',
                      cmdV=cf_id+'/cmdV',
                      cmd_vel=cf_id+'/cmd_vel',
                      Vtemp=cf_id+'/cmdVtemp',
                      mocap=cf_id+'/mocap')
    L = 1000
    while not rospy.is_shutdown():
        plotter.plot_locs(L)
        plotter.plot_eulers(L)
        plotter.plot_vels(L)
        plotter.rate.sleep()
