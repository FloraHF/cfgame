#!/usr/bin/env python

import rospy
import tf
import numpy as np
from copy import deepcopy
from math import sin, cos, sqrt, atan2, acos, pi
from geometry_msgs.msg import PoseStamped, Twist
from optitrack_broadcast.msg import Mocap

from coords import phy_to_thtd, phy_to_thtalpha

class Strategy():

    def __init__(self, player_id, velocity,
                 worldFrame, frame, rate=10,
                 z=.4, r = .5, a=.8,
                 player_dict={'D1': 'cf4', 'D2': 'cf5', 'I': 'cf3'}):

        self._worldFrame = worldFrame
        self._frame = frame
        self._rate = rospy.Rate(rate)
        self._id = player_id
        self._v = velocity
        self._z = z
        self._r = r
        a = a.split('/')
        self._a = float(a[0])/float(a[1])
        # print(float(a.split('/')[0]))
        self._LB = acos(self._a)
        self._player_dict = player_dict

        self._init_locations = {'D1': np.array([-.6, -0.1]),
                           'D2': np.array([.6, -0.1]),
                           'I': np.array([0., .0])}
        print(self._init_locations['D1'])
        self._locations = deepcopy(self._init_locations)
        self._goal_msg = PoseStamped()
        self.updateGoal(goal=self._init_locations[self._id], init=True)
        self._sub_callback_dict = {'D1': self._getLocD1, 'D2': self._getLocD2, 'I': self._getLocI}
        self._subs = dict()
        for p_id, cf_frame in player_dict.items():
            self._subs.update({p_id: rospy.Subscriber('/' + cf_frame + '/mocap', Mocap, self._sub_callback_dict[p_id])})

        self._goal_pub = rospy.Publisher('goal', PoseStamped, queue_size=1)
        self._cmdV_pub = rospy.Publisher('cmdV', Twist, queue_size=1)


    def _getLocD1(self, data):
        self._locations['D1'] = np.array([data.position[0], data.position[1]])

    def _getLocD2(self, data):
        self._locations['D2'] = np.array([data.position[0], data.position[1]])

    def _getLocI(self, data):
        self._locations['I'] = np.array([data.position[0], data.position[1]])

    def _is_capture(self):
        d1 = np.linalg.norm(self._locations['D1'] - self._locations['I'])
        d2 = np.linalg.norm(self._locations['D2'] - self._locations['I'])
        cap = (d1 < self._r) or (d2 < self._r)
        print('captured:', cap, d1, d2)
        return cap

    def updateGoal(self, goal=None, init=False):

        if init:
            self._goal_msg.header.seq = 0
            self._goal_msg.header.frame_id = self._worldFrame
        else:
            self._goal_msg.header.seq += 1

        self._goal_msg.header.stamp = rospy.Time.now()

        if goal is not None:
            self._goal_msg.pose.position.x = goal[0]
            self._goal_msg.pose.position.y = goal[1]
            self._goal_msg.pose.position.z = self._z
        else:
            self._goal_msg.pose.position.x = self._init_locations[self._id][0]
            self._goal_msg.pose.position.y = self._init_locations[self._id][1]
            self._goal_msg.pose.position.z = self._z

        quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        self._goal_msg.pose.orientation.x = quaternion[0]
        self._goal_msg.pose.orientation.y = quaternion[1]
        self._goal_msg.pose.orientation.z = quaternion[2]
        self._goal_msg.pose.orientation.w = quaternion[3]

        if self._is_capture() and self._id == 'I':
            self._goal_msg.pose.position.z = 0.0

    def _z_strategy(self, x):

        s, bases = phy_to_thtd(x)
        if self._id == 'D1':
            return -pi/2 + atan2(bases[0][1], bases[0][0])
        elif self._id == 'D2':
            return pi/2 + atan2(bases[1][1], bases[1][0])
        elif self._id == 'I':
            cpsi = s[1] * sin(s[2])
            spsi = -(s[0] - s[1] * cos(s[2]))
            psi = atan2(spsi, cpsi)
            return psi + atan2(-bases[1][1], -bases[1][0])

    def _i_strategy(self, x):
        s = phy_to_thtalpha(x)
        ds, bases = phy_to_thtd(x)

        if self._id == 'D1':
            phi_2 = pi / 2 - s[1]
            psi = -(pi / 2 - self._LB / 2 + s[1])
            d = ds[1] * (sin(phi_2) / sin(self._LB / 2))
            l1 = sqrt(ds[0] ** 2 + d ** 2 - 2 * ds[0] * d * cos(s[2] + psi))

            cA = (d ** 2 + l1 ** 2 - ds[0] ** 2) / (2 * d * l1)
            sA = sin(s[2] + psi) * (ds[0] / l1)
            A = atan2(sA, cA)
            phi_1 = -(pi - (s[2] + psi) - A)
            return phi_1 + atan2(bases[0][1], bases[0][0])
        elif self._id == 'D2':
            phi_2 = pi / 2 - s[1]
            return phi_2 + atan2(bases[1][1], bases[1][0])
        elif self._id == 'I':
            psi = -(pi / 2 - self._LB / 2 + s[1])
            return psi + atan2(-bases[1][1], -bases[1][0])

    def hover(self):
        # self._sendCmd(1)
        while not rospy.is_shutdown():
            self.updateGoal(goal=self._init_locations[self._id])
            self._goal_pub.publish(self._goal_msg)
            self._rate.sleep()
            # print(self._id, self._init_locations[self._id])

    # def waypoints(self, wps=[np.array([1., 1.])], ts=[3.], z=0.5, v=0.5):
    #
    #     start_t = rospy.Time.now()
    #     k, km = 0, len(ts)
    #     while not rospy.is_shutdown():
    #         if (rospy.Time.now() - start_t > ts[k]):
    #             k = max(k+1, km-1)
    #         goal = np.concatenate(wp[k], np.array([z]))
    #         self.updateGoal(goal=goal)
    #         self._goal_pub.publish(self._goal_msg)
    #         self._rate.sleep()
    #
    def game(self, policy=_i_strategy):

        while not rospy.is_shutdown():
            heading = policy(self, self._locations)
            vx = self._v * cos(heading)
            vy = self._v * sin(heading)
            cmdV = Twist()
            cmdV.linear.x = vx
            cmdV.linear.y = vy
            self._cmdV_pub.publish(cmdV)
            # print([self._goal[2]])
            # print(self._locations[self._id])
            self.updateGoal()
            self._goal_pub.publish(self._goal_msg)

            self._rate.sleep()


if __name__ == '__main__':
    rospy.init_node('game_strategy', anonymous=True)

    player_id = rospy.get_param("~player_id", 'D1')
    velocity = rospy.get_param("~velocity", .1)
    cap_range = rospy.get_param("~cap_range", .5)
    speed_ratio = rospy.get_param("~speed_ratio", .5)
    z = rospy.get_param("~z", .5)

    worldFrame = rospy.get_param("~worldFrame", "/world")
    frame = rospy.get_param("~frame", "/cf2")

    strategy = Strategy(player_id, velocity,
                        worldFrame, frame,
                        z=z, r=cap_range, a=speed_ratio)

    strategy.game()
