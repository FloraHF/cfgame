#!/usr/bin/env python

import rospy
import tf
import numpy as np
from copy import deepcopy
from math import sin, cos, sqrt, atan2, acos, pi

from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped, Twist
from optitrack_broadcast.msg import Mocap

from coords import phy_to_thtd, phy_to_thtalpha, phy_to_xyz


class Strategy():

    def __init__(self, player_id, velocity,
                 worldFrame, frame, rate=10,
                 z=.4, r=.5, a=.8,
                 player_dict={'D1': 'cf4', 'D2': 'cf5', 'I': 'cf3'}):

        self._worldFrame = worldFrame
        self._frame = frame
        self._player_dict = player_dict
        self._rate = rospy.Rate(rate)

        self._id = player_id
        self._v = velocity
        self._z = z
        self._r = r
        self._cap_time = 2.
        a = a.split('/')
        self._a = float(a[0]) / float(a[1])
        self._LB = acos(self._a)

        self._init_locations = {'D1': np.array([-.6, -0.0]),
                                'D2': np.array([.6, -0.0]),
                                'I': np.array([0., 0.2])}
        self._locations = deepcopy(self._init_locations)

        self._goal_msg = PoseStamped()
        self._updateGoal(goal=self._init_locations[self._id], init=True)

        self._sub_callback_dict = {'D1': self._getLocD1, 'D2': self._getLocD2, 'I': self._getLocI}
        self._subs = dict()
        for p_id, cf_frame in player_dict.items():
            self._subs.update({p_id: rospy.Subscriber('/' + cf_frame + '/mocap', Mocap, self._sub_callback_dict[p_id])})

        self._goal_pub = rospy.Publisher('goal', PoseStamped, queue_size=1)
        self._cmdV_pub = rospy.Publisher('cmdV', Twist, queue_size=1)

        # print(self._goal_msg.pose.position.x, self._goal_msg.pose.position.y)

        srv_name = '/' + player_dict[self._id] + '/cftakeoff'
        rospy.wait_for_service(srv_name)
        rospy.loginfo('found' + srv_name + 'service')
        self.takeoff = rospy.ServiceProxy(srv_name, Empty)

        srv_name = '/' + player_dict[self._id] + '/cfauto'
        rospy.wait_for_service(srv_name)
        rospy.loginfo('found' + srv_name + 'service')
        self.auto = rospy.ServiceProxy(srv_name, Empty)

        srv_name = '/' + player_dict[self._id] + '/cfland'
        rospy.wait_for_service(srv_name)
        rospy.loginfo('found' + srv_name + 'service')
        self.land = rospy.ServiceProxy(srv_name, Empty)

    def _get_time(self):
        t = rospy.Time.now()
        return t.secs + t.nsecs * 1e-9

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
        # print('captured:', cap, d1, d2)
        return cap

    def _updateGoal(self, goal=None, init=False):

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

    def _z_strategy(self, x):

        s, bases = phy_to_thtd(x)
        if self._id == 'D1':
            return -pi / 2 + atan2(bases[0][1], bases[0][0])
        elif self._id == 'D2':
            return pi / 2 + atan2(bases[1][1], bases[1][0])
        elif self._id == 'I':
            cpsi = s[1] * sin(s[2])
            spsi = -(s[0] - s[1] * cos(s[2]))
            psi = atan2(spsi, cpsi)
            return psi + atan2(-bases[1][1], -bases[1][0])

    def _h_strategy(self, x):

        s = phy_to_xyz(x)
        x_ = {'D1': np.array([0, -s[2]]),
              'D2': np.array([0, s[2]]),
              'I': np.array([s[0], s[1]])}

        Delta = sqrt(
            np.maximum(s[0] ** 2 - (1 - 1 / self._a ** 2) * (s[0] ** 2 + s[1] ** 2 - (s[2] / self._a) ** 2), 0))
        if (s[0] + Delta) / (1 - 1 / self._a ** 2) - s[0] > 0:
            xP = (s[0] + Delta) / (1 - 1 / self._a ** 2)
        else:
            xP = - (s[0] + Delta) / (1 - 1 / self._a ** 2)

        P = np.array([xP, 0, 0])
        D1_P = P - np.concatenate((x_['D1'], [0]))
        D2_P = P - np.concatenate((x_['D2'], [0]))
        I_P = P - np.concatenate((x_['I'], [0]))
        D1_I, D2_I, D1_D2 = get_vecs(x_)

        if self._id == 'D1':
            phi_1 = atan2(np.cross(D1_I, D1_P)[-1], np.dot(D1_I, D1_P))
            return phi_1 + atan2(D1_I[1], D1_I[0])
        elif self._id == 'D2':
            phi_2 = atan2(np.cross(D2_I, D2_P)[-1], np.dot(D2_I, D2_P))
            return phi_2 + atan2(D2_I[1], D2_I[0])
        elif self._id == 'I':
            psi = atan2(np.cross(-D2_I, I_P)[-1], np.dot(-D2_I, I_P))
            return psi + atan2(-D2_I[1], -D2_I[0])

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

    def _m_strategy(self, x):

        s = phy_to_xyz(x)
        if s[0] < -0.1:
            return self._h_strategy(x)
        else:
            return self._i_strategy(x)

    def hover(self):
        while not rospy.is_shutdown():
            self._updateGoal(goal=self._init_locations[self._id])
            self._goal_pub.publish(self._goal_msg)
            self._rate.sleep()

    def waypoints(self):
        rospy.sleep(10)
        _t = self._get_time()
        pts = [np.array([-.5,  .5]),
               np.array([-.5, -.5]),
               np.array([ .5, -.5]),
               np.array([ .5,  .5]),
               np.array([-.5,  .5])]
        step = 3
        temp = 0
        pt_id = 0
        while not rospy.is_shutdown():
            t = self._get_time()
            dt = t - _t
            _t = t
            if temp < step:
                temp += dt
            else:
                temp = 0
                pt_id = min(pt_id+1, 4)
            self._updateGoal(goal=pts[pt_id])
            # print(pts[pt_id])
            self._goal_pub.publish(self._goal_msg)
            self._rate.sleep()


    def game(self, policy=_z_strategy):

        _t = self._get_time()
        _cap = False
        end = False
        time_inrange = 0
        time_end = 0
        while not rospy.is_shutdown():
            t = self._get_time()
            dt = t - _t
            _t = t
            if not end:
                if self._is_capture():
                    cap = True
                    if _cap:
                        time_inrange += dt
                    else:
                        time_inrange = 0
                    _cap = cap
                if time_inrange >= self._cap_time:
                    end = True
                    self._updateGoal(self._locations[self._id])
                    self._goal_pub.publish(self._goal_msg)
                    self._auto()
                    continue
                heading = policy(self, self._locations)
                vx = self._v * cos(heading)
                vy = self._v * sin(heading)
                cmdV = Twist()
                cmdV.linear.x = vx
                cmdV.linear.y = vy
                self._cmdV_pub.publish(cmdV)
                self._updateGoal()
                self._goal_pub.publish(self._goal_msg)
            else:
                self._goal_pub.publish(self._goal_msg)
                if self._id == 'I':
                    self._land()

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

    # if strategy._id =='D1':
    #     print('takeoff')
    #     strategy.takeoff()
    strategy.waypoints()