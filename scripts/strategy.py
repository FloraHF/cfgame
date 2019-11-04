#!/usr/bin/env python

import rospy
import tf
import numpy as np
from copy import deepcopy
from math import sin, cos, sqrt, atan2, asin, acos, pi

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped, Twist
from optitrack_broadcast.msg import Mocap


class Strategy():

    def __init__(self, player_id, velocity,
                 worldFrame, frame, rate=10,
                 z=.4, r=.5,
                 player_dict={'D1': 'cf4', 'D2': 'cf5', 'I': 'cf3'}):

        self._worldFrame = worldFrame
        self._frame = frame
        self._player_dict = player_dict
        self._rate = rospy.Rate(rate)
        self._state = None

        self._id = player_id
        self._v = velocity
        self._z = z
        self._r = r
        self._cap_time = .5
        # a = a.split('/')
        # self._a = float(a[0]) / float(a[1])
        # self._LB = acos(self._a)

        self._init_locations = {'D1': np.array([-.6, -0.16]),
                                'D2': np.array([.6, -0.16]),
                                'I': np.array([-0.1, .45])}
        self._locations = deepcopy(self._init_locations)
        self._velocities = {'D1': np.zeros(2), 'D2': np.zeros(2), 'I': np.zeros(2)}
        self._nv = 20
        self._vel_norm = {'D1': [], 'D2': [], 'I': []}
        self._vecs = {'D1_I': np.concatenate((self._locations['I'] - self._locations['D1'], [0])),
                      'D2_I': np.concatenate((self._locations['I'] - self._locations['D2'], [0])),
                      'D1_D2': np.concatenate((self._locations['D2'] - self._locations['D1'], [0]))}

        self._p = None
        self._last_cap = False
        self._end = False
        self._time_inrange = 0.
        self._time_end = 0.

        self._wpts_time = 0.
        self._wpts_segt = 3.

        self._goal_msg = PoseStamped()
        self._updateGoal(goal=self._init_locations[self._id], init=True)

        self._sub_callback_dict = {'D1': self._getD1, 'D2': self._getD2, 'I': self._getI}
        self._subs = dict()
        for p_id, cf_frame in player_dict.items():
            self._subs.update({p_id: rospy.Subscriber('/' + cf_frame + '/mocap', Mocap, self._sub_callback_dict[p_id])})

        self._goal_pub = rospy.Publisher('goal', PoseStamped, queue_size=1)
        self._cmdV_pub = rospy.Publisher('cmdV', Twist, queue_size=1)
        self._policy_pub = rospy.Publisher('policy', String, queue_size=1)
        self._a_pub = rospy.Publisher('a', Float32, queue_size=1)

        rospy.Service('/'+self._player_dict[self._id]+'/set_takeoff', Empty, self._set_takeoff)
        rospy.Service('/'+self._player_dict[self._id]+'/set_play', Empty, self._set_play)
        rospy.Service('/'+self._player_dict[self._id]+'/set_land', Empty, self._set_land)

        self._takeoff = self._service_client('/cftakeoff')
        self._auto = self._service_client('/cfauto')
        self._land = self._service_client('/cfland')
        self._play = self._service_client('/cfplay')

    def _service_client(self, name):
        srv_name = '/' + self._player_dict[self._id] + name
        rospy.wait_for_service(srv_name)
        rospy.loginfo('found' + srv_name + 'service')
        return rospy.ServiceProxy(srv_name, Empty)

    def _set_takeoff(self, req):
        self._state = 'hover'
        rospy.sleep(.11)
        self._takeoff()
        return EmptyResponse()

    def _set_play(self, req):
        self._state = 'play'
        self._play()
        return EmptyResponse()

    def _set_land(self, req):
        self._land()
        return EmptyResponse()

    def _get_time(self):
        t = rospy.Time.now()
        return t.secs + t.nsecs * 1e-9

    def _get_a(self):
        if len(self._vel_norm['D1'])>10 and len(self._vel_norm['D2'])>10 and len(self._vel_norm['I'])>10:
            vd1 = np.array(self._vel_norm['D1']).mean()
            vd2 = np.array(self._vel_norm['D2']).mean()
            vi = np.array(self._vel_norm['I']).mean()
            # print((vd1 + vd2)/(2 * vi))
            a = min((vd1 + vd2)/(2 * vi), 0.99)
        else:
            a = 0.8
        self._a_pub.publish(a)
        return a

    def _getD1(self, data):
        self._locations['D1'] = np.array([data.position[0], data.position[1]])
        self._velocities['D1'] = np.array([data.velocity[0], data.velocity[1]])
        self._vecs['D1_I'] = np.concatenate((self._locations['I'] - self._locations['D1'], [0]))
        self._vecs['D2_D1'] = np.concatenate((self._locations['D2'] - self._locations['D1'], [0]))
        if len(self._vel_norm['D1']) > self._nv:
            self._vel_norm['D1'].pop(0)
        self._vel_norm['D1'].append(sqrt(data.velocity[0] ** 2 + data.velocity[1] ** 2))

    def _getD2(self, data):
        self._locations['D2'] = np.array([data.position[0], data.position[1]])
        self._velocities['D2'] = np.array([data.velocity[0], data.velocity[1]])
        self._vecs['D2_D1'] = np.concatenate((self._locations['D2'] - self._locations['D1'], [0]))
        self._vecs['D2_I'] = np.concatenate((self._locations['D2'] - self._locations['I'], [0]))
        if len(self._vel_norm['D2']) > self._nv:
            self._vel_norm['D2'].pop(0)
        self._vel_norm['D2'].append(sqrt(data.velocity[0] ** 2 + data.velocity[1] ** 2))

    def _getI(self, data):
        self._locations['I'] = np.array([data.position[0], data.position[1]])
        self._velocities['I'] = np.array([data.velocity[0], data.velocity[1]])
        self._vecs['D1_I'] = np.concatenate((self._locations['I'] - self._locations['D1'], [0]))
        self._vecs['D2_I'] = np.concatenate((self._locations['D2'] - self._locations['I'], [0]))
        if len(self._vel_norm['I']) > self._nv:
            self._vel_norm['I'].pop(0)
        self._vel_norm['I'].append(sqrt(data.velocity[0] ** 2 + data.velocity[1] ** 2))

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

    def _get_vecs(self, x):
        D1 = np.concatenate((x['D1'], [0]))
        D2 = np.concatenate((x['D2'], [0]))
        I = np.concatenate((x['I'], [0]))
        D1_I = I - D1
        D2_I = I - D2
        D1_D2 = D2 - D1
        return D1_I, D2_I, D1_D2

    def _get_xyz(self):
        z = np.linalg.norm(self._vecs['D1_D2']) / 2
        x = -np.cross(self._vecs['D1_D2'], self._vecs['D1_I'])[-1] / (2 * z)
        y = np.dot(self._vecs['D1_D2'], self._vecs['D1_I']) / (2 * z) - z
        return x, y, z

    def _get_theta(self):
        k1 = atan2(np.cross(self._vecs['D1_D2'], self._vecs['D1_I'])[-1],
                   np.dot(self._vecs['D1_D2'], self._vecs['D1_I']))  # angle between D1_D2 to D1_I
        k2 = atan2(np.cross(self._vecs['D1_D2'], self._vecs['D2_I'])[-1],
                   np.dot(self._vecs['D1_D2'], self._vecs['D2_I']))  # angle between D1_D2 to D2_I
        tht = k2 - k1
        if k1 < 0:
            tht += 2 * pi
        return tht

    def _get_d(self):
        d1 = max(np.linalg.norm(self._vecs['D1_I']), self._r)
        d2 = max(np.linalg.norm(self._vecs['D2_I']), self._r)
        return d1, d2

    def _get_alpha(self):
        d1, d2 = self._get_d()
        a1 = asin(self._r / d1)
        a2 = asin(self._r / d2)
        return d1, d2, a1, a2

    def _z_strategy(self):
        d1, d2 = self._get_d()
        tht = self._get_theta()
        if self._id == 'D1':
            return -pi / 2
        elif self._id == 'D2':
            return pi / 2
        elif self._id == 'I':
            cpsi = d2 * sin(tht)
            spsi = -(d1 - d2 * cos(tht))
            psi = atan2(spsi, cpsi)
            return psi

    def _h_strategy(self, x, y, z, a):
        x_ = {'D1': np.array([0, -z]),
              'D2': np.array([0, z]),
              'I': np.array([x, y])}

        Delta = sqrt(np.maximum(x ** 2 - (1 - 1 / a ** 2) * (x ** 2 + y ** 2 - (z / a) ** 2), 0))
        if (x + Delta) / (1 - 1 / a ** 2) - x > 0:
            xP = (x + Delta) / (1 - 1 / a ** 2)
        else:
            xP = -(x + Delta) / (1 - 1 / a ** 2)

        P = np.array([xP, 0, 0])
        D1_P = P - np.concatenate((x_['D1'], [0]))
        D2_P = P - np.concatenate((x_['D2'], [0]))
        I_P = P - np.concatenate((x_['I'], [0]))
        D1_I, D2_I, D1_D2 = self._get_vecs(x_)

        if self._id == 'D1':
            phi_1 = atan2(np.cross(D1_I, D1_P)[-1], np.dot(D1_I, D1_P))
            return phi_1
        elif self._id == 'D2':
            phi_2 = atan2(np.cross(D2_I, D2_P)[-1], np.dot(D2_I, D2_P))
            return phi_2
        elif self._id == 'I':
            psi = atan2(np.cross(-D2_I, I_P)[-1], np.dot(-D2_I, I_P))
            return psi

    def _i_strategy(self, d1, d2, a1, a2, tht, a):
        LB = acos(a)
        if self._id == 'D1':
            phi_2 = pi/2 - a2
            # print(phi_2, a2)
            psi = -(pi/2 - LB + a2)
            d = d2*(sin(phi_2)/sin(LB))
            l1 = sqrt(d1**2 + d**2 - 2*d1*d*cos(tht + psi))
            # print(phi_2, d2, d, l1)
            cA = (d**2 + l1**2 - d1**2)/(2*d*l1)
            sA = sin(tht + psi)*(d1/l1)
            A = atan2(sA, cA)
            return -(pi - (tht + psi) - A)
        elif self._id == 'D2':
            return pi/2 - a2
        elif self._id == 'I':
            return -(pi/2 - LB + a2)

    def _m_strategy(self):

        x, y, z = self._get_xyz()
        tht = self._get_theta()
        d1, d2, a1, a2 = self._get_alpha()
        a = self._get_a()

        close = self._r * 1.3
        if np.linalg.norm(self._vecs['D1_I']) < close and np.linalg.norm(self._vecs['D2_I']) < close:  # in both range
            self._policy_pub.publish('both close')
            if self._id == 'D1':
                p = 0
            elif self._id == 'D2':
                p = 0
            elif self._id == 'I':
                p = -tht / 2
        elif np.linalg.norm(self._vecs['D1_I']) < close:  # in D1's range
            self._policy_pub.publish('D1 close')
            if self._id == 'D1':
                p = 0.96 * self._p
            elif self._id == 'D2':
                p = 0
            elif self._id == 'I':
                vD1 = np.concatenate((self._velocities['D1'], [0]))
                phi_1 = atan2(np.cross(self._vecs['D1_I'], vD1)[-1], np.dot(self._vecs['D1_I'], vD1))
                vD1_mag = np.linalg.norm(vD1)
                if self._v > vD1_mag:
                    psi = - abs(acos(vD1_mag * cos(phi_1) / self._v))
                else:
                    psi = - abs(phi_1)
                p = pi - tht + psi
        else:
            if x < -0.05:
                self._policy_pub.publish('h_strategy')
                p = self._h_strategy(x, y, z, a)
            else:
                self._policy_pub.publish('i_strategy')
                p = self._i_strategy(d1, d2, a1, a2, tht, a)

        self._p = p

        if self._id == 'D1':
            p = p + atan2(self._vecs['D1_I'][0], self._vecs['D1_I'][1])
        elif self._id == 'D2':
            p = p + atan2(self._vecs['D2_I'][0], self._vecs['D2_I'][1])
        elif self._id == 'I':
            p = p + atan2(-self._vecs['D2_I'][0], -self._vecs['D2_I'][1])
        return p

    def _hover(self):
        self._updateGoal(goal=self._init_locations[self._id])
        self._goal_pub.publish(self._goal_msg)

    def _set_waypoints(self):
        pts = [np.array([-.5, .5]),
               np.array([-.5, -.5]),
               np.array([.5, -.5]),
               np.array([.5, .5]),
               np.array([-.5, .5]),
               np.array([-.5, -.5]),
               np.array([.5, -.5]),
               np.array([.5, .5]),
               np.array([-.5, .5]),
               np.array([-.5, -.5]),
               np.array([.5, -.5]),
               np.array([.5, .5]),
               np.array([-.5, .5])]
        return pts

    def _waypoints(self, dt, pts):
        # rospy.sleep(10)
        # _t = self._get_time()
        # pts = [np.array([-.5, .5]),
        #        np.array([-.5, -.5]),
        #        np.array([.5, -.5]),
        #        np.array([.5, .5]),
        #        np.array([-.5, .5]),
        #        np.array([-.5, -.5]),
        #        np.array([.5, -.5]),
        #        np.array([.5, .5]),
        #        np.array([-.5, .5]),
        #        np.array([-.5, -.5]),
        #        np.array([.5, -.5]),
        #        np.array([.5, .5]),
        #        np.array([-.5, .5])]
        # step = 3
        # temp = 0
        # pt_id = 0
        # while not rospy.is_shutdown():
            # t = self._get_time()
            # dt = t - _t
            # _t = t
        if self._wpts_time < self._wpts_segt:
            self._wpts_time += dt
        else:
            self._wpts_time = 0
            pt_id = min(pt_id + 1, 12)
        self._updateGoal(goal=pts[pt_id])
        # print(pts[pt_id])
        self._goal_pub.publish(self._goal_msg)

    def _game(self, dt, policy=_m_strategy):
        # _t = self._get_time()
        # _cap = False
        # end = False
        # time_inrange = 0
        # time_end = 0
        # while not rospy.is_shutdown():
        # t = self._get_time()
        # dt = t - _t
        # _t = t
        if not self._end:
            heading = policy(self)
            vx = self._v * cos(heading)
            vy = self._v * sin(heading)
            cmdV = Twist()
            cmdV.linear.x = vx
            cmdV.linear.y = vy
            self._cmdV_pub.publish(cmdV)
            self._updateGoal(self._locations[self._id])
            self._goal_pub.publish(self._goal_msg)
            if self._is_capture():
                if self._last_cap:
                    self._time_inrange += dt
                else:
                    self._time_inrange = 0
                self._last_cap = True
                print(self._time_inrange)
            if self._time_inrange >= self._cap_time:
                self._end = True
                # self._updateGoal(self._locations[self._id])
                # self._goal_pub.publish(self._goal_msg)
                self._auto()
        else:
            self._goal_pub.publish(self._goal_msg)
            if self._id == 'I':
                self._land()

            # self._rate.sleep()

    def iteration(self, event):
        if self._state == 'hover':
            self._hover()
        elif self._state == 'wpts':
            wpts = self._set_waypoints()
            self._waypoints(event.current_real - event.last_real, wpts)
        elif self._state == 'play':
            self._game(event.current_real - event.last_real)
        elif self._state == 'land':
            pass

    # def run(self, frequency):
    #     rospy.init_node('strategy', anonymous=True)
    #     rospy.Timer(rospy.Duration(1.0/frequency), self._iteration)
    #     rospy.spin()

if __name__ == '__main__':
    rospy.init_node('strategy', anonymous=True)

    player_id = rospy.get_param("~player_id", 'D1')
    velocity = rospy.get_param("~velocity", .1)
    cap_range = rospy.get_param("~cap_range", .5)
    z = rospy.get_param("~z", .5)

    worldFrame = rospy.get_param("~worldFrame", "/world")
    frame = rospy.get_param("~frame", "/cf2")

    strategy = Strategy(player_id, velocity,
                        worldFrame, frame,
                        z=z, r=cap_range)

    rospy.Timer(rospy.Duration(1.0/15), strategy.iteration)
    rospy.spin()
