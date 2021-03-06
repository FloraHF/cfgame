#!/usr/bin/env python

import rospy
import tf
import numpy as np
from copy import deepcopy
from math import sin, cos, sqrt, atan2, asin, acos, pi

from tensorflow.keras.models import load_model

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped, Twist
from optitrack_broadcast.msg import Mocap


class Strategy(object):

    def __init__(self, player_id, velocity,
                 worldFrame, frame, rate=10,
                 z=.4, r=.5, a='',
                 r_close=1.2, k_close=.9,
                 Ds='', Is=''):

        self._player_dict = dict()
        for i, D in enumerate(Ds):
            if D != '':
                self._player_dict['D'+str(i+1)] = D
        for i, I in enumerate(Is):
            if I != '':
                self._player_dict['I'+str(i+1)] = I

        self._worldFrame = worldFrame
        self._frame = frame
        self._rate = rospy.Rate(rate)
        self._state = None

        self._id = player_id
        self._v = velocity
        self._z = z
        self._r = r
        self._policy_fn = load_model('PolicyFn_'+self._id)
        self._cap_time = .2
        a = a.split('/')
        self._a_anl = float(a[0]) / float(a[1])
        self._LB = acos(self._a_anl)

        self._init_locations = {'D1': np.array([-0.7077, -0.0895]),
                                'D2': np.array([0.7077, -0.0895]),
                                'I1': np.array([-0.125, 0.642])}
        self._locations = deepcopy(self._init_locations)
        self._velocities = {'D1': np.zeros(2), 'D2': np.zeros(2), 'I1': np.zeros(2)}
        self._nv = 20
        self._vel_norm = {'D1': [], 'D2': [], 'I1': []}
        self._vecs = {'D1_I1': np.concatenate((self._locations['I1'] - self._locations['D1'], [0])),
                      'D2_I1': np.concatenate((self._locations['I1'] - self._locations['D2'], [0])),
                      'D1_D2': np.concatenate((self._locations['D2'] - self._locations['D1'], [0]))}

        self._k_close = k_close
        self._r_close = r_close*r
        self._last_cap = False
        self._end = False
        self._activate = False
        self._time_inrange = 0.
        self._time_end = 0.

        self._wpts_time = 0.
        self._wpts_segt = 3.

        self._goal_msg = PoseStamped()
        self._updateGoal(goal=self._init_locations[self._id], init=True)

        self._sub_callback_dict = {'D1': self._getD1, 'D2': self._getD2, 'I1': self._getI}
        self._subs = dict()
        for p_id, cf_frame in player_dict.items():
            self._subs.update({p_id: rospy.Subscriber('/' + cf_frame + '/mocap', Mocap, self._sub_callback_dict[p_id])})

        self._goal_pub = rospy.Publisher('goal', PoseStamped, queue_size=1)
        self._cmdV_pub = rospy.Publisher('cmdV', Twist, queue_size=1)
        self._policy_pub = rospy.Publisher('policy', String, queue_size=1)
        self._a_pub = rospy.Publisher('a', Float32, queue_size=1)
        self._heading_rel_pub = rospy.Publisher('heading_rel', Float32, queue_size=1)
        self._heading_act_pub = rospy.Publisher('heading_act', Float32, queue_size=1)
        self._heading_anl_pub = rospy.Publisher('heading_anl', Float32, queue_size=1)

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
        self._activate = True
        self._takeoff()
        return EmptyResponse()

    def _set_play(self, req):
        self._state = 'play'
        self._end = False
        self._play()
        return EmptyResponse()

    def _set_land(self, req):
        self._land()
        return EmptyResponse()

    def _get_time(self):
        t = rospy.Time.now()
        return t.secs + t.nsecs * 1e-9

    def _get_a(self):
        if len(self._vel_norm['D1'])>10 and len(self._vel_norm['D2'])>10 and len(self._vel_norm['I1'])>10:
            vd1 = np.array(self._vel_norm['D1']).mean()
            vd2 = np.array(self._vel_norm['D2']).mean()
            vi = np.array(self._vel_norm['I1']).mean()
            # print((vd1 + vd2)/(2 * vi))
            a = min((vd1 + vd2)/(2 * vi), 0.99)
        else:
            a = self._a_anl
        self._a_pub.publish(a)
        return a

    def _getD1(self, data):
        self._locations['D1'] = np.array([data.position[0], data.position[1]])
        self._velocities['D1'] = np.array([data.velocity[0], data.velocity[1]])
        self._vecs['D1_I1'] = np.concatenate((self._locations['I1'] - self._locations['D1'], [0]))
        self._vecs['D2_D1'] = np.concatenate((self._locations['D1'] - self._locations['D2'], [0]))
        if len(self._vel_norm['D1']) > self._nv:
            self._vel_norm['D1'].pop(0)
        self._vel_norm['D1'].append(sqrt(data.velocity[0] ** 2 + data.velocity[1] ** 2))

    def _getD2(self, data):
        self._locations['D2'] = np.array([data.position[0], data.position[1]])
        self._velocities['D2'] = np.array([data.velocity[0], data.velocity[1]])
        self._vecs['D2_D1'] = np.concatenate((self._locations['D1'] - self._locations['D2'], [0]))
        self._vecs['D2_I1'] = np.concatenate((self._locations['I1'] - self._locations['D2'], [0]))
        if len(self._vel_norm['D2']) > self._nv:
            self._vel_norm['D2'].pop(0)
        self._vel_norm['D2'].append(sqrt(data.velocity[0] ** 2 + data.velocity[1] ** 2))

    def _getI(self, data):
        self._locations['I1'] = np.array([data.position[0], data.position[1]])
        self._velocities['I1'] = np.array([data.velocity[0], data.velocity[1]])
        self._vecs['D1_I1'] = np.concatenate((self._locations['I1'] - self._locations['D1'], [0]))
        self._vecs['D2_I1'] = np.concatenate((self._locations['I1'] - self._locations['D2'], [0]))
        if len(self._vel_norm['I1']) > self._nv:
            self._vel_norm['I1'].pop(0)
        self._vel_norm['I1'].append(sqrt(data.velocity[0] ** 2 + data.velocity[1] ** 2))

    def _is_capture(self):
        d1 = np.linalg.norm(self._locations['D1'] - self._locations['I1'])
        d2 = np.linalg.norm(self._locations['D2'] - self._locations['I1'])
        cap = (d1 < self._r) or (d2 < self._r)
        # print(d1, cap)
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

    def _get_xyz(self):
        z = np.linalg.norm(self._vecs['D1_D2']) / 2
        x = -np.cross(self._vecs['D1_D2'], self._vecs['D1_I1'])[-1] / (2 * z)
        y = np.dot(self._vecs['D1_D2'], self._vecs['D1_I1']) / (2 * z) - z
        return x, y, z

    def _get_theta(self):
        k1 = atan2(np.cross(self._vecs['D1_D2'], self._vecs['D1_I1'])[-1],
                   np.dot(self._vecs['D1_D2'], self._vecs['D1_I1']))  # angle between D1_D2 to D1_I
        k2 = atan2(np.cross(self._vecs['D1_D2'], self._vecs['D2_I1'])[-1],
                   np.dot(self._vecs['D1_D2'], self._vecs['D2_I1']))  # angle between D1_D2 to D2_I
        tht = k2 - k1
        if k1 < 0:
            tht += 2 * pi
        return tht

    def _get_d(self):
        d1 = max(np.linalg.norm(self._vecs['D1_I1']), self._r)
        d2 = max(np.linalg.norm(self._vecs['D2_I1']), self._r)
        return d1, d2

    def _get_alpha(self):
        d1, d2 = self._get_d()
        a1 = asin(self._r / d1)
        a2 = asin(self._r / d2)
        return d1, d2, a1, a2

    def _z_strategy(self, a):
        self._policy_pub.publish('z')

        d1, d2 = self._get_d()
        tht = self._get_theta()
        if self._id == 'D1':
            return -pi / 2 + self._base_D1()
        elif self._id == 'D2':
            return pi / 2 + self._base_D2()
        elif self._id == 'I1':
            cpsi = d2 * sin(tht)
            spsi = -(d1 - d2 * cos(tht))
            psi = atan2(spsi, cpsi)
            return psi + self._base_I()

    def _h_strategy(self, x, y, z, a):
        self._policy_pub.publish('h')

        x_ = {'D1': np.array([0, -z]),
              'D2': np.array([0, z]),
              'I1': np.array([x, y])}

        Delta = sqrt(np.maximum(x ** 2 - (1 - 1 / a ** 2) * (x ** 2 + y ** 2 - (z / a) ** 2), 0))
        if (x + Delta) / (1 - 1 / a ** 2) - x > 0:
            xP = (x + Delta) / (1 - 1 / a ** 2)
        else:
            xP = -(x + Delta) / (1 - 1 / a ** 2)

        P = np.array([xP, 0, 0])
        D1_P = P - np.concatenate((x_['D1'], [0]))
        D2_P = P - np.concatenate((x_['D2'], [0]))
        I_P = P - np.concatenate((x_['I1'], [0]))
        D1 = np.concatenate((x_['D1'], [0]))
        D2 = np.concatenate((x_['D2'], [0]))
        I = np.concatenate((x_['I1'], [0]))
        D1_I = I - D1
        D2_I = I - D2
        D1_D2 = D2 - D1

        if self._id == 'D1':
            phi_1 = atan2(np.cross(D1_I, D1_P)[-1], np.dot(D1_I, D1_P)) + self._base_D1()
            return phi_1
        elif self._id == 'D2':
            phi_2 = atan2(np.cross(D2_I, D2_P)[-1], np.dot(D2_I, D2_P)) + self._base_D2()
            return phi_2
        elif self._id == 'I1':
            psi = atan2(np.cross(-D2_I, I_P)[-1], np.dot(-D2_I, I_P)) + self._base_I()
            return psi

    def _i_strategy(self, d1, d2, a1, a2, tht, a):
        self._policy_pub.publish('i')

        LB = acos(a)
        if self._id == 'D1':
            phi_2 = pi/2 - a2 + 0.01
            # print(phi_2, a2)
            psi = -(pi/2 - LB + a2)
            d = d2*(sin(phi_2)/sin(LB))
            l1 = sqrt(d1**2 + d**2 - 2*d1*d*cos(tht + psi))
            # print(phi_2, d2, d, l1)
            cA = (d**2 + l1**2 - d1**2)/(2*d*l1)
            sA = sin(tht + psi)*(d1/l1)
            A = atan2(sA, cA)
            return -(pi - (tht + psi) - A)  + self._base_D1()
        elif self._id == 'D2':
            return pi/2 - a2  + self._base_D2()
        elif self._id == 'I1':
            return -(pi/2 - LB + a2)  + self._base_I()

    def _m_strategy(self, a):
        x, y, z = self._get_xyz()
        if x < -0.05:
            p = self._h_strategy(x, y, z, a)
        else:
            d1, d2, a1, a2 = self._get_alpha()
            p = self._i_strategy(d1, d2, a1, a2, tht, a)
        return p        

    def _nn_strategy(self, a):
        x = np.concatenate((self._locations['D1'], self._locations['I1'], self._locations['D2']))
        return self._policy_fn.predict(x[None])[0]

    def _c_strategy(self, a=None, none_close_strategy=_m_strategy): 

        if a is None:
            a = self._get_a()

        #========== both defenders are close ==========#
        if np.linalg.norm(self._vecs['D1_I1']) < self._r_close and np.linalg.norm(self._vecs['D2_I1']) < self._r_close:  # in both range
            self._policy_pub.publish('both')
            if self._id == 'D1':
                vD1 = np.concatenate((self._velocities['D1'], [0]))
                phi_1 = atan2(np.cross(self._vecs['D1_I1'], vD1)[-1], np.dot(self._vecs['D1_I1'], vD1))
                p = self._k_close*phi_1 + self._base_D1()

            elif self._id == 'D2':
                vD2 = np.concatenate((self._velocities['D2'], [0]))
                phi_2 = atan2(np.cross(self._vecs['D2_I1'], vD2)[-1], np.dot(self._vecs['D2_I1'], vD2))
                p = self._k_close*phi_2 + self._base_D2()

            elif self._id == 'I1':
                p = -self._get_theta()/2 + self._base_I()

        #=============== only D1 is close ===============#
        elif np.linalg.norm(self._vecs['D1_I1']) < self._r_close:  # in D1's range
            vD1 = np.concatenate((self._velocities['D1'], [0]))
            phi_1 = atan2(np.cross(self._vecs['D1_I1'], vD1)[-1], np.dot(self._vecs['D1_I1'], vD1))
            if self._id == 'D1':
                self._policy_pub.publish('D1')
                p = self._k_close * phi_1 + self._base_D1()

            elif self._id == 'D2':
                p = none_close_strategy(self, a)

            elif self._id == 'I1':
                self._policy_pub.publish('D1')
                vD1_mag = np.linalg.norm(vD1)
                if self._v > vD1_mag:
                    psi = - abs(acos(vD1_mag * cos(phi_1) / self._v))
                else:
                    psi = - abs(phi_1)
                p = pi - self._get_theta() + psi + self._base_I()

        #=============== only D2 is close ===============#
        elif np.linalg.norm(self._vecs['D2_I1']) < self._r_close:
            vD2 = np.concatenate((self._velocities['D2'], [0]))
            phi_2 = atan2(np.cross(self._vecs['D2_I1'], vD2)[-1], np.dot(self._vecs['D2_I1'], vD2))
            if self._id == 'D1':
                p = none_close_strategy(self, a)

            if self._id == 'D2':
                self._policy_pub.publish('D2')
                p = self._k_close * phi_2 + self._base_D2()

            elif self._id == 'I1':
                self._policy_pub.publish('D2')
                vD2_mag = np.linalg.norm(vD2)
                if self._v > vD2_mag:
                    psi = abs(acos(vD2_mag * cos(phi_2) / self._v))
                else:
                    psi = abs(phi_2)
                p = psi - pi + self._base_I()

        #============== no defender is close =============#
        else:
            p = none_close_strategy(self, a)           

        return p

    # def _act_strategy(self, D_policy, I_policy):
    #     if self._id == 'I':
    #         p = self._c_strategy(self, none_close_strategy=I_policy)
    #     else:
    #         p = self._c_strategy(self, none_close_strategy=I_policy)
    #     self._heading_rel_pub.publish(p)
    #     heading = self._relative_to_physical(p)
    #     self._heading_act_pub.publish(heading)
    #     return heading

    # def _anl_strategy(self, D_policy, I_policy, a):
    #     if self._id == 'I':
    #         p = self._c_strategy(self, a=a, none_close_strategy=I_policy)
    #     else:
    #         p = self._c_strategy(self, a=a, none_close_strategy=I_policy)
    #     heading = self._relative_to_physical(p)
    #     self._heading_anl_pub.publish(heading)

    def _base_D1(self):
        return atan2(self._vecs['D1_I1'][1], self._vecs['D1_I1'][0])

    def _base_D2(self):
        return atan2(self._vecs['D2_I1'][1], self._vecs['D2_I1'][0])
    
    def _base_I(self):
        return atan2(-self._vecs['D2_I1'][1], -self._vecs['D2_I1'][0])

    def _f_strategy(self, a): # strategy for fast defender

        self._policy_pub.publish('f')

        def target(x): # line
            return x[1]

        def dominant_region(x, a=a):
            xi = self._locations['I1']
            xds = [self._locations['D1'], self._locations['D2']]
            for i, xd in xds:
                if i == 0:
                    inDR = a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - self._r)
                else:
                    inDR = max(inDR, a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - Config.CAP_RANGE))
            return inDR 

        on_dr = NonlinearConstraint(dominant_region, -np.inf, 0)
        xt = minimize(target, self._locations['I1'], constraints=(on_dr,)).x
        vec = np.concatenate((xt - self._locations[self._id], [0]))
        xaxis = np.array([1, 0, 0])

        return atan2(np.cross(xaxis, vec)[-1], np.dot(xaxis, vec))

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
        pt_id = 0
        if self._wpts_time < self._wpts_segt:
            self._wpts_time += dt
        else:
            self._wpts_time = 0
            pt_id = min(pt_id + 1, 12)
        self._updateGoal(goal=pts[pt_id])
        # print(pts[pt_id])
        self._goal_pub.publish(self._goal_msg)

    def _game(self, dt, D_policy=_z_strategy, I_policy=_m_strategy):

        if not self._end:
            if 'D' in self._id:
                heading = self._c_strategy(_z_strategy) # chose from _z_, _m_, _nn_
                # heading = self._nn_strategy() # no adjustment due to close
            elif 'I' in self._id:
                heading = self._c_strategy(_m_strategy) # chose from _z_, _m_, _nn_
                # heading = self._nn_strategy() # no adjustment due to close

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
                # print(self._time_inrange)
            if self._time_inrange > self._cap_time:
                self._end = True
                print(self._id+self._frame+': game end')
                self._auto()
        else:
            self._goal_pub.publish(self._goal_msg)
            # print('landing')
            if self._activate:
                if self._id == 'I1':
                    self._land()
                self._activate = False

    def iteration(self, event):
        if self._state == 'hover':
            self._hover()
        elif self._state == 'wpts':
            wpts = self._set_waypoints()
            t = event.current_real - event.last_real
            self._waypoints(t.secs + t.nsecs*1e-9, wpts)
        elif self._state == 'play':
            t = event.current_real - event.last_real
            self._game(t.secs + t.nsecs*1e-9)
        elif self._state == 'land':
            pass

if __name__ == '__main__':
    rospy.init_node('strategy', anonymous=True)

    Ds = rospy.get_param("~Ds", '').split(',')
    Is = rospy.get_param("~I", '').split(',')
    r_close = rospy.get_param("~r_close", 1.)
    k_close = rospy.get_param("~k_close", .9)

    player_id = rospy.get_param("~player_id", 'D1')
    velocity = rospy.get_param("~velocity", .1)
    cap_range = rospy.get_param("~cap_range", .25)
    z = rospy.get_param("~z", .5)
    a = rospy.get_param("~a", '')

    worldFrame = rospy.get_param("~worldFrame", "/world")
    frame = rospy.get_param("~frame", "/cf2")

    strategy = Strategy(player_id, velocity,
                        worldFrame, frame,
                        z=z, r=cap_range, a=a,
                        Ds=Ds, Is=Is)

    rospy.Timer(rospy.Duration(1.0/15), strategy.iteration)
    rospy.spin()
