#!/usr/bin/env python

import os
import numpy as np
from math import pi, cos, sin, sqrt
import rospy
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Twist
from crazyflie_game.msg import Mocap
from player_recorder import DataRecorder


class GameRecorder(object):

    def __init__(self, Ds='', Is='',
                 vd=0., vi=0.,
                 r=0., r_close=1.2, k_close=0.9,
                 max_size=1e4,
                 rate=10):

        self._player_dict = dict()
        for i, D in enumerate(Ds):
            if D != '':
                self._player_dict['D'+str(i+1)] = D
        for i, I in enumerate(Is):
            if I != '':
                self._player_dict['I'+str(i+1)] = I

        script_dir = os.path.dirname(__file__)
        self._info_dir = os.path.join(script_dir, 'Results/info.csv')
        with open(self._info_dir, 'a') as f:
            for role, cf in self._player_dict.items():
                f.write(role + ',' + cf + '\n')
            f.write('vd,%.2f'%vd + '\n')
            f.write('vi,%.2f'%vi + '\n')
            f.write('rc,%.2f'%r + '\n')
            f.write('r_close,%.2f'%r_close + '\n')
            f.write('k_close,%.2f'%k_close + '\n')

        self._init_time = self._get_time()
        self._save_interval = 50
        self.rate = rospy.Rate(rate)

        self._locations = {'D1': DataRecorder(max_size=max_size),
                           'D2': DataRecorder(max_size=max_size),
                           'I1': DataRecorder(max_size=max_size)}
        self._headings = {'D1': DataRecorder(max_size=max_size),
                          'D2': DataRecorder(max_size=max_size),
                          'I1': DataRecorder(max_size=max_size)}
        self._location_sub_callback_dict = {'D1': self._getLocD1, 'D2': self._getLocD2, 'I1': self._getLocI}
        self._location_subs = dict()
        for p_id, cf_frame in self._player_dict.items():
            self._location_subs.update({p_id: rospy.Subscriber('/' + cf_frame + '/mocap', Mocap, self._location_sub_callback_dict[p_id])})

        self._heading_sub_callback_dict = {'D1': self._getHeadingD1, 'D2': self._getHeadingD2, 'I1': self._getHeadingI}
        self._heading_subs = dict()
        for p_id, cf_frame in self._player_dict.items():
            self._heading_subs.update(
                {p_id: rospy.Subscriber('/' + cf_frame + '/heading_anl', Float32, self._heading_sub_callback_dict[p_id])})

        self._locs_plot = self._init_locs_plot()

        script_dir = os.path.dirname(__file__)
        # last_res_id = 0
        # for _, dirc, file in os.walk(script_dir):
        #     for d in dirc:
        #         if 'res_' in d:
        #             if int(d.split('_')[-1]) > last_res_id:
        #                 last_res_id = int(d.split('_')[-1])
        self._results_dir = os.path.join(script_dir, 'Results' + '/')
        # print('!!!!!!!!!!!!', self._results_dir)
        # rospy.Publisher('result_dir', String, queue_size=1).publish(self._res)
        if not os.path.isdir(self._results_dir):
            os.makedirs(self._results_dir)

    def _get_time(self):
        t = rospy.Time.now()
        return t.secs + t.nsecs * 1e-9

    def _getLocD1(self, data):
        self._locations['D1'].record(self._get_time() - self._init_time, np.array([data.position[0], data.position[1]]))

    def _getLocD2(self, data):
        self._locations['D2'].record(self._get_time() - self._init_time, np.array([data.position[0], data.position[1]]))

    def _getLocI(self, data):
        self._locations['I1'].record(self._get_time() - self._init_time, np.array([data.position[0], data.position[1]]))

    def _getHeadingD1(self, data):
        self._headings['D1'].record(self._get_time() - self._init_time, data.data)

    def _getHeadingD2(self, data):
        self._headings['D2'].record(self._get_time() - self._init_time, data.data)

    def _getHeadingI(self, data):
        self._headings['I1'].record(self._get_time() - self._init_time, data.data)

    def _init_locs_plot(self):
        fig, ax = plt.subplots(tight_layout=True)
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.grid()
        # gs = gridspec.GridSpec(2, 2)
        # axs = [fig.add_subplot(gs[2, 0]) for i in range(2)]
        # axs.append(fig.add_subplot(gs[:, 1]))
        # axs[0].set_xlabel('t(s)')
        # axs[0].set_ylabel('ID1(m)')
        # axs[1].set_xlabel('t(s)')
        # axs[1].set_ylabel('ID2(m)')
        # axs[2].set_xlabel('x(m)')
        # axs[2].set_ylabel('y(m)')

        plt.show(block=False)
        return {'fig': fig, 'axs': ax}

    def plot_locs(self):
        xD1 = np.asarray(self._locations['D1'].data)
        xD2 = np.asarray(self._locations['D2'].data)
        xI = np.asarray(self._locations['I1'].data)

        hD1 = np.asarray(self._headings['D1'].data)
        hD2 = np.asarray(self._headings['D2'].data)
        hI = np.asarray(self._headings['I1'].data)
        # print(hD1)

        def get_cap_ring(xd):
            rs = []
            for tht in np.linspace(0, 6.28, 50):
                x = xd[0] + .25*cos(tht)
                y = xd[1] + .25*sin(tht)
                rs.append((np.array([x, y])))
            return np.asarray(rs)

        def draw_anl_arrow(ax, x, y, heading):
            vx, vy = cos(heading), sin(heading)
            lenv = 10*sqrt(vx**2 + vy**2)
            vx, vy = vx/lenv, vy/lenv
            ax.arrow(x, y, vx, vy, fc='b', ec='b', head_width=.01, zorder=10)

        if len(xD1) > 100 and len(xD2) > 100 and len(xI) > 100:
            self._locs_plot['axs'].clear()
            self._locs_plot['axs'].plot(xD1[100:-1, 0], xD1[100:-1, 1], 'b')
            self._locs_plot['axs'].plot(xD2[100:-1, 0], xD2[100:-1, 1], 'g')
            self._locs_plot['axs'].plot(xI[100:-1, 0], xI[100:-1, 1], 'r')
            ring1 = get_cap_ring(xD1[-1, :])
            ring2 = get_cap_ring(xD2[-1, :])
            self._locs_plot['axs'].plot(ring1[:, 0], ring1[:, 1], 'b')
            self._locs_plot['axs'].plot(ring2[:, 0], ring2[:, 1], 'g')
            if len(hD1) > 1 and len(hD2) > 1 and len(hI) > 1:
                draw_anl_arrow(self._locs_plot['axs'], xD1[-1, 0], xD1[-1, 1], hD1[-1])
                draw_anl_arrow(self._locs_plot['axs'], xD2[-1, 0], xD2[-1, 1], hD2[-1])
                draw_anl_arrow(self._locs_plot['axs'], xI[-1, 0], xI[-1, 1], hI[-1])
            self._locs_plot['fig'].canvas.draw()
            self._locs_plot['axs'].grid()
            self._locs_plot['axs'].axis('equal')
            if int(len(xI)) % self._save_interval == 0:
                self._locs_plot['fig'].savefig(self._results_dir + 'traj.png')


if __name__ == '__main__':

    rospy.init_node('game_recorder', anonymous=True)

    Ds = rospy.get_param("~Ds", '').split(',')
    Is = rospy.get_param("~Is", '').split(',')
    vd = rospy.get_param("~vd", 0.)
    vi = rospy.get_param("~vi", 0.)
    r = rospy.get_param("~a", 0.)
    r_close = rospy.get_param("~r_close", 1.)
    k_close = rospy.get_param("~k_close", .9)

    recorder = GameRecorder(Ds=Ds, Is=Is, vd=vd, vi=vi, r=rk,
                            r_close=r_close, k_close=k_close)

    while not rospy.is_shutdown():
        recorder.plot_locs()
