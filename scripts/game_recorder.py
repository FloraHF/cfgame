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
from plotter import DataRecorder

class GameRecorder(object):

    def __init__(self, D1='cf4', D2='cf5', I='cf3',
                 max_size=1e4,
                 rate=10):

        self._init_time = self._get_time()
        self._save_interval = 50
        self._player_dict = {'D1': D1, 'D2': D2, 'I': I}
        self.rate = rospy.Rate(rate)

        self._locations = {'D1': DataRecorder(max_size=max_size), 
                            'D2': DataRecorder(max_size=max_size), 
                            'I': DataRecorder(max_size=max_size)}
        self._sub_callback_dict = {'D1': self._getLocD1, 'D2': self._getLocD2, 'I': self._getLocI}
        self._subs = dict()
        for p_id, cf_frame in self._player_dict.items():
            self._subs.update({p_id: rospy.Subscriber('/' + cf_frame + '/mocap', Mocap, self._sub_callback_dict[p_id])})

        self._locs_plot = self._init_locs_plot()

        script_dir = os.path.dirname(__file__)
        self._results_dir = os.path.join(script_dir, 'Results/')
        if not os.path.isdir(self._results_dir):
            os.makedirs(self._results_dir)

    def _get_time(self):
        t = rospy.Time.now()
        return t.secs + t.nsecs * 1e-9

    def _getLocD1(self, data):
        self._locations['D1'].record(self._get_time()-self._init_time, np.array([data.position[0], data.position[1]]))

    def _getLocD2(self, data):
        self._locations['D2'].record(self._get_time()-self._init_time, np.array([data.position[0], data.position[1]]))

    def _getLocI(self, data):
        self._locations['I'].record(self._get_time()-self._init_time, np.array([data.position[0], data.position[1]]))

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
        xI = np.asarray(self._locations['I'].data)

        if len(xD1)>10 and len(xD2)>10 and len(xI)>10:
            self._locs_plot['axs'].clear()
            self._locs_plot['axs'].plot(xD1[10:,0], xD1[10:,1], 'b')
            self._locs_plot['axs'].plot(xD2[10:,0], xD2[10:,1], 'g')
            self._locs_plot['axs'].plot(xI[10:,0], xI[10:,1], 'r')
            self._locs_plot['fig'].canvas.draw()
            self._locs_plot['axs'].grid()
            if int(len(xI)) % self._save_interval == 0:
                self._locs_plot['fig'].savefig(self._results_dir + 'traj.png')


if __name__ == '__main__':

    rospy.init_node('game_recorder', anonymous=True)
    D1 = rospy.get_param("~D1", 'cf4')
    D2 = rospy.get_param("~D2", 'cf5')
    I = rospy.get_param("~I", 'cf3')
    recorder = GameRecorder(D1=D1, D2=D2, I=I)
    while not rospy.is_shutdown():
        recorder.plot_locs()
