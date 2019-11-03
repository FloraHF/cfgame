import os
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

script_dir = os.path.dirname(__file__)


def plot_data(cf_id, files=['location.csv', 'goal.csv'], labels=['x(m)', 'y(m)', 'z(m)'], res_dir='res1/Results/'):
    data_dir = os.path.join(script_dir, res_dir + cf_id + '/data/')
    fig_dir = os.path.join(script_dir, res_dir + cf_id + '/fig/')
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig = plt.figure(tight_layout=True)
    n = 3
    if files[0] == 'euler.csv':
        n = 4
    gs = gridspec.GridSpec(n, 1)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(n)]

    linestyles = ['b-', 'b--']
    for file, linestyle in zip(files, linestyles):
        with open(data_dir + file) as f:
            data = f.readlines()
            t, xs = [], [[], [], []]
            for line in data:
                datastr = line.split(',')
                t.append(float(datastr[0]))
                xs[0].append(float(datastr[1]))
                xs[1].append(float(datastr[2]))
                xs[2].append(float(datastr[3]))
        if files[0] == 'euler.csv':
            if file == 'euler.csv':
                for x, ax in zip(xs, axs[:3]):
                    ax.plot(t, x, linestyle)
            elif file == 'cmd_vel.csv':
                for x, ax in zip(xs, [axs[0], axs[1], axs[3]]):
                    ax.plot(t, x, linestyle)
        else:
            for x, ax in zip(xs, axs):
                ax.plot(t, x, linestyle)

    for ax, label in zip(axs, labels):
        ax.set_xlabel('t(s)')
        ax.set_ylabel(label)
        ax.grid()

    fig.savefig(fig_dir + files[0].split('.')[0] + '.png')


def plot_traj(cfs=['cf4', 'cf5', 'cf3'], res_dir='res1/Results/'):

    def cap_ring(xd, r=0.25):
        rs = []
        for tht in np.linspace(0, 2*pi, 50):
            x = xd[0] + r*cos(tht)
            y = xd[1] + r*sin(tht)
            rs.append((np.array([x, y])))
        return np.asarray(rs)

    data_dirs = [os.path.join(script_dir, res_dir + cf + '/data/') for cf in cfs]
    fig_dir = os.path.join(script_dir, res_dir)
    xs = [[], [], []]
    ys = [[], [], []]
    policies = [[], [], []]
    fig, ax = plt.subplots()
    colors = ['g', 'b', 'r']
    for data_dir, x, y in zip(data_dirs, xs, ys):
        with open(data_dir + 'location.csv') as f:
            data = f.readlines()
            for line in data:
                datastr = line.split(',')
                x.append(float(datastr[1]))
                y.append(float(datastr[2]))
        # with open(data_dir + 'policy.csv') as f:
        #     data = f.readlines()
        #     for line in data:
        #         p.append(line.split(',')[0])

    i_start, i_end = 1000, len(xs[0])-500
    for x, y, p, color in zip(xs, ys, policies, colors):
            ax.plot(x[i_start: i_end], y[i_start: i_end], color)

    for i in range(i_start, i_end):
        if i%80 == 0:
            ax.plot([xs[0][i], xs[2][i]], [ys[0][i], ys[2][i]], 'b--')
            ax.plot([xs[1][i], xs[2][i]], [ys[1][i], ys[2][i]], 'b--')
            # ring_1 = cap_ring([xs[0][i], ys[0][i]])
            # ring_2 = cap_ring([xs[1][i], ys[1][i]])
            # ax.plot(ring_1[:,0], ring_1[:,1], 'g')
            # ax.plot(ring_2[:,0], ring_2[:,1], 'b')
        if i == i_end-1:
            ring_1 = cap_ring([xs[0][i], ys[0][i]])
            ring_2 = cap_ring([xs[1][i], ys[1][i]])
            ax.plot(ring_1[:,0], ring_1[:,1], 'g')
            ax.plot(ring_2[:,0], ring_2[:,1], 'b')

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.axis('equal')
    ax.grid()

    fig.savefig(fig_dir + 'traj.png')

cfs = ['cf4', 'cf5', 'cf3']
res = 'res3/Results/'
for cf in cfs:
    plot_data(cf, files=['location.csv', 'goal.csv'], 
                labels=['x(m)', 'y(m)', 'z(m)'], 
                res_dir=res)

    plot_data(cf, files=['velocity.csv', 'cmdVtemp.csv'], 
                labels=['vz(m/s)', 'vy(m/s)', 'vz(m/s)'], 
                res_dir=res)

    plot_data(cf, files=['euler.csv', 'cmd_vel.csv'], 
                labels=['roll(/s)', 'pitch(/s)', 'yaw(/s)', 'thrust(PWM)'], 
                res_dir=res)

plot_traj(cfs=cfs, res_dir=res)
