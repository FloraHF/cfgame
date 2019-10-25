import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


script_dir = os.path.dirname(__file__)

def plot_data(cf_id, files=['location.csv', 'goal.csv'], labels=['x(m)', 'y(m)', 'z(m)']):

	data_dir = os.path.join(script_dir, 'Results'+cf_id+'/data/')
	fig_dir = os.path.join(script_dir, 'Results'+cf_id+'/fig/')

	fig = plt.figure(tight_layout=True)
	n = 3
	if files[0] == 'euler.csv':
		n = 4
	gs = gridspec.GridSpec(n, 1)
	axs = [fig.add_subplot(gs[i, 0]) for i in range(n)]

	linestyles = ['-', '--']
	for file, linestyle in zip(files, linestyles):
		with open(data_dir+file) as f:
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
				for x, ax in zip(xs, axs[[0,1,3]]):
					ax.plot(t, x, linestyle)
		else:
			for x, ax in zip(xs, axs):
				ax.plot(t, x, linestyle)
	
	for ax, label in zip(axs, labels):
	    ax.set_xlabel('t(s)')
	    ax.set_ylabel(label)
	    ax.grid()

	fig.savefig(fig_dir + files[0].split('.')[0] + '.png')

def plot_traj(cfs=['cf4', 'cf5', 'cf3']):

	data_dirs = [os.path.join(script_dir, 'Results/'+cf+'/data/') for cf in cfs]
	fig_dir = os.path.join(script_dir, 'Results/')

	fig, ax = plt.subplots()
	colors = ['g', 'b', 'r']
	for data_dir, color in zip(data_dirs, colors):
		with open(data_dir+'location.csv') as f:
			data = f.readlines()
			t, xs = [], [[], [], []]
			for line in data:
				datastr = line.split(',')
				x.append(float(datastr[1]))
				y.append(float(datastr[2]))
		ax.plot(x, y, color)
	ax.set_xlabel('x(m)')
	ax.set_ylabel('y(m)')
	ax.axis('equal')
	ax.grid()

	fig.savefig(fig_dir + 'traj.png')


# plot_data(files=['location.csv', 'goal.csv'], labels=['x(m)', 'y(m)', 'z(m)'])
# plot_data(files=['velocity.csv', 'cmdV.csv'], labels=['vz(m/s)', 'vy(m/s)', 'vz(m/s)'])
# plot_data(files=['euler.csv', 'cmd_vel.csv'], labels=['roll(/s)', 'pitch(/s)', 'yaw(/s)', 'thrust(PWM)'])
# plot_traj(cfs=['cf4', 'cf5', 'cf3'])