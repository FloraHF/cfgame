import matplotlib.pyplot as plt
import os
import numpy as np
from math import sin, cos, acos, atan2, sqrt, pi
from scipy.interpolate import interp1d

class ReplayPool(object):

	def __init__(self,  cfs=['cf4', 'cf5', 'cf0'],
						res_dir='init_location_1/res1/',
						r=.25):

		self._script_dir = os.path.dirname(__file__)
		self._res_dir = res_dir
		self._cfs = cfs
		self._r = r 
		self._r_close = 1.2*r
		self._k_close = 0.9

		self._t_start, self._t_end, self._t_close, self._fp = self._read_policy()
		self._fx, self._fy = self._read_xy(self._t_start, self._t_end, file='location.csv')
		self._fvx, self._fvy = self._read_xy(self._t_start, self._t_end, file='velocity.csv')
		self._fa = self._read_a(self._t_start, self._t_end)
		self._fh = self._read_heading(self._t_start, self._t_end)

		self._colors = ['b', 'g', 'r']

	########################## read files ###########################
	# read policy
	def _read_policy(self):
		tps, ps, cts = [[], [], []], [[], [], []], [None, None, None]
		for i, (cf, t, p) in enumerate(zip(self._cfs, tps, ps)):
			data_dir = os.path.join(self._script_dir, self._res_dir + cf + '/data/')
			with open(data_dir + 'policy.csv') as f:
			    data = f.readlines()
			    for line in data:
			        datastr = line.split(',')
			        time = float(datastr[0])
			        t.append(time)
			        policy = datastr[1]
			        if 'h_' in policy:
			        	policy_id = 3
			        elif 'i_' in policy:
			        	policy_id = 2
			        elif 'z_' in policy:
			        	policy_id = 1
			        elif ('D1' in policy or 'D2' in policy) and 'close' in policy:
			        	policy_id = 0
			        	if cts[i] is None:
			        		cts[i] = time
			        elif 'both' in policy and 'close' in policy:
			        	policy_id = -1
			        p.append(policy_id)

		tps = [np.asarray(t) for t in tps]
		ps = [np.asarray(p) for p in ps]
		t_start = min([t[0] for t in tps])
		t_end = max([t[-1] for t in tps])
		f_policy = [interp1d(t, p) for t, p in zip(tps, ps)]

		return t_start, t_end, cts, f_policy

	# read location
	def _read_xy(self, t_start, t_end, file='location.csv'):
		f_x, f_y = [], []
		for cf in self._cfs:
			t, x, y = [], [], []
			data_dir = os.path.join(self._script_dir, self._res_dir + cf + '/data/')
			with open(data_dir + file) as f:
			    data = f.readlines()
			    for line in data:
			        datastr = line.split(',')
			        time = float(datastr[0])
			        if time > t_start-0.2 and time < t_end+0.2:
			        	t.append(time)
			        	x.append(float(datastr[1]))
			        	y.append(float(datastr[2]))
			t = np.asarray(t)
			x = np.asarray(x)
			y = np.asarray(y)
			f_x.append(interp1d(t, x))
			f_y.append(interp1d(t, y))

		return f_x, f_y

	# velocity
	def _read_a(self, t_start, t_end):
		f_a = []
		for cf in self._cfs:
			t, a = [], []
			data_dir = os.path.join(self._script_dir, self._res_dir + cf + '/data/')
			with open(data_dir + 'a.csv') as f:
			    data = f.readlines()
			    for line in data:
			        datastr = line.split(',')
			        time = float(datastr[0])
			        if time > t_start-0.1 and time < t_end+0.1:
			        	t.append(time)
			        	a.append(float(datastr[1]))       
			t = np.asarray(t)
			a = np.asarray(a)
			f_a.append(interp1d(t, a))

		return f_a

	def _read_heading(self, t_start, t_end):
		f_h = []
		for cf in self._cfs:
			t, h = [], []
			data_dir = os.path.join(self._script_dir, self._res_dir + cf + '/data/')
			with open(data_dir + 'actual_heading.csv') as f:
			    data = f.readlines()
			    for line in data:
			        datastr = line.split(',')
			        time = float(datastr[0])
			        if time > t_start-0.1 and time < t_end+0.1:
			        	t.append(time)
			        	h.append(float(datastr[1]))       
			t = np.asarray(t)
			h = np.asarray(h)
			f_h.append(interp1d(t, h))

		return f_h

	def _get_location(self, t):
		xd1, yd1 = self._fx[0](t), self._fy[0](t)
		xd2, yd2 = self._fx[1](t), self._fy[1](t)
		xi, yi = self._fx[2](t), self._fy[2](t)
		return xd1, yd1, xd2, yd2, xi, yi

	def _get_velocity(self, t):
		vxd1, vyd1 = self._fvx[0](t), self._fvy[0](t)
		vxd2, vyd2 = self._fvx[1](t), self._fvy[1](t)
		vxi, vyi = self._fvx[2](t), self._fvy[2](t)
		return vxd1, vyd1, vxd2, vyd2, vxi, vyi

	def _get_heading(self, t):
		return self._fh[0](t), self._fh[1](t), self._fh[2](t)

	########################## coords ###########################
	def _get_vecs(self, xd1, yd1, xd2, yd2, xi, yi):
	    D1 = np.array([xd1, yd1, 0])
	    D2 = np.array([xd2, yd2, 0])
	    I = np.array([xi, yi, 0])
	    D1_I = I - D1
	    D2_I = I - D2
	    D1_D2 = D2 - D1
	    return D1_I, D2_I, D1_D2

	def _get_xyz(self, D1_I, D2_I, D1_D2):
	    z = np.linalg.norm(D1_D2/2)
	    x = -np.cross(D1_D2, D1_I)[-1]/(2*z)
	    y = np.dot(D1_D2, D1_I)/(2*z) - z
	    return x, y, z

	def _get_theta(self, D1_I, D2_I, D1_D2):
	    k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I))
	    k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I))  # angle between D1_D2 to D2_I
	    tht = k2 - k1
	    if k1 < 0:
	        tht += 2*pi
	    return tht

	def _get_d(self, D1_I, D2_I, D1_D2):
	    d1 = max(np.linalg.norm(D1_I), r)
	    d2 = max(np.linalg.norm(D2_I), r)
	    return d1, d2

	def _get_alpha(self, D1_I, D2_I, D1_D2):
	    d1, d2 = self._get_d(D1_I, D2_I, D1_D2)
	    a1 = asin(r/d1)
	    a2 = asin(r/d2)
	    return d1, d2, a1, a2

	########################## policies ###########################
	def _get_physical_heading(self, phi_1, phi_2, psi, D1_I, D2_I, D1_D2):
		# print(phi_1*180/pi, phi_2*180/pi, psi*180/pi)
		dphi_1 = atan2(D1_I[1], D1_I[0])
		dphi_2 = atan2(D2_I[1], D2_I[0])
		dpsi = atan2(-D2_I[1], -D2_I[0])
		# print(dphi_1*180/pi, dphi_2*180/pi, dpsi*180/pi)
		phi_1 += dphi_1
		phi_2 += dphi_2
		psi += psi
		return phi_1, phi_2, psi

	def _h_strategy(self, xd1, yd1, xd2, yd2, xi, yi, a=.1/.15):

		D1_I, D2_I, D1_D2 = self._get_vecs(xd1, yd1, xd2, yd2, xi, yi)
		x, y, z = self._get_xyz(D1_I, D2_I, D1_D2)

		xd1_, yd1_ = 0, -z
		xd2_, yd2_ = 0,  z
		xi_, yi_   = x, y

		# print(xd1_, yd1_, xd2_, yd2_, xi_, yi_)
		Delta = sqrt(np.maximum(x**2 - (1 - 1/a**2)*(x**2 + y**2 - (z/a)**2), 0))
		if (x + Delta)/(1 - 1/a**2) - x > 0:
		    xP = (x + Delta)/(1 - 1/a**2)
		else:
		    xP = -(x + Delta)/(1 - 1/a**2)

		P = np.array([xP, 0, 0])
		D1_P = P - np.array([xd1_, yd1_, 0])
		D2_P = P - np.array([xd2_, yd2_, 0])
		I_P = P - np.array([xi_, yi_, 0])
		D1_I_, D2_I_, D1_D2_ = self._get_vecs(xd1_, yd1_, xd2_, yd2_, xi_, yi_)

		phi_1 = atan2(np.cross(D1_I_, D1_P)[-1], np.dot(D1_I_, D1_P))
		phi_2 = atan2(np.cross(D2_I_, D2_P)[-1], np.dot(D2_I_, D2_P))
		psi = atan2(np.cross(-D2_I_, I_P)[-1], np.dot(-D2_I_, I_P))

		# print(phi_1*180/pi, phi_2*180/pi, psi*180/pi)

		return phi_1, phi_2, psi

	def _adjust_strategy(self, phi_1, phi_2, psi, D1_I, D2_I, D1_D2):
		tht = self._get_theta(D1_I, D2_I, D1_D2)
		if np.linalg.norm(D1_I) < self._r_close and np.linalg.norm(D2_I) < self._r_close:  # in both range
			vD1 = np.array([vxd1, vyd1, 0])
			vD2 = np.array([vxd2, vyd2, 0])
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_1 = self._k_close*phi_1
			phi_2 = self._k_close*phi_2
			psi = -tht / 2
		elif np.linalg.norm(D1_I) < self._r_close:
			vD1 = np.array([vxd1, vyd1, 0])
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = self._k_close*phi_1
			if np.linalg.norm(vI) > np.linalg.norm(vD1):
			    psi = - abs(acos(np.linalg.norm(vD1)*cos(phi_1)/np.linalg.norm(vI)))
			else:
			    psi = - abs(phi_1)
			psi = pi - tht + psi
		elif np.linalg.norm(D2_I) < self._r_close:
			vD2 = np.array([vxd2, vyd2, 0])
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			psi = self._k_close * phi_2
			if np.linalg.norm(vI) > np.linalg.norm(vD2):
			    psi = abs(acos(np.linalg.norm(vD2)*cos(phi_2)/np.linalg.norm(vI)))
			else:
			    psi = abs(phi_2)
			psi = psi - pi

		return phi_1, phi_2, psi


	########################## plotting functions ########################
	def _plot_cap_ring(self, ax, xd, yd, r=0.25, color='r'):
	    rs = []
	    for tht in np.linspace(0, 2*pi, 50):
	        x = xd + r*cos(tht)
	        y = yd + r*sin(tht)
	        rs.append((np.array([x, y])))
	    rs = np.asarray(rs)
	    ax.plot(rs[:,0], rs[:,1], color)

	def _plot_velocity_arrow(self, ax, n=10):

		for k in np.linspace(0.1, .9, 10):
			t = k*(self._t_end - self._t_start) + self._t_start

			xd1, yd1, xd2, yd2, xi, yi = self._get_location(t)
			D1_I, D2_I, D1_D2 = self._get_vecs(xd1, yd1, xd2, yd2, xi, yi)
			vxd1, vyd1, vxd2, vyd2, vxi, vyi = self._get_velocity(t)
			# headings = self._get_heading(t)

			for i in range(3):
				heading = self._h_strategy(xd1, yd1, xd2, yd2, xi, yi, a=self._fa[i](t))[i]
				# heading = headings[i]
				vx, vy = cos(heading)/20, sin(heading)/20
				# vx, vy = self._fvx[i](t), self._fvy[i](t)
				# lenv = 30*sqrt(vx**2 + vy**2)
				# vx, vy = vx/lenv, vy/lenv
				ax.arrow(self._fx[i](t), self._fy[i](t), vx, vy, 
						fc='k', head_width=.01, zorder=10)

	def plot_traj(self):

		fig, ax = plt.subplots()
		ts = np.linspace(self._t_start, self._t_end, 50)
		for i in range(3):
			ax.plot(self._fx[i](ts), self._fy[i](ts), self._colors[i])
			ax.plot(self._fx[i](self._t_end), self._fy[i](self._t_end), 'o'+self._colors[i])
			if self._t_close[i] is not None:
				ax.plot(self._fx[i](self._t_close[i]), self._fy[i](self._t_close[i]), 'x'+self._colors[i])
			if i != 2:
				self._plot_cap_ring(ax, self._fx[i](self._t_end), self._fy[i](self._t_end), color=self._colors[i]+'-')
				if self._t_close[i] is not None:
					self._plot_cap_ring(ax, self._fx[i](self._t_close[i]), self._fy[i](self._t_close[i]), color=self._colors[i]+'--')
		self._plot_velocity_arrow(ax)
		ax.axis('equal')
		ax.grid()
		plt.show()


########################## main function ##########################
if __name__ == '__main__':

	replayer = ReplayPool()
	replayer.plot_traj()
