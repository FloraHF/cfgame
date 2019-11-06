import matplotlib.pyplot as plt
import os
import numpy as np
from math import sin, cos, acos, atan2, sqrt, pi
from scipy.interpolate import interp1d

script_dir = os.path.dirname(__file__)
res_dir = 'res1/'
cfs = ['cf4', 'cf5', 'cf3']
r = 0.25


########################## read files ###########################
# read policy
def read_policy():
	tps, ps, cts = [[], [], []], [[], [], []], [None, None, None]
	for i, (cf, t, p) in enumerate(zip(cfs, tps, ps)):
		data_dir = os.path.join(script_dir, res_dir + cf + '/data/')
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
		        elif 'D1' in policy and 'close' in policy:
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
def read_xy(t_start, t_end, file='location.csv'):
	f_x, f_y = [], []
	for cf in cfs:
		t, x, y = [], [], []
		data_dir = os.path.join(script_dir, res_dir + cf + '/data/')
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
def read_a(t_start, t_end):
	f_a = []
	for cf in cfs:
		t, a = [], []
		data_dir = os.path.join(script_dir, res_dir + cf + '/data/')
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

########################## coords ###########################
def get_vecs(xd1, yd1, xd2, yd2, xi, yi):
    D1 = np.array([xd1, yd1, 0])
    D2 = np.array([xd2, yd2, 0])
    I = np.array([xi, yi, 0])
    D1_I = I - D1
    D2_I = I - D2
    D1_D2 = D2 - D1
    return D1_I, D2_I, D1_D2

def get_xyz(D1_I, D2_I, D1_D2):
    z = np.linalg.norm(D1_D2/2)
    x = -np.cross(D1_D2, D1_I)[-1]/(2*z)
    y = np.dot(D1_D2, D1_I)/(2*z) - z
    return x, y, z

def get_theta(D1_I, D2_I, D1_D2):
    k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I))
    k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I))  # angle between D1_D2 to D2_I
    tht = k2 - k1
    if k1 < 0:
        tht += 2*pi
    return tht

def get_d(D1_I, D2_I, D1_D2):
    d1 = max(np.linalg.norm(D1_I), r)
    d2 = max(np.linalg.norm(D2_I), r)
    return d1, d2

def get_alpha(D1_I, D2_I, D1_D2):
    d1, d2 = get_d(D1_I, D2_I, D1_D2)
    a1 = asin(r/d1)
    a2 = asin(r/d2)
    return d1, d2, a1, a2

########################## policies ###########################
def get_physical_heading(phi_1, phi_2, psi, D1_I, D2_I, D1_D2):
	# print(phi_1*180/pi, phi_2*180/pi, psi*180/pi)
	dphi_1 = atan2(D1_I[1], D1_I[0])
	dphi_2 = atan2(D2_I[1], D2_I[0])
	dpsi = atan2(-D2_I[1], -D2_I[0])
	# print(dphi_1*180/pi, dphi_2*180/pi, dpsi*180/pi)
	phi_1 += dphi_1
	phi_2 += dphi_2
	psi += psi
	return phi_1, phi_2, psi

def h_strategy(xd1, yd1, xd2, yd2, xi, yi, a=.1/.15):

	D1_I, D2_I, D1_D2 = get_vecs(xd1, yd1, xd2, yd2, xi, yi)
	x, y, z = get_xyz(D1_I, D2_I, D1_D2)

	xd1_, yd1_ = 0, -z
	xd2_, yd2_ = 0,  z
	xi_, yi_   = x, y

	# print(xd1_, yd1_, xd2_, yd2_, xi_, yi_)
	Delta = sqrt(np.maximum(x**2 - (1 - 1/a**2)*(x**2 + y**2 - (z/a)**2), 0))
	if (x + Delta) / (1 - 1/a**2) - x > 0:
	    xP = (x + Delta) / (1 - 1/a**2)
	else:
	    xP = -(x + Delta) / (1 - 1/a**2)

	P = np.array([xP, 0, 0])
	D1_P = P - np.array([xd1_, yd1_, 0])
	D2_P = P - np.array([xd2_, yd2_, 0])
	I_P = P - np.array([xi_, yi_, 0])
	D1_I_, D2_I_, D1_D2_ = get_vecs(xd1_, yd1_, xd2_, yd2_, xi_, yi_)

	phi_1 = atan2(np.cross(D1_I_, D1_P)[-1], np.dot(D1_I_, D1_P))
	phi_2 = atan2(np.cross(D2_I_, D2_P)[-1], np.dot(D2_I_, D2_P))
	psi = atan2(np.cross(-D2_I_, I_P)[-1], np.dot(-D2_I_, I_P))

	# print(phi_1*180/pi, phi_2*180/pi, psi*180/pi)

	return get_physical_heading(phi_1, phi_2, psi, D1_I, D2_I, D1_D2)




########################## plotting functions ########################
def plot_cap_ring(ax, xd, yd, r=0.25, color='r'):
    rs = []
    for tht in np.linspace(0, 2*pi, 50):
        x = xd + r*cos(tht)
        y = yd + r*sin(tht)
        rs.append((np.array([x, y])))
    rs = np.asarray(rs)
    ax.plot(rs[:,0], rs[:,1], color)

########################## main function ##########################
t_start, t_end, t_close, fp = read_policy()
colors = ['b', 'g', 'r']
fx, fy = read_xy(t_start, t_end, file='location.csv')
fvx, fvy = read_xy(t_start, t_end, file='velocity.csv')
fa = read_a(t_start, t_end)

fig, ax = plt.subplots()

# trajectory
ts = np.linspace(t_start, t_end, 50)
for i in range(3):
	ax.plot(fx[i](ts), fy[i](ts), colors[i])
	ax.plot(fx[i](t_close[i]), fy[i](t_close[i]), 'x'+colors[i])
	ax.plot(fx[i](t_end), fy[i](t_end), 'o'+colors[i])
	if i != 2:
		plot_cap_ring(ax, fx[i](t_end), fy[i](t_end), color=colors[i]+'-')
		plot_cap_ring(ax, fx[i](t_close[i]), fy[i](t_close[i]), color=colors[i]+'--')

# velocity vector
# for t in [17.8]:
for k in np.linspace(0.1, .9, 10):
	t = k*(t_end - t_start) + t_start
	xd1, yd1 = fx[0](t), fy[0](t)
	xd2, yd2 = fx[1](t), fy[1](t)
	xi, yi = fx[2](t), fy[2](t)
	D1_I, D2_I, D1_D2 = get_vecs(xd1, yd1, xd2, yd2, xi, yi)

	vxd1, vyd1 = fvx[0](t), fvy[0](t)
	vxi, vyi = fvx[2](t), fvy[2](t)
	Vd1 = np.array([vxd1, vyd1, 0])
	phi_1 = atan2(np.cross(D1_I, Vd1)[-1], np.dot(D1_I, Vd1))
	Vd1_mag = np.linalg.norm(Vd1)
	if .15 > Vd1_mag:
		psi = - abs(acos(Vd1_mag * cos(phi_1) / .15))
	else:
		psi = - abs(phi_1)
	print(Vd1_mag*cos(phi_1) - .15*cos(psi), phi_1*180/pi, psi*180/pi)
	psi = pi - get_theta(D1_I, D2_I, D1_D2) + psi	

	for i in range(3):
		heading = h_strategy(xd1, yd1, xd2, yd2, xi, yi, a=fa[i](t))[i]
		vx, vy = cos(heading)/20, sin(heading)/20
		# vx, vy = fvx[i](t), fvy[i](t)
		# lenv = 30*sqrt(vx**2 + vy**2)
		# vx, vy = vx/lenv, vy/lenv
		plt.arrow(fx[i](t), fy[i](t), vx, vy, 
				fc='k', head_width=.01, zorder=10)

ax.axis('equal')
ax.grid()
plt.show()