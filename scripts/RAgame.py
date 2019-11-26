#!/usr/bin/env python3

import os
from copy import deepcopy
import rospy
import numpy as np
from math import sin, cos, sqrt, atan2, asin, acos, pi

# import tensorflow as tf
# from tensorflow.keras.models import load_model
from keras.models import load_model

from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseStamped, Twist

from optitrack_broadcast.msg import Mocap
from strategy_fastD import deep_target_strategy

class RAgame(object):

	def __init__(self, zDs, zIs, rate=10, param_file='',
				 Ds='', Is='',
				 logger_dir='res1'):

		self.rate = rospy.Rate(rate)
		self.states = dict()

		self.param_id = param_file.split('.')[0].split('_')[-1]
		self.vdes = dict()
		self.zs = dict()
		self.x0s = dict()
		self.xes = dict()
		self.goal_msgs = dict()
		self.policy_fns = dict()
		self.nv = 20

		self.cap_time = .2
		self.last_cap = False
		self.end = False
		self.time_inrange = 0.
		self.time_end = 0.		

		self.vs = dict()
		self.xs = dict()
		self.vnorms = dict()

		self.mocap_sub_callbacks = {'D0': self.getD0, 'D1': self.getD1, 'I0': self.getI0}
		self.mocap_subs = dict()

		self.players = dict()
		self.goal_pubs = dict()
		self.cmdV_pubs = dict()
		self.policy_pubs = dict()
		self.a_pub = rospy.Publisher('/a', Float32, queue_size=1)
		
		self.takeoff_clients = dict()
		self.auto_clients = dict()
		self.land_clients = dict()
		self.play_clients = dict()

		script_dir = os.path.dirname(__file__)
		with open(os.path.join(script_dir+'params/', param_file), 'r') as f:
			lines = f.readlines()
			for line in lines:
				if 'x' in line:
					data = line.split(',')
					role = data[0][1:]
					self.x0s[role] = np.array([float(data[1]), float(data[2])])
					self.xes[role] = np.array([float(data[1]), float(data[2])])
					self.xs[role] = np.array([float(data[1]), float(data[2])])
				if 'vd' in line:
					vd = float(line.split(',')[-1])
				if 'vi' in line:
					vi = float(line.split(',')[-1])
				if 'rc' in line:
					self.r = float(line.split(',')[-1])
				if 'r_close' in line:
					self.r_close = float(line.split(',')[-1])*self.r
				if 'k_close' in line:
					self.k_close = float(line.split(',')[-1])
				if 'S' in line:
					self.S = float(line.split(',')[-1])
				if 'T' in line:
					self.T = float(line.split(',')[-1])
				if 'gmm' in line:
					self.gmm = float(line.split(',')[-1])		
				if 'D' in line:
					self.D = float(line.split(',')[-1])
				if 'delta' in line:
					self.delta = float(line.split(',')[-1])
		self.a = vd/vi

		for i, (D, z) in enumerate(zip(Ds, zDs)):
			if D != '':
				role = 'D'+str(i)
				self.players[role] = D
				self.zs[role] = float(z)
				self.goal_msgs[role] = PoseStamped()
				# print(D, role)
				self.updateGoal(role, goal=self.x0s[role], init=True)

				self.vnorms[role] = []
				self.vs[role] = np.zeros(2)
				self.vdes[role] = vd
				self.policy_fns[role] = load_model(os.path.join(script_dir, 'Policies/PolicyFn_'+role+'.h5'))
				# self.policy_fns[role] = load_model('PolicyFn_' + role)
				self.states[role] = None

				self.goal_pubs[role] = rospy.Publisher('/'+D+'/goal', PoseStamped, queue_size=1)
				self.cmdV_pubs[role] = rospy.Publisher('/'+D+'/cmdV', Twist, queue_size=1)
				self.policy_pubs[role] = rospy.Publisher('/'+D+'/policy', String, queue_size=1)
				# self.a_pubs[role]	= rospy.Publisher('/'+D+'/a', PoseStamped, queue_size=1)
				
				self.takeoff_clients[role]= self.service_client(D, '/cftakeoff')
				self.land_clients[role]   = self.service_client(D, '/cfland')
				self.play_clients[role]   = self.service_client(D, '/cfplay')
				self.auto_clients[role]   = self.service_client(D, '/cfauto')
				self.mocap_subs[role] = rospy.Subscriber('/'+D+'/mocap', Mocap, self.mocap_sub_callbacks[role])

		for i, (I, z) in enumerate(zip(Is, zIs)):
			if I != '':
				role = 'I'+str(i)
				self.players[role] = I
				self.zs[role] = float(z)
				self.goal_msgs[role] = PoseStamped()
				self.updateGoal(role, goal=self.x0s[role], init=True)

				self.vnorms[role] = []
				self.vs[role] = np.zeros(2)
				self.vdes[role] = vi
				self.policy_fns[role] = load_model(os.path.join(script_dir, 'Policies/PolicyFn_'+role+'.h5'))
				# self.policy_fns[role] = load_model('PolicyFn_' + role)
				self.states[role] = None

				self.goal_pubs[role] = rospy.Publisher('/'+I+'/goal', PoseStamped, queue_size=1)
				self.cmdV_pubs[role] = rospy.Publisher('/'+I+'/cmdV', Twist, queue_size=1)
				self.policy_pubs[role] = rospy.Publisher('/'+I+'/policy', String, queue_size=1)
				
				self.takeoff_clients[role]= self.service_client(I, '/cftakeoff')
				self.land_clients[role]   = self.service_client(I, '/cfland')
				self.play_clients[role]   = self.service_client(I, '/cfplay')
				self.auto_clients[role] = self.service_client(I, '/cfauto')
				self.mocap_subs[role] = rospy.Subscriber('/'+I+'/mocap', Mocap, self.mocap_sub_callbacks[role])

		# for role in self.players:
		# 	print(self.goal_msgs[role].pose.position.x, self.goal_msgs[role].pose.position.y)
		rospy.Service('alltakeoff', Empty, self.alltakeoff)
		rospy.Service('allplay', Empty, self.allplay)
		rospy.Service('allland', Empty, self.allland)

		with open(script_dir+'/'+logger_dir+'/info.csv', 'a') as f:
			f.write('paramid,'+self.param_id)
			for role, cf in self.players.items():
				f.write(role + ',' + cf + '\n')
				f.write('x'+role+',%.5f, %.5f\n'%(self.x0s[role][0], self.x0s[role][1]))
			f.write('vd,%.2f'%vd + '\n')
			f.write('vi,%.2f'%vi + '\n')
			f.write('rc,%.2f'%self.r + '\n')
			if vd < vi:
				f.write('r_close,%.2f'%self.r_close + '\n')
				f.write('k_close,%.2f'%self.k_close + '\n')	
				f.write('S,%.10f\n'%self.S)
				f.write('T,%.10f\n'%self.T)
				f.write('gmm,%.10f\n'%self.gmm)
				f.write('D,%.10f\n'%self.D)
				f.write('delta,%.10f\n'%self.delta)		

	def line_target(self, x):
		return x[1]

	def service_client(self, cf, name):
		srv_name = '/' + cf + name
		rospy.wait_for_service(srv_name)
		rospy.loginfo('found' + srv_name + 'service')
		return rospy.ServiceProxy(srv_name, Empty)

	def alltakeoff(self, req):
		for role, takeoff in self.takeoff_clients.items():
			self.states[role] = 'hover'
			takeoff()
		return EmptyResponse()

	def allplay(self, req):
		for role, play in self.play_clients.items():
			self.states[role] = 'play'
			play()
		return EmptyResponse()

	def allland(self, req):
		for role, land in self.land_clients.items():
			self.states[role] = 'land'
			land()
		return EmptyResponse()

	def get_time(self):
		t = rospy.Time.now()
		return t.secs + t.nsecs * 1e-9

	def getD0(self, data):
		self.xs['D0'] = np.array([data.position[0], data.position[1]])
		self.vs['D0'] = np.array([data.velocity[0], data.velocity[1]])
		if len(self.vnorms['D0']) > self.nv:
			self.vnorms['D0'].pop(0)
		self.vnorms['D0'].append(sqrt(data.velocity[0]**2 + data.velocity[1]**2))		

	def getD1(self, data):
		self.xs['D1'] = np.array([data.position[0], data.position[1]])
		self.vs['D1'] = np.array([data.velocity[0], data.velocity[1]])
		if len(self.vnorms['D0']) > self.nv:
			self.vnorms['D0'].pop(0)
		self.vnorms['D0'].append(sqrt(data.velocity[0]**2 + data.velocity[1]**2))		

	def getI0(self, data):
		self.xs['I0'] = np.array([data.position[0], data.position[1]])
		self.vs['I0'] = np.array([data.velocity[0], data.velocity[1]])
		if len(self.vnorms['D0']) > self.nv:
			self.vnorms['D0'].pop(0)
		self.vnorms['D0'].append(sqrt(data.velocity[0]**2 + data.velocity[1]**2))		  

	def is_capture(self):
		d0 = np.linalg.norm(self.xs['D0'] - self.xs['I0'])
		d1 = np.linalg.norm(self.xs['D1'] - self.xs['I0'])
		return d0 < self.r or d1 < self.r
	
	def updateGoal(self, role, goal=None, init=False):

		if init:
			self.goal_msgs[role].header.seq = 0
			self.goal_msgs[role].header.frame_id = '/world'
		else:
			self.goal_msgs[role].header.seq += 1

		self.goal_msgs[role].header.stamp = rospy.Time.now()

		if goal is not None:
			self.goal_msgs[role].pose.position.x = goal[0]
			self.goal_msgs[role].pose.position.y = goal[1]
			self.goal_msgs[role].pose.position.z = self.zs[role]
		else:
			self.goal_msgs[role].pose.position.x = self.x0s[role][0]
			self.goal_msgs[role].pose.position.y = self.x0s[role][1]
			self.goal_msgs[role].pose.position.z = self.zs[role]

		# quaternion = transform.transformations.quaternion_from_euler(0, 0, 0)
		self.goal_msgs[role].pose.orientation.x = 0
		self.goal_msgs[role].pose.orientation.y = 0
		self.goal_msgs[role].pose.orientation.z = 0
		self.goal_msgs[role].pose.orientation.w = 1

	def get_a(self):
		if len(self.vnorms['D0'])>2 and len(self.vnorms['D1'])>2 and len(self.vnorms['I0'])>2:
			vd0 = np.array(self.vnorms['D0']).mean()
			vd1 = np.array(self.vnorms['D1']).mean()
			vi0 = np.array(self.vnorms['I0']).mean()
			# print((vd1 + vd2)/(2 * vi))
			a = min((vd0 + vd1)/(2*vi0), 0.99)
		else:
			a = self.a
		self.a_pub.publish(a)
		return a

	def get_vecs(self):
		D1 = np.concatenate((self.xs['D0'], [0]))
		D2 = np.concatenate((self.xs['D1'], [0]))
		I = np.concatenate((self.xs['I0'], [0]))
		D1_I = I - D1
		D2_I = I - D2
		D1_D2 = D2 - D1
		return D1_I, D2_I, D1_D2

	def get_base(self, D1_I, D2_I, D1_D2):
		base_d1 = atan2(D1_I[1], D1_I[0])
		base_d2 = atan2(D2_I[1], D2_I[0])
		base_i = atan2(-D2_I[1], -D2_I[0])
		# print(base_d1*180/pi, base_d2*180/pi, base_i*180/pi)
		return {'D0': base_d1, 'D1': base_d2, 'I0': base_i}

	def get_xyz(self, D1_I, D2_I, D1_D2):
		z = np.linalg.norm(D1_D2)/2
		x = -np.cross(D1_D2, D1_I)[-1]/(2*z)
		y = np.dot(D1_D2, D1_I)/(2*z) - z
		return x, y, z

	def get_theta(self, D1_I, D2_I, D1_D2):
		k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I))  # angle between D1_D2 to D1_I
		k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I))  # angle between D1_D2 to D2_I
		tht = k2 - k1
		if k1 < 0:
			tht += 2 * pi
		return tht

	def get_d(self, D1_I, D2_I, D1_D2):
		d1 = max(np.linalg.norm(D1_I), self.r)
		d2 = max(np.linalg.norm(D2_I), self.r)
		return d1, d2

	def get_alpha(self, D1_I, D2_I, D1_D2):
		d1, d2 = self.get_d(D1_I, D2_I, D1_D2)
		a1 = asin(self.r/d1)
		a2 = asin(self.r/d2)
		return d1, d2, a1, a2

	def nn_strategy(self):
		acts = dict()
		x = np.concatenate((self.xs['D0'], self.xs['I0'], self.xs['D1']))
		for role, p in self.policy_fns.items():
			acts[role] = p.predict(x[None])[0]
		acts['policy'] = 'nn_strategy'
		return acts

	def f_strategy(self):
		psis, phis = deep_target_strategy(self.xs['I0'], (self.xs['D0'], self.xs['D1']), self.line_target, self.a, self.r)
		# print(psis, phis)
		acts = dict()
		for role, p in self.players.items():
			if 'D' in role:
				acts[role] = phis[int(role[-1])]
			elif 'I' in role:
				acts[role] = psis[int(role[-1])]
		acts['policy'] = 'f_strategy'
		return acts

	def z_strategy(self):
		D1_I, D2_I, D1_D2 = self.get_vecs()
		base = self.get_base(D1_I, D2_I, D1_D2)
		d1, d2 = self.get_d(D1_I, D2_I, D1_D2)
		tht = self.get_theta(D1_I, D2_I, D1_D2)
		phi_1 = -pi / 2
		phi_2 = pi / 2
		cpsi = d2 * sin(tht)
		spsi = -(d1 - d2 * cos(tht))
		psi = atan2(spsi, cpsi)
		# print('%.2f, %.2f, %.2f'%(phi_1*180/pi, phi_2*180/pi, psi*180/pi))

		phi_1 += base['D0']
		phi_2 += base['D1']
		psi += base['I0']

		# print('%.2f, %.2f, %.2f'%(phi_1*180/pi, phi_2*180/pi, psi*180/pi))

		return {'D0': phi_1, 'D1': phi_2, 'I0': psi, 'policy': 'z_strategy'}

	def h_strategy(self):
		D1_I, D2_I, D1_D2 = self.get_vecs()
		base = self.get_base(D1_I, D2_I, D1_D2)
		x, y, z = self.get_xyz(D1_I, D2_I, D1_D2)
		a = self.get_a()

		Delta = sqrt(np.maximum(x ** 2 - (1 - 1/self.a**2)*(x**2 + y**2 - (z/self.a)**2), 0))
		if (x + Delta) / (1 - 1/self.a ** 2) - x > 0:
			xP = (x + Delta)/(1 - 1/self.a**2)
		else:
			xP = -(x + Delta)/(1 - 1/self.a**2)

		P = np.array([xP, 0, 0])
		D0_ = np.array([0, -z, 0])
		D1_ = np.array([0, z, 0])
		I0_ = np.array([x, y, 0])
		D0_P = P - D0_
		D1_P = P - D1_
		I0_P = P - I0_
		D0_I0_ = I0_ - D0_
		D1_I0_ = I0_ - D1_
		D0_D1_ = D1_ - D0_

		phi_1 = atan2(np.cross(D0_I0_, D0_P)[-1], np.dot(D0_I0_, D0_P))
		phi_2 = atan2(np.cross(D1_I0_, D1_P)[-1], np.dot(D1_I0_, D1_P))
		psi = atan2(np.cross(-D1_I0_, I0_P)[-1], np.dot(-D1_I0_, I0_P))
		# print(phi_1, phi_2, psi)
		phi_1 += base['D0']
		phi_2 += base['D1']
		psi += base['I0']

		return {'D0': phi_1, 'D1': phi_2, 'I0': psi, 'policy': 'h_strategy'}

	def i_strategy(self):
		
		D1_I, D2_I, D1_D2 = self.get_vecs()
		base = self.get_base(D1_I, D2_I, D1_D2)
		d1, d2, a1, a2 = self.get_alpha(D1_I, D2_I, D1_D2)
		tht = self.get_theta(D1_I, D2_I, D1_D2)
		
		LB = acos(self.a)
		
		phi_2 = pi/2 - a2 + 0.01
		psi = -(pi/2 - LB + a2)
		d = d2*(sin(phi_2)/sin(LB))
		l1 = sqrt(d1**2 + d**2 - 2*d1*d*cos(tht + psi))
		cA = (d**2 + l1**2 - d1**2)/(2*d*l1)
		sA = sin(tht + psi)*(d1/l1)
		A = atan2(sA, cA)
		phi_1 = -(pi - (tht + psi) - A) + base['D0']
		phi_2 = pi/2 - a2 + base['D1']
		psi = -(pi/2 - LB + a2) + base['I0']
		return {'D0': phi_1, 'D1': phi_2, 'I0': psi,'policy': 'i_strategy'}

	def m_strategy(self):
		D1_I, D2_I, D1_D2 = self.get_vecs()
		x, y, z = self.get_xyz(D1_I, D2_I, D1_D2)
		if x < 0.1:
			act = self.h_strategy()
		else:
			act = self.i_strategy()
		return act

	def c_strategy(self, dstrategy=m_strategy, istrategy=m_strategy):
		D0_I0, D1_I0, D0_D1 = self.get_vecs()
		base = self.get_base(D0_I0, D1_I0, D0_D1)
		
		#=============== both defenders are close ===============#
		if np.linalg.norm(D0_I0) < self.r_close and np.linalg.norm(D1_I0) < self.r_close:  # in both range
			for role in self.players:
				self.policy_pubs[role].publish('both close')
			vD1 = np.concatenate((self.vs['D0'], [0]))
			phi_1 = atan2(np.cross(D0_I0, vD1)[-1], np.dot(D0_I0, vD1))
			phi_1 = self.k_close*phi_1 + base['D0']

			vD2 = np.concatenate((self.vs['D1'], [0]))
			phi_2 = atan2(np.cross(D1_I0, vD2)[-1], np.dot(D1_I0, vD2))
			phi_2 = self.k_close*phi_2 + base['D1']		 
			
			psi = -self.get_theta(D0_I0, D1_I0, D0_D1)/2 + base['I0']

			return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

		#=============== only D1 is close ===============#
		elif np.linalg.norm(D0_I0) < self.r_close:  # in D1's range
			# print(self.vs['D0'])
			vD1 = np.concatenate((self.vs['D0'], [0]))
			phi_1 = atan2(np.cross(D0_I0, vD1)[-1], np.dot(D0_I0, vD1))
			phi_1 = self.k_close*phi_1 + base['D0']
			self.policy_pubs['D0'].publish('D0 close')

			act = dstrategy(self)
			phi_2 = act['D1']
			self.policy_pubs['D1'].publish(act['policy'])

			if self.vdes['I0'] > self.vnorms['D0'][-1]:
				psi = - abs(acos(self.vnorms['D0'][-1] * cos(phi_1) / self.vdes['I0']))
			else:
				psi = - abs(phi_1)
			psi = pi - self.get_theta(D0_I0, D1_I0, D0_D1) + psi + base['I0']
			self.policy_pubs['I0'].publish('D0 close')

			return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

		#=============== only D2 is close ===============#
		elif np.linalg.norm(D1_I0) < self.r_close:
			vD2 = np.concatenate((self.vs['D1'], [0]))
			phi_2 = atan2(np.cross(D1_I0, vD2)[-1], np.dot(D1_I0, vD2))
			act = dstrategy(self)
			phi_1 = act['D0']
			self.policy_pubs['D0'].publish(act['policy'])

			phi_2 = self.k_close * phi_2 + base['D1']
			self.policy_pubs['D1'].publish('D1 close')

			if self.vdes['I0'] > self.vnorms['D1'][-1]:
				psi = abs(acos(self.vnorms['D1'][-1] * cos(phi_2) / self.vdes['I0']))
			else:
				psi = abs(phi_2)
			psi = psi - pi + base['I0']
			self.policy_pubs['D1'].publish('D1 close')

			return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

		#============== no defender is close =============#
		else:
			dact = dstrategy(self)
			iact = istrategy(self)
			for role in self.players:
				if 'D' in role:
					self.policy_pubs[role].publish(dact['policy'])
				if 'I' in role:
					self.policy_pubs[role].publish(iact['policy'])
			return {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}

	def hover(self, role, x):
		self.updateGoal(role, x)
		self.goal_pubs[role].publish(self.goal_msgs[role])

	def game(self, dt, dstrategy=m_strategy, istrategy=nn_strategy, close_adjust=False):

		if not self.end:
			# print('Playing')
			if close_adjust:
				actions = self.c_strategy(dstrategy=dstrategy, istrategy=istrategy)
			else:
				dact = dstrategy(self)
				iact = istrategy(self)
				actions = {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}
				for role in self.players:
					if 'D' in role:
						self.policy_pubs[role].publish(dact['policy'])
					elif 'I' in role:
						self.policy_pubs[role].publish(iact['policy'])

			for role in self.players:
				vx = self.vdes[role] * cos(actions[role])
				vy = self.vdes[role] * sin(actions[role])
				cmdV = Twist()
				cmdV.linear.x = vx
				cmdV.linear.y = vy
				self.cmdV_pubs[role].publish(cmdV)
				self.updateGoal(role, self.xs[role])
				self.goal_pubs[role].publish(self.goal_msgs[role])

			if self.is_capture():
				if self.last_cap:
					self.time_inrange += dt
				else:
					self.time_inrange = 0
				self.last_cap = True
				# print(self._time_inrange)
			if self.time_inrange > self.cap_time:
				self.end = True
				print('!!!!!!!!!!!!!!!!!!captured, game end!!!!!!!!!!!!')
				for role in self.players:
					self.xes[role] = deepcopy(self.xs[role])
				for role in self.players:
					if 'I' in role:
						if self.states[role] != 'land':
							self.states[role] = 'land'
							self.land_clients[role]()
					elif 'D' in role:
						self.states[role] = 'hover'
						self.auto_clients[role]()
						# print(role + self.states[role])

	def iteration(self, event):
		for role in self.players:
			if self.states[role] == 'hover':
				self.hover(role, self.xes[role])
				# print(role +'hover at: ', self.xes[role])
			elif self.states[role] == 'play':
				t = event.current_real - event.last_real
				self.game(t.secs + t.nsecs*1e-9)	  

if __name__ == '__main__':
	rospy.init_node('RAgame', anonymous=True)

	logger_dir = rospy.get_param("~logger_dir", '')
	Ds = rospy.get_param("~Ds", '').split(',')
	Is = rospy.get_param("~Is", '').split(',')

	zDs = rospy.get_param("~zDs", '')
	zIs = rospy.get_param("~zIs", '')
	if isinstance(zDs, str):
		zDs = zDs.split(',')
	elif isinstance(zDs, float):
		zDs = [zDs]
	if isinstance(zIs, str):
		zIs = zIs.split(',')
	elif isinstance(zIs, float):
		zIs = [zIs]

	param_file = rospy.get_param("~param_file", 'analytical_traj_param.csv')

	game = RAgame(zDs, zIs, rate=10, 
				  param_file=param_file,
				  Ds=Ds, Is=Is,
				  logger_dir=logger_dir)

	rospy.Timer(rospy.Duration(1.0/15), game.iteration)
	rospy.spin()				