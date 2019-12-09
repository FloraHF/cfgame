import numpy as np 
from math import sqrt, atan2, cos, acos, pi

def get_norm(x):
	return sqrt(x[0]**2 + x[1]**2)

def Iwin_wrapper(strategy):

	def wrapper(*args, **kwargs):
		D1_I, D2_I, D1_D2 = args[0].get_vecs()
		d1, d2, a1, a2 = args[0].get_alpha(D1_I, D2_I, D1_D2)
		tht = args[0].get_theta(D1_I, D2_I, D1_D2)

		if tht - (a1 + a2) - (pi - 2*args[0].gmm) > 0:
			acts = args[0].w_strategy()
			return acts
		else:
			acts = strategy(args[0])
			# print(strategy.__name__)
			for role in args[0].players:
				acts['p_'+role] = strategy.__name__
			return acts

	return wrapper

def nullWrapper(strategy):

	def wrapper(*args, **kwargs):
		acts = strategy(args[1])
		for role in args[0].players:
			acts['p_'+role] = strategy.__name__
		return acts

	return wrapper

def closeWrapper(dstrategy, istrategy): 

	def wrapper(*args, **kwargs):
		D1_I, D2_I, D1_D2 = args[0].get_vecs()
		base = args[0].get_base(D1_I, D2_I, D1_D2)

		vs = args[0].get_velocity()
		# print(args[0].last_act)
		if (get_norm(D1_I) < args[0].r_close and get_norm(D2_I) < args[0].r_close) or args[0].last_act['p_I0'] == 'both close':  # in both range
			vD1 = np.concatenate((vs['D0'], [0]))
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = args[0].k_close*phi_1 + base['D0']

			vD2 = np.concatenate((vs['D1'], [0]))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_2 = args[0].k_close*phi_2 + base['D1']		 
			
			psi = -args[0].get_theta(D1_I, D2_I, D1_D2)/2 + base['I0']

			action = {'D0': phi_1, 'D1': phi_2, 'I0': psi}
			for role in args[0].players:
				action['p_'+role] = 'both close'

		#=============== only D1 is close ===============#
		elif get_norm(D1_I) < args[0].r_close:  # in D1's range
			# print('D1 close', dstrategy.__name__)
			# print(vs['D0'])
			vD1 = np.concatenate((vs['D0'], [0]))
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = args[0].k_close*phi_1
			# print(phi_1)

			raw_act = dstrategy()
			phi_2 = raw_act['D1']

			I_T = np.concatenate((args[0].projection_on_target(args[1]['I0']) - args[1]['I0'], [0]))
			angT = atan2(np.cross(D1_I, I_T)[-1], np.dot(D1_I, I_T))
			if args[0].vd >= args[0].vi:
				psi = phi_1
			else:
				psi = - abs(acos(args[0].vd*cos(phi_1)/args[0].vi))

			psi = max(psi, angT) + base['D0']

			action = {'D0': phi_1+base['D0'], 'D1': phi_2, 'I0': psi}
			action['p_D0'] = 'D0 close'
			action['p_D1'] = raw_act['p_D1']
			action['p_I0'] = 'D0 close'

		#=============== only D2 is close ===============#
		elif get_norm(D2_I) < args[0].r_close:
			# print('D2 close')
			vD2 = np.concatenate((vs['D1'], [0]))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_2 = args[0].k_close * phi_2

			raw_act = dstrategy()
			# print(raw_act)
			phi_1 = raw_act['D0']

			I_T = np.concatenate((args[0].projection_on_target(args[1]['I0']) - args[1]['I0'], [0]))
			angT = atan2(np.cross(D2_I, I_T)[-1], np.dot(D2_I, I_T))
			if args[0].vd >= args[0].vi:
				psi = phi_2
			else:
				psi = abs(acos(args[0].vd * cos(phi_2)/args[0].vi))

			psi = min(psi, angT) + base['D1']

			action = {'D0': phi_1, 'D1': phi_2+base['D1'], 'I0': psi}
			action['p_D0'] = raw_act['p_D0']
			action['p_D1'] = 'D1 close'
			action['p_I0'] = 'D1 close'

		#============== no defender is close =============#
		else:
			# print(dstrategy.__name__, istrategy.__name__)
			dact = dstrategy()
			iact = istrategy()
			
			action = {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}
			for role in args[0].players:
				if 'D' in role:
					action['p_'+role] = dact['p_'+role]
				if 'I' in role:
					action['p_'+role] = iact['p_'+role]

		args[0].last_act = action
		return action

	return wrapper

def mixWrapper(dstrategy, istrategy): 

	def wrapper(*args, **kwargs):

		dact = dstrategy(args[1])
		iact = istrategy(args[1])
		
		action = {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}
		# print(dact)
		for role in args[0].players:
			if 'D' in role:
				action['p_'+role] = dact['p_'+role]
			if 'I' in role:
				action['p_'+role] = iact['p_'+role]
		# print(action)
		return action

	return wrapper	