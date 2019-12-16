#!/usr/bin/env python3
import numpy as np 
from math import sqrt, atan2, cos, acos, pi

def get_norm(x):
	return sqrt(x[0]**2 + x[1]**2)

def Iwin_wrapper(strategy):

	def wrapper(*args, **kwargs):
		# print(args)
		D1_I, D2_I, D1_D2 = args[0].get_vecs()
		d1, d2, a1, a2 = args[0].get_alpha(D1_I, D2_I, D1_D2)
		tht = args[0].get_theta(D1_I, D2_I, D1_D2)

		acts = strategy(args[0])
		for role in args[0].players:
			acts['p_'+role] = strategy.__name__

		if tht - (a1 + a2) - pi + 2*args[0].gmm0 > 0:
			adj_acts = args[0].w_strategy()
			acts['I0'] = adj_acts['I0']
			acts['p_I0'] = adj_acts['p_I0']
			# print(strategy.__name__)

		return acts

	return wrapper

def nullWrapper(strategy):

	def wrapper(*args, **kwargs):
		acts = strategy(args[0])
		for role in args[0].players:
			acts['p_'+role] = strategy.__name__
		return acts

	return wrapper

def closeWrapper(strategy):

	def wrapper(*args, **kwargs):
		# print(args)
		game = args[0]
		D1_I, D2_I, D1_D2 = game.get_vecs()
		base = game.get_base(D1_I, D2_I, D1_D2)

		vs = game.vs
		# print(game.last_act)
		if (get_norm(D1_I) < game.r_close and get_norm(D2_I) < game.r_close) or game.last_act['p_I0'] == 'both close':  # in both range
			vD1 = np.concatenate((vs['D0'], [0]))
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = game.k_close*phi_1 + base['D0']

			vD2 = np.concatenate((vs['D1'], [0]))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_2 = game.k_close*phi_2 + base['D1']		 
			
			psi = -game.get_theta(D1_I, D2_I, D1_D2)/2 + base['I0']

			action = {'D0': phi_1, 'D1': phi_2, 'I0': psi}
			for role in game.players:
				action['p_'+role] = 'both close'

		#=============== only D1 is close ===============#
		elif get_norm(D1_I) < game.r_close:  # in D1's range
			# print('D1 close', dstrategy.__name__)
			# print(vs['D0'])
			vD1 = np.concatenate((vs['D0'], [0]))
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = game.k_close*phi_1
			# print('phi_1', phi_1)

			raw_act = game.policy_dict[game.dstrategy]()
			phi_2 = raw_act['D1']

			I_T = np.concatenate((game.projection_on_target(game.xs['I0']) - game.xs['I0'], [0]))
			angT = atan2(np.cross(D1_I, I_T)[-1], np.dot(D1_I, I_T))
			# if game.vd >= game.vi:
			if game.vdes['I0'] <= game.vnorms['D0'][-1]:				
				psi = phi_1
			else:
				psi = - abs(acos(game.vnorms['D0'][-1]*cos(phi_1)/game.vdes['I0']))

			psi = max(psi, angT) + base['D0']

			action = {'D0': phi_1+base['D0'], 'D1': phi_2, 'I0': psi}
			action['p_D0'] = 'D0 close'
			action['p_D1'] = raw_act['p_D1']
			action['p_I0'] = 'D0 close'

		#=============== only D2 is close ===============#
		elif get_norm(D2_I) < game.r_close:
			# print('D2 close')
			vD2 = np.concatenate((vs['D1'], [0]))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_2 = game.k_close * phi_2
			# print('phi_2', phi_2)

			raw_act = game.policy_dict[game.dstrategy]()
			# print(raw_act)
			phi_1 = raw_act['D0']

			I_T = np.concatenate((game.projection_on_target(game.xs['I0']) - game.xs['I0'], [0]))
			angT = atan2(np.cross(D2_I, I_T)[-1], np.dot(D2_I, I_T))
			# if game.vd >= game.vi:
			if game.vdes['I0'] <= game.vnorms['D1'][-1]:
				psi = phi_2
			else:
				psi = abs(acos(game.vnorms['D1'][-1]*cos(phi_2)/game.vdes['I0']))

			psi = min(psi, angT) + base['D1']

			action = {'D0': phi_1, 'D1': phi_2+base['D1'], 'I0': psi}
			action['p_D0'] = raw_act['p_D0']
			action['p_D1'] = 'D1 close'
			action['p_I0'] = 'D1 close'

		#============== no defender is close =============#
		else:
			# print(dstrategy.__name__, istrategy.__name__)
			dact = game.policy_dict[game.dstrategy]()
			iact = game.policy_dict[game.istrategy]()
			
			action = {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}
			for role in game.players:
				if 'D' in role:
					action['p_'+role] = dact['p_'+role]
				if 'I' in role:
					action['p_'+role] = iact['p_'+role]

		game.last_act = action
		return action

		# 	return wrapper
		# else:
		# 	def wrapper(*args, **kwargs):

		# 		dact = game[game.dstrategy](game)
		# 		iact = game[game.istrategy](game)
				
		# 		action = {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}
		# 		# print(dact)
		# 		for role in game.players:
		# 			if 'D' in role:
		# 				action['p_'+role] = dact['p_'+role]
		# 			if 'I' in role:
		# 				action['p_'+role] = iact['p_'+role]
		# 		# print(action)
		# 		return action

	return wrapper	

	# return outer_wrapper()

def mixWrapper(strategy):

	def wrapper(*args, **kwargs):

		dact = args[0].policy_dict[args[0].dstrategy]()
		iact = args[0].policy_dict[args[0].istrategy]()
		
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
