#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse


class gameServer(object):

	def __init__(self, Ds='', Is=''):

		self._player_dict = dict()
		for i, D in enumerate(Ds):
			if D != '':
				self._player_dict['D'+str(i+1)] = D
		for i, I in enumerate(Is):
			if I != '':
				self._player_dict['I'+str(i+1)] = I

		self._takeoff_clients = self._create_client_list('/set_takeoff')
		self._play_clients = self._create_client_list('/set_play')
		self._land_clients = self._create_client_list('/set_land')

		rospy.Service('alltakeoff', Empty, self._alltakeoff)
		rospy.Service('allplay', Empty, self._allplay)
		rospy.Service('allland', Empty, self._allland)

	def _create_client_list(self, name):
		clients = []
		for plyr, cf in self._player_dict.items():
			if len(cf) > 0:
				srv_name = '/' + cf + name
				rospy.loginfo('game controller: waiting for ' + srv_name + ' service')
				rospy.wait_for_service(srv_name)
				rospy.loginfo('game controller: found' + srv_name + ' service')
				clients.append(rospy.ServiceProxy(srv_name, Empty))
		return clients

	def _alltakeoff(self, req):
		for takeoff in self._takeoff_clients:
			takeoff()
		return EmptyResponse()
		
	def _allplay(self, req):
		for play in self._play_clients:
			play()
		return EmptyResponse()

	def _allland(self, req):
		for land in self._land_clients:
			land()
		return EmptyResponse()

if __name__ == '__main__':

	rospy.init_node('game_server', anonymous=True)

	Ds = rospy.get_param("~Ds", '').split(',')
	Is = rospy.get_param("~Is", '').split(',')

	game_server = gameServer(Ds=Ds, Is=Is)

	rospy.spin()
