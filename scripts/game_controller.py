import rospy
from std_srvs.srv import Empty

class gameController(object):

	def __init__(self, D1='cf4', D2='cf5', I='cf3'):
		self._player_dict = {'D1': D1, 'D2': D2, 'I': I}
		self._takeoff_clients = self._create_client_list('/set_takeoff')
		self._play_clients = self._create_client_list('/set_play')
		self._land_clients = self._create_client_list('/set_land')

	    rospy.Service('alltackoff', Empty, self._alltakeoff)
	    rospy.Service('allplay', Empty, self._allplay)
	    rospy.Service('allland', Empty, self._allland)

	def _create_client_list(self, name):
		clients = []
		for plyr, cf in self._player_dict.items()
			srv_name = '/' + cf + name
			rospy.wait_for_service(srv_name)
			rospy.loginfo('game controller: found' + srv_name + 'service')
			clients.append(rospy.ServiceProxy(srv_name, Empty))
		return clients

	def _alltakeoff(self, request):
		for takeoff in self._takeoff_clients:
			takeoff()
		return Empty()
		
	def _allplay(self, request):
		for play in self._play_clients:
			play()
		return Empty()

	def _allland(self, request):
		for land in self._land_clients:
			land()
		return Empty()

if __name__ = '__main__':

    rospy.init_node('game_controller', anonymous=True)

    D1 = rospy.get_param("~D1", 'cf4')
    D2 = rospy.get_param("~D2", 'cf5')
    I = rospy.get_param("~I", 'cf3')

    game_controller = gameController(D1=D1, D2=D2, I=I)

    rospy.spin()