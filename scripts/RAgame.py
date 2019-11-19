class Strategy(object):

	def __init__(self, vd, vi, zsD, zsI, r, 
				 worldFrame, frame, rate=10, a='',
				 r_close=1.2, k_close=.9,
				 Ds='', Is=''):

		self.worldFrame = worldFrame
		self.frame = frame
		self.rate = rospy.Rate(rate)
		# self.states = dict()

		self.vd = vd
		self.vi = vi
		self.zs = dict()
		self.x0s = dict()
		self.goal_msgs = dict()
		self.policy_fns = dict()
		self.r = r
		self.nv = 20
		a = a.split('/')
        self.a_anl = float(a[0]) / float(a[1])
        self.LB = acos(self.a_anl)

		self.cap_time = .2
        self.k_close = k_close
        self.r_close = r_close*r
        self.last_cap = False
        self.end = False
        self.activates = dict()
        self.time_inrange = 0.
        self.time_end = 0.        

        self.velocities = dict()
		self.locations = dict()
		self.vel_norms = dict()

		self.mocap_sub_callbacks = {'D0': self.getD0, 'D1': self.getD1, 'I0': self.getI0}
        self.mocap_subs = dict()

		self.players = dict()
		self.goal_pubs = dict()
		self.cmdV_pubs = dict()
		self.policy_pubs = dict()
		
        self.takeoff = dict()
        self.auto = dict()
        self.land = dict()
        self.play = dict()

        script_dir = os.path.dirname(__file__)
        self._info_dir = os.path.join(script_dir, 'info_slowD.csv')
		with open(self._info_dir, 'r') as f:
        	lines = f.readlines()
        	for line in lines:
	            if 'x' in line:
	            	data = line.split(',')
	            	role = data[0][1:]
	            	self.x0s[role] = np.array([float(data[1]), float(data[2])])
	            	self.locations[role] = np.array([float(data[1]), float(data[2])])      
		self._vecs = {'D0_I0': np.concatenate((self.locations['I0'] - self.locations['D0'], [0])),
                      'D1_I0': np.concatenate((self.locations['I0'] - self.locations['D1'], [0])),
                      'D0_D1': np.concatenate((self.locations['D1'] - self.locations['D0'], [0]))}

		for i, (D, z) in enumerate(zip(Ds, zsD)):
			if D != '':
				role = 'D'+str(i)
				self.player_dict[role] = D
				self.zs[role] = z
				self.activates[role] = False
				self.goal_msgs[role] = PoseStamped()
				self.updateGoal(D, goal=self.x0s[role], init=True)

				self.vel_norms[role] = []
				self.velocities[role] = np.zeros(2)
				self.policy_fns[role] = load_model('PolicyFn_'+D)
				# self.states[role] = None

				self.goal_pubs[role] = rospy.Publisher('/'+D+'/goal', PoseStamped, queue_size=1)
				self.cmdV_pubs[role] = rospy.Publisher('/'+D+'/cmdV', PoseStamped, queue_size=1)
				self.plcy_pubs[role] = rospy.Publisher('/'+D+'/policy', PoseStamped, queue_size=1)
				self.a_pubs[role]    = rospy.Publisher('/'+D+'/a', PoseStamped, queue_size=1)
				
				self.takeoff_clients[role]= self.service_client(D, '/cftakeoff')
				self.land_clients[role]   = self.service_client(D, '/cfland')
				self.play_clients[role]   = self.service_client(D, '/cfplay')
				self.auto_clients[role]   = self.service_client(D, '/cfauto')
				self.mocap_subs[role] = rospy.Subscriber('/'+D+'/mocap', Mocap, self.mocap_sub_callbacks[role])

		for i, (I, z) in enumerate(zip(Is, zsI)):
			if I != '':
				role = 'I'+str(i)
				self._player_dict[role] = I
				self.goal_pubs[role] = rospy.Publisher('/'+I+'/goal', PoseStamped, queue_size=1)

		rospy.Service('alltakeoff', Empty, self.alltakeoff)
		rospy.Service('allplay', Empty, self.allplay)
		rospy.Service('allland', Empty, self.allland)

	def service_client(self, cf, name):
        srv_name = '/' + cf + name
        rospy.wait_for_service(srv_name)
        rospy.loginfo('found' + srv_name + 'service')
        return rospy.ServiceProxy(srv_name, Empty)

	def alltakeoff(self, req):
		for role, takeoff in self.takeoff_clients.items():
			takeoff()
		return EmptyResponse()

	def allplay(self, req):
		for role, play in self.play_clients.items():
			takeoff()
		return EmptyResponse()

	def allland(self, req):
		for role, land in self.land_clients.items():
			takeoff()
		return EmptyResponse()

    def getD0(self, data):
        self.locations['D0'] = np.array([data.position[0], data.position[1]])
        self.velocities['D0'] = np.array([data.velocity[0], data.velocity[1]])
        self.vecs['D0_I0'] = np.concatenate((self._locations['I0'] - self._locations['D0'], [0]))
        self.vecs['D1_D0'] = np.concatenate((self._locations['D0'] - self._locations['D1'], [0]))
        if len(self.vel_norms['D0']) > self.nv:
            self._vel_norms['D0'].pop(0)
        self.vel_norms['D0'].append(sqrt(data.velocity[0]**2 + data.velocity[1]**2))		