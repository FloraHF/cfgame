#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty, EmptyResponse


def takeoff_response(req):
    return EmptyResponse()


def land_response(req):
    return EmptyResponse()


def play_response(req):
    return EmptyResponse()


def emergency_response(req):
    return EmptyResponse()


if __name__ == "__main__":

    rospy.init_node('services')

    rospy.Service('takeoff', Empty, takeoff_response)
    print("takeoff service created")

    rospy.Service('land', Empty, land_response)
    print("landing service created")

    rospy.Service('emergency', Empty, emergency_response)
    print("emergency service created")

    rospy.Service('play', Empty, play_response)
    print("play service created")  
      
    rospy.spin()
