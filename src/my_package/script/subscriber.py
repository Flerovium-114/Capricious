#!/usr/bin/env python

# import libraries
import rospy
from std_msgs.msg import String


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)


def listener():
    # initialize 'listener' node
    rospy.init_node('listener', anonymous=True)
    # node subscribes to the topic 'convo'
    rospy.Subscriber("convo", String, callback)

    rospy.spin() # ensures that python doesn't exit the code until the node is stopped

if __name__ == '__main__':
    listener()
