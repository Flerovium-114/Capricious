#!/usr/bin/env python
    
# import neceessary libraries
import rospy
from std_msgs.msg import String
    
def talker() :
    # For node to publish to the topic 'convo'
    pub = rospy.Publisher('convo', String, queue_size=10)

    # Initialize node named talker
    rospy.init_node('talker', anonymous=True)

    rate = rospy.Rate(10) # 10hz 

    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str) #writes message to the node's logfile
        pub.publish(hello_str)   # publish the message
        rate.sleep()

# Standard Python __name__ check and interruption check 
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
