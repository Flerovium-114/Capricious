#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Define the ROS node and topic
rospy.init_node('shape_detection')
image_pub = rospy.Publisher("image_topic", Image, queue_size=10)

# Define the CV bridge
bridge = CvBridge()

# Define the video capture
cap = cv2.VideoCapture(0)

# Loop to read the video and detect shapes
while not rospy.is_shutdown():
    # Read the video frame
    ret, frame = cap.read()

    # Convert the video frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour
    for contour in contours:
        # Find the shape of the contour
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)  # increase precision to take in smaller images and vice versa
        shape = len(approx)

        # Draw the shape and label it
        if shape == 3:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            cv2.putText(frame, "Triangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif shape == 4:
            cv2.drawContours(frame, [approx], 0, (0, 0, 255), 3)
            cv2.putText(frame, "Rectangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	elif shape == 5:
            cv2.drawContours(frame, [approx], 0, (0, 255, 255), 3)
            cv2.putText(frame, "Pentagon", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
	elif shape == 6:
            cv2.drawContours(frame, [approx], 0, (50, 25, 255), 3)
            cv2.putText(frame, "Hexagon", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 25, 255), 2)

        #else:
	    #cv2.drawContours(frame, [approx], 0, (255, 0, 0), 3)
            #cv2.putText(frame, "Circle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert the image to a ROS message and publish it
    try:
        image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    except CvBridgeError as e:
        print(e)

    # Display the video frame
    cv2.imshow("Frame", frame)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()


