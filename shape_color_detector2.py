#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

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
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Apply edge detection
    #edges = cv2.Canny(gray, 100, 200)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

   # Define color ranges for detection
    # In this example, we're detecting red, green, and blue colors
    lower_red = (0, 100, 100)
    upper_red = (10, 255, 255)
    lower_green = (50, 100, 100)
    upper_green = (70, 255, 255)
    lower_blue = (110, 100, 100)
    upper_blue = (130, 255, 255)

    # Create masks for each color range
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the masks
    mask = cv2.bitwise_or(mask_red, mask_green, mask_blue)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours in the image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour
    for contour in contours:
        # Find the shape of the contour
        approx = cv2.approxPolyDP(contour,   0.05*cv2.arcLength(contour, True), True)  # increase precision to take in smaller images and vice versa
        shape = len(approx)

        # Use moments to determine the centroid of the contour
        moments = cv2.moments(contour)
        #print(moments['m00'])

	if moments['m00'] != 0:      # to eliminate division by zero error
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
	else:
	    continue

        # Determine the color of the shape based on the centroid
        color = ""
        if mask[cy, cx] == mask_red.max():
            color = "red"
        elif mask[cy, cx] == mask_green.max():
            color = "green"
        elif mask[cy, cx] == mask_blue.max():
            color = "blue"

        # Draw the shape and label it with colour
        if shape == 3:
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            cv2.putText(frame, color + "Triangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif shape == 4:
            cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
            cv2.putText(frame, color + "Rectangle", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	elif shape == 5:
            cv2.drawContours(frame, [approx], 0, (0, 255, 255), 2)
            cv2.putText(frame, color + "Pentagon", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
	elif shape == 6:
            cv2.drawContours(frame, [contour], 0, (50, 25, 255), 2)
            cv2.putText(frame,color + "Hexagon", (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 25, 255), 2)
    
         # Commenting out the circles because it was detecting too many circles in the environment
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


