import cv2
import numpy as np
import os

script_path = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(script_path, '..', 'data/collect_stop_sign/')

# Load image
image = cv2.imread(PATH + '0000120.00.jpg')

# Convert image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(hsv[7][10])
# Define lower and upper bounds for the color you want to threshold
lower_color = np.array([100, 90, 100])
upper_color = np.array([150, 120, 150])

lower_color = np.array([120, 70, 100])
upper_color = np.array([180,150,150])

# Threshold the HSV image to get only the color within the specified range
mask = cv2.inRange(hsv, lower_color, upper_color)

# Apply bitwise AND operation to get the color regions from the original image
color_thresholded_image = cv2.bitwise_and(image, image, mask=mask)

# Display the result
cv2.imshow('Color Thresholded Image', color_thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
