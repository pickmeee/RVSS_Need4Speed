import cv2
import numpy as np
import os


def mouseRGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorsB = image[y, x, 0]
        colorsG = image[y, x, 1]
        colorsR = image[y, x, 2]
        colors = image[y, x]
        hsv_value = np.uint8([[[colorsB, colorsG, colorsR]]])
        hsv = cv2.cvtColor(hsv_value, cv2.COLOR_BGR2HSV)

        print("HSV : ", hsv)
        # print("Red: ", colorsR)
        # print("Green: ", colorsG)
        # print("Blue: ", colorsB)
        # print("BRG Format: ", colors)
        # print("Coordinates of pixel: X: ", x, "Y: ", y)

if __name__ == "__main__":
    l1 = 500
    l2 = 500
    l3 = 500
    h1 = 0
    h2 = 0
    h3 = 0

    script_path = os.path.dirname(os.path.realpath(__file__))
    PATH = os.path.join(script_path, '..', 'data/collect_stop_sign/')

    # Load image
    image = cv2.imread(PATH + '0007900.00.jpg')

    cv2.namedWindow('mouseRGB')
    cv2.setMouseCallback('mouseRGB', mouseRGB)


    # Do until esc pressed
    while (1):
        cv2.imshow('mouseRGB', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    # if esc pressed, finish.
    cv2.destroyAllWindows()




