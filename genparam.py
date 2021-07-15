import numpy as np
import math
import os

def main():
    theta_x = 20 / 180 * np.pi 
    theta_y = -22 / 180 * np.pi
    theta_z = 0 # should always be 0

    K = np.array([
        [774.882, 0, 480],
        [0, 774.882, 270],
        [0, 0, 1]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(theta_x), -1 * math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ])

    Ry = np.array([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-1*math.sin(theta_y), 0, math.cos(theta_y)]
    ])

    Rz = np.array([
        [math.cos(theta_z), -1 * math.sin(theta_z), 0],
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    param = K @ Rx @ Ry @ Rz
    # print(param[0][2])
    s = ""
    for i in range(0, 3):
        for j in range(0, 3):
            s = s + "{:.3f}".format(param[i][j]) + " "
        s = s + "0.0 "
    
    print(s)
    f = open("calib/MVI_40152.txt", "w")
    f.write("P_rect_02: " + s)

if __name__ == '__main__':
    main()
