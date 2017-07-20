import cv2
import math
import pickle
import numpy as np
from tsp_solver.greedy import solve_tsp
import matplotlib.pyplot as plt
import csv
from os import listdir
import os



def make_dist_matrix(x, y):
    """Creates distance matrix for the given coordinate vectors"""
    N = len(x)
    xx = np.vstack( (x,)*N )
    yy = np.vstack( (y,)*N )
    return np.sqrt( (xx - xx.T)**2 + (yy - yy.T)**2 )

directory = "D:/Machine Learning/number/test_image/Thin_noNoise/"
filedic = listdir(directory)

file_extension = os.path.splitext(os.path.basename(filedic[0]))

print(file_extension)

out = open( directory + "RNN_test.csv", 'w', newline='')
for num in range(len(filedic)):

    print(num)
    # Training data
    #ori = cv2.imread( directory + filedic[num], cv2.IMREAD_GRAYSCALE)

    # Testing data
    ori = cv2.imread( directory + str(num) + file_extension[1], cv2.IMREAD_GRAYSCALE)

    cv2.imshow("img", ori)
    cv2.waitKey(20)
    mg = np.array(ori)
    ret, temp = cv2.threshold(ori, 10, 255, cv2.THRESH_BINARY)
    pix = np.array([0,0])

    imgArray = np.array(temp)
    for x in range(len(temp)):
        for y in range(len(temp[1])):
            if temp[x][y] == 255:
                #pix = np.array([[x,y]])
                pix = np.vstack((pix, np.atleast_2d(np.array([[x,y]]))))
                
            
    point = pix.flatten()
    x = point[2: len(point): 2]
    y = point[3: len(point): 2]

    pos = np.stack((x,y))

    path = solve_tsp(make_dist_matrix(x, y))

    #plt.plot( pos[1, path],pos[0, path], 'k-')
    #plt.gca().invert_yaxis()
    #plt.show()


    # Use the first character in the filename to distinguish label
    # If this is used in a testing data set, please ignore the first column in the generated csv file.
    
    dx = [ord(filedic[num][:1])-48]
    dy = [ord(filedic[num][:1])-48]

    pos_tsp_x = (pos[0, path])
    pos_tsp_y = (pos[1, path])

    for i in range(0, len(pos_tsp_x)-1):
        dx.append(pos_tsp_x[i+1] - pos_tsp_x[i])
        dy.append(pos_tsp_y[i+1] - pos_tsp_y[i])


    #print(dx)
    #print(dy)

    writer = csv.writer(out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    writer.writerow(dx)
    writer.writerow(dy)
    

out.close()
cv2.waitKey(0)

