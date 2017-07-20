import cv2
import math
import pickle
import numpy as np
from tsp_solver.greedy import solve_tsp
import matplotlib.pyplot as plt
import csv

def make_dist_matrix(x, y):
    """Creates distance matrix for the given coordinate vectors"""
    N = len(x)
    xx = np.vstack( (x,)*N )
    yy = np.vstack( (y,)*N )
    return np.sqrt( (xx - xx.T)**2 + (yy - yy.T)**2 )
  

ori = cv2.imread("D:/Machine Learning/number/Thin_noNoise/5_273.jpg", cv2.IMREAD_GRAYSCALE)
img = np.array(ori)
ret, temp = cv2.threshold(ori, 10, 255, cv2.THRESH_BINARY)
cv2.imshow("ori", temp)
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

path = solve_tsp(make_dist_matrix(x, y),)

print(pix)
print(path)



plt.plot( pos[1, path],pos[0, path], 'k-')

for i, (_x, _y) in enumerate(zip(pos[1, path],pos[0, path])):
    plt.text(_x, _y, i, color = 'red', fontsize = 10)
plt.gca().invert_yaxis()
plt.show()



dx = []
dy = []
pos_tsp_x = (pos[0, path])
pos_tsp_y = (pos[1, path])
for i in range(0, len(pos_tsp_x)-1):
    dx.append(pos_tsp_x[i+1] - pos_tsp_x[i])
    dy.append(pos_tsp_y[i+1] - pos_tsp_y[i])


cv2.waitKey(0)

