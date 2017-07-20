import cv2
import numpy as np
import csv
import math

with open("test.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    data = list(spamreader)


data_np = np.array(data[1:])
data_np = data_np.astype(np.uint8)

i = 0

for row in data_np:

    dim = int(math.sqrt(len(row)))
    image = np.reshape(row, (dim, dim))
    print(image)
    cv2.imshow("number", image)
    cv2.waitKey(30)
    cv2.imwrite("D:/Machine Learning/number/test_image/" + str(i) + ".jpeg", image)
    i = i+1
    
