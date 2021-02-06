#!/usr/bin/env python3

import cv2
import numpy as np


def loss(img, other):
    res = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            res += (img.item(i, j)/255 - other.item(i, j)/255)**2
    return res

def mean(img):
    res = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            res += img.item(i, j)/255
    return res/(len(img)*len(img[0]))

def loss_2(img, other):
    s_1 = 0
    s_2 = 0
    s_3 = 0
    m_f = mean(img)
    m_g = mean(other)
    for i in range(len(img)):
        for j in range(len(img[0])):
            s_1 += (img.item(i, j)/255 - m_f)*(other.item(i, j)/255 - m_g)
            s_2 += (img.item(i, j)/255 - m_f)**2
            s_3 += (other.item(i, j)/255 - m_g)**2
    return s_1/np.sqrt((s_2*s_3))

img = cv2.imread("./../data/clean_finger.png", 0)

other = cv2.imread("./../data/tx_finger.png", 0)
rows, cols = img.shape

other_2 = cv2.imread("./../data/txy_finger.png", 0)

## to create the plot for tx_finger


# with open("data.txt", "w") as f:

#     for k in range(-300, 300):
#         M = np.float32([[1,0,k],[0,1,0]])
#         translated = cv2.warpAffine(other,M,(cols, rows))
#         f.write(str(loss(img, translated)))
#         f.write("\n")

## to create the seconde plot for txy_finger

# with open("data_2.txt", "w") as f:
#     for i in range(-20, 20):
#         for j in range(-20, 20):
#             M = np.float32([[1,0,i*5],[0,1,j*5]])
#             translated = cv2.warpAffine(other,M,(cols, rows))
#             f.write(str(loss(img, translated)) + ",")
#         f.write("\n")

# to do the plot for tx but with the second loss function

# with open("data.txt", "w") as f:
#     for k in range(-200, 200):
#         M = np.float32([[1,0,k],[0,1,0]])
#         translated = cv2.warpAffine(other,M,(cols, rows))
#         f.write(str(loss_2(img, translated)))
#         f.write("\n")

with open("data_2.txt", "w") as f:
    for i in range(-20, 20):
        for j in range(-20, 20):
            M = np.float32([[1,0,i*5],[0,1,j*5]])
            translated = cv2.warpAffine(other,M,(cols, rows))
            f.write(str(loss_2(img, translated)) + ",")
        f.write("\n")