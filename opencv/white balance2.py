import numpy as np
import cv2

def awb(i):
    j = str(i)
    img = cv2.imread('C:/Users/User/PycharmProjects/untitled1/crawling/' + j + '.jpg', cv2.IMREAD_COLOR)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    cv2.imshow('loadimage', img)
    cv2.waitKey(1)

    y, cr, cb = cv2.split(img_ycrcb)

    cr_mean = np.mean(cr)
    cb_mean = np.mean(cb)

    crm = int(round(128 - cr_mean))
    cbm = int(round(128 - cb_mean))

    y2 = y
    cr2 = cv2.add(cr, crm)
    cb2 = cv2.add(cb, cbm)

    img2_ycrcb = cv2.merge((y2, cr2, cb2))
    img2 = cv2.cvtColor(img2_ycrcb, cv2.COLOR_YCR_CB2BGR)

    cv2.imshow('loadimage2', img2)
    cv2.waitKey(1)

    cv2.imwrite('C:/Users/User/PycharmProjects/untitled1/awb/' + j + '_wb.jpg', img2)

i = 1

while i <= 1211:
    awb(i)
    i += 1

cv2.destroyAllWindow()
