import numpy as np
import cv2

MIN_MATCH_COUNT = 10

img1 = cv2.imread('C:/Users/User/PycharmProjects/untitled1/heart.png',1)          # queryImage
img2 = cv2.imread('C:/Users/User/PycharmProjects/untitled1/gg.jpg',1) # trainImage

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('loadimage', img2)
cv2.waitKey(0)

# Initiate SURF detector
surf = cv2.xfeatures2d.SURF_create()

# find the keypoints and descriptors with SURF
kp1, des1 = surf.detectAndCompute(img1_gray,None)
kp2, des2 = surf.detectAndCompute(img2_gray,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2_gray = cv2.polylines(img2_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1_gray,kp1,img2_gray,kp2,good,None,**draw_params)

cv2.imshow('SURF', img3)
cv2.waitKey(0)

point = np.int32(dst)
point1 = point[0][0]
point2 = point[1][0]
point3 = point[2][0]
point4 = point[3][0]

x = int((point1[0]+point2[0])/2)
y = int((point1[1]+point4[1])/2)
w = int((point3[0]+point4[0])/2)
h = int((point2[1]+point3[1])/2)

img_trim = img2[y:h, x:w]
cv2.imshow('trim', img_trim)
cv2.waitKey(0)

img_trim_hsv = cv2.cvtColor(img_trim, cv2.COLOR_BGR2HSV)

lower_blue = np.array([110, 30, 100])
upper_blue = np.array([130, 255, 255])
lower_green = np.array([50, 30, 100])
upper_green = np.array([70, 255, 255])
lower_red = np.array([-10, 30, 100])
upper_red = np.array([255, 255, 255])

mask_red = cv2.inRange(img_trim_hsv, lower_red, upper_red)

res = cv2.bitwise_and(img_trim, img_trim, mask=mask_red )

cv2.imshow('red', res)
cv2.waitKey(0)

res2 = img_trim-res
cv2.imshow('res',res2)
cv2.waitKey(0)

res2_ycrcb = cv2.cvtColor(res2, cv2.COLOR_BGR2YCR_CB)
y, cr, cb = cv2.split(res2_ycrcb)

y_mean = np.mean(y)
cr_mean = np.mean(cr)
cb_mean = np.mean(cb)

ym = int(round(239 - y_mean))
crm = int(round(128 - cr_mean))
cbm = int(round(128 - cb_mean))

y2 = cv2.add(y, ym)
cr2 = cv2.add(cr, crm)
cb2 = cv2.add(cb, cbm)

res2_ycrcb = cv2.merge((y2, cr2, cb2))
res2 = cv2.cvtColor(res2_ycrcb, cv2.COLOR_YCR_CB2BGR)

cv2.imshow('loadimage2', res2)
cv2.waitKey(0)

img2_ycrcb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
img2_y, img2_cr, img2_cb = cv2.split(img2_ycrcb)

img2_y2 = cv2.add(img2_y, ym)
img2_cr2 = cv2.add(img2_cr, crm)
img2_cb2 = cv2.add(img2_cb, cbm)

img2_ycrcb = cv2.merge((img2_y, img2_cr2, img2_cb2))
img2 = cv2.cvtColor(img2_ycrcb, cv2.COLOR_YCR_CB2BGR)

cv2.imshow('loadimage3', img2)
cv2.waitKey(0)

