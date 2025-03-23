import numpy as np
import cv2


def makeFrames():
    cap = cv2.VideoCapture('res07-background-video.mp4')
    i = 1
    while (cap.isOpened()):
        print(i)
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imwrite('frames bg/' + str(i) + '.jpg', frame)
        i += 1
    cap.release()

def warp(img1, img2):
    thresh = 0.7
    sift = cv2.SIFT_create()
    keyPoints1, dst1 = sift.detectAndCompute(img1, None)
    keyPoints2, dst2 = sift.detectAndCompute(img2, None)
    matches = cv2.BFMatcher().knnMatch(dst1, dst2, 2)
    points1 = []
    points2 = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < thresh * m2.distance:
            points1.append(keyPoints1[m1.queryIdx].pt)
            points2.append(keyPoints2[m1.trainIdx].pt)

    H, maskRsc = cv2.findHomography(np.uint16(points1), np.uint16(points2), cv2.RANSAC, 5, maxIters=1000)
    return H

makeFrames()
I = cv2.imread('frames/1.jpg')
hei, wid, chan = I.shape
thresh = 30000
out = cv2.VideoWriter("res08-foreground-video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (wid, hei))
for i in range(270, 900):
    print(i)
    I = cv2.imread('frames/' + str(i + 1) + '.jpg')
    Ibg = cv2.imread('frames bg/' + str(i + 1) + '.jpg')
    H=warp(I,Ibg)
    Iw=cv2.warpPerspective(I, H, (wid, hei))
    Idif = np.sum((np.float64(Iw) - np.float64(Ibg)) ** 2, axis=2)
    Ifg = I.copy()
    ii, jj = np.where(Idif > thresh)
    Ifg[ii,jj,:]=[0,0,255]
    out.write(np.uint8(Ifg))
out.release()
