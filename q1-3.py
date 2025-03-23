import numpy as np
import cv2


def pad(img, w, h):
    hei, wid, chan = img.shape
    imgPad = np.zeros((hei + 2 * h, wid + 2 * w, chan), img.dtype)
    imgPad[h:hei + h, w:wid + w, :] = img
    return imgPad


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


def getHomography(img1, img2):
    hei, wid, chan = img1.shape
    img1r = cv2.resize(img1, (wid // k, hei // k))
    img2r = cv2.resize(img2, (wid // k, hei // k))
    H = warp(img1r, img2r)
    H = np.matmul(np.matmul(S, H), np.linalg.inv(S))
    return H


def write(H):
    string = ''
    for i in range(3):
        for j in range(3):
            string = string + str(H[i, j]) + ' '
    file.write(string + '\r')


k = 2
file = open("Homographies test.txt", "w")
padW, padH = 2500, 1000
I1 = cv2.imread('frames/90.jpg')
I2 = cv2.imread('frames/270.jpg')
I3 = cv2.imread('frames/450.jpg')
I4 = cv2.imread('frames/630.jpg')
I5 = cv2.imread('frames/810.jpg')
hei, wid, chan = I1.shape
S = np.array([[k, 0., 0.], [0., k, 0.], [0., 0., 1.]])
I1p = pad(I1, padW, padH)
I2p = pad(I2, padW, padH)
I3p = pad(I3, padW, padH)
I4p = pad(I4, padW, padH)
I5p = pad(I5, padW, padH)
heip, widp, chanp = I1p.shape
H12 = getHomography(I1p, I2p)
H23 = getHomography(I2p, I3p)
H43 = getHomography(I4p, I3p)
H54 = getHomography(I5p, I4p)
H13 = np.matmul(H12, H23)
H53 = np.matmul(H54, H43)

for i in range(0, 900):
    print(i)
    filePath = 'frames/' + str(i + 1) + '.jpg'
    I = cv2.imread(filePath)
    Ip = pad(I, padW, padH)
    if i < 180:
        Ikey, Hkey = I1p, H13
    elif i < 360:
        Ikey, Hkey = I2p, H23
    elif i < 540:
        Ikey, Hkey = I3p, np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elif i < 720:
        Ikey, Hkey = I4p, H43
    else:
        Ikey, Hkey = I5p, H53
    H = getHomography(Ip, Ikey)
    H = np.matmul(H, Hkey)
    write(H)
    Iwarped = cv2.warpPerspective(Ip, H, (widp, heip))
    cv2.imwrite('warped frames/' + str(i + 1) + '.jpg', Iwarped)

out = cv2.VideoWriter("res05-reference-plane.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (widp, heip))
for i in range(900):
    I = cv2.imread('warped frames/' + str(i + 1) + '.jpg')
    out.write(I)
out.release()
