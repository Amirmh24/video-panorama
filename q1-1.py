import numpy as np
import cv2

def makeFrames():
    cap = cv2.VideoCapture('video.mp4')
    i = 1
    while (cap.isOpened()):
        print(i)
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imwrite('frames/' + str(i) + '.jpg', frame)
        if i==900:break
        i += 1
    cap.release()

def norm(X):
    X = X / X[2]
    return X[:2]


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

    H, maskRsc = cv2.findHomography(np.uint16(points1), np.uint16(points2), cv2.RANSAC, 5, maxIters=176)
    return H


def getMask(img):
    hei, wid, chan = img.shape
    mask = np.zeros((hei, wid))
    for c in range(3):
        mask[img[:, :, c] != 0] = 1
    kernel = (3, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def put(img, bck):
    mask = getMask(img)
    for c in range(3):
        i, j = np.where(mask == 1)
        bck[i, j, c] = img[i, j, c]
    return mask, bck


makeFrames()
padW, padH = 2000, 500
I1 = cv2.imread('frames/270.jpg')
I2 = cv2.imread('frames/450.jpg')
hei, wid, chan = I1.shape
I1 = cv2.resize(I1, (wid, hei))
I2 = cv2.resize(I2, (wid, hei))
I1p = pad(I1, padW, padH)
I2p = pad(I2, padW, padH)
heip, widp, chanp = I1p.shape

H = warp(I1, I2)
p = np.array([[600, 300], [600, 900], [1200, 900], [1200, 300]], dtype=np.float32)
q = cv2.transform(np.array([p]), np.linalg.inv(H))[0]
q = np.array([norm(q[0]), norm(q[1]), norm(q[2]), norm(q[3])], dtype=np.int32)
I450rect = cv2.rectangle(I2.copy(), tuple(np.int32(p[0])), tuple(np.int32(p[2])), color=(0, 0, 255), thickness=3)
I270rect = cv2.polylines(I1.copy(), [q.reshape((-1, 1, 2))], color=(0, 0, 255), isClosed=True, thickness=3)
cv2.imwrite('res02-270-rect.jpg', I270rect)
cv2.imwrite('res01-450-rect.jpg', I450rect)

H = warp(I1p, I2p)
I1p = cv2.warpPerspective(I1p, H, (widp, heip))
IMerged = np.zeros((heip, widp, 3))
mask1, IMerged = put(I1p, IMerged)
mask2, IMerged = put(I2p, IMerged)
cv2.imwrite('res03-270-450-panorama.jpg', IMerged)
