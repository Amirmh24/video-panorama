import numpy as np
import cv2


class Line:
    def __init__(self, p1, p2):
        self.a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        self.b = p1[1] - self.a * p1[0]

    def get(self, x):
        return self.a * x + self.b


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


def cleanList(imask, jmask):
    imaskCleaned, jmaskCleaned = np.array([imask[0]]), np.array([jmask[0]])
    for i in range(len(imask)):
        s = True
        k = len(imaskCleaned)
        for j in range(k):
            dist = ((imask[i] - imaskCleaned[j]) ** 2 + (jmask[i] - jmaskCleaned[j]) ** 2) ** (1 / 2)
            if (dist < 50):
                s = False
                break
        if (s == True):
            imaskCleaned = np.append(imaskCleaned, imask[i])
            jmaskCleaned = np.append(jmaskCleaned, jmask[i])
    return imaskCleaned, jmaskCleaned


def getMergedMask(msk1, msk2):
    msk1 = cv2.Canny(np.uint8(msk1) * 255, 100, 150)
    msk2 = cv2.Canny(np.uint8(msk2) * 255, 100, 150)
    intersection = np.uint8((msk1 / 2 + msk2 / 2)) // 255
    i, j = np.where(intersection == 1)
    i, j = cleanList(i, j)
    p1 = (i[0], j[0])
    p2 = (i[1], j[1])
    l = Line(p1, p2)
    mask = np.zeros(msk1.shape)
    for i in range(mask.shape[0]):
        mask[i, 0:int(l.get(i))] = 1
    # cv2.imwrite('test.jpg', (mask1 + mask2 + mask) * 80)
    return mask


def merge(img1, img2, mask):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    levels = 4
    # gaussian filter size
    k = 15
    height, width, shape = img1.shape
    gPyramid1, gPyramid2, lPyramid1, lPyramid2 = [], [], [], []
    for i in range(levels):
        img1g = cv2.GaussianBlur(img1.copy(), (k, k), 0)
        img2g = cv2.GaussianBlur(img2.copy(), (k, k), 0)
        img1l = img1 - img1g
        img2l = img2 - img2g
        gPyramid1.append(img1g)
        gPyramid2.append(img2g)
        lPyramid1.append(img1l)
        lPyramid2.append(img2l)
        img1 = cv2.resize(img1g.copy(), (int(img1g.shape[1] / 2), int(img1g.shape[0] / 2)))
        img2 = cv2.resize(img2g.copy(), (int(img2g.shape[1] / 2), int(img2g.shape[0] / 2)))
    Imskg = cv2.resize(mask.copy(), (int(width / 2 ** (levels - 1)), int(height / 2 ** (levels - 1))))
    Imskg = cv2.GaussianBlur(Imskg, (k, k), 0)[:, :, np.newaxis]
    Ians = Imskg * gPyramid1[-1] + (1. - Imskg) * gPyramid2[-1]
    for i in range(levels):
        Ians = cv2.resize(Ians, (Ians.shape[1] * 2, Ians.shape[0] * 2))
        Imskg = cv2.resize(mask.copy(), (int(width / 2 ** (levels - i - 1)), int(height / 2 ** (levels - i - 1))))
        Imskg = cv2.GaussianBlur(Imskg, (k, k), 0)[:, :, np.newaxis]
        laplacian = Imskg * lPyramid1[levels - i - 1] + (1. - Imskg) * lPyramid2[levels - i - 1]
        Ians = cv2.resize(Ians, (laplacian.shape[1], laplacian.shape[0]))
        Ians = Ians + laplacian
    return Ians


k = 1
padW, padH = 2000 // k, 1000 // k
I1 = cv2.imread('frames/90.jpg')
I2 = cv2.imread('frames/270.jpg')
I3 = cv2.imread('frames/450.jpg')
I4 = cv2.imread('frames/630.jpg')
I5 = cv2.imread('frames/810.jpg')
hei, wid, chan = I1.shape
I1 = cv2.resize(I1, (wid // k, hei // k))
I2 = cv2.resize(I2, (wid // k, hei // k))
I3 = cv2.resize(I3, (wid // k, hei // k))
I4 = cv2.resize(I4, (wid // k, hei // k))
I5 = cv2.resize(I5, (wid // k, hei // k))
I1p = pad(I1, padW, padH)
I2p = pad(I2, padW, padH)
I3p = pad(I3, padW, padH)
I4p = pad(I4, padW, padH)
I5p = pad(I5, padW, padH)
heip, widp, chanp = I1p.shape

H12 = warp(I1p, I2p)
H23 = warp(I2p, I3p)
H43 = warp(I4p, I3p)
H54 = warp(I5p, I4p)
H13 = np.matmul(H12, H23)
H53 = np.matmul(H54, H43)

I1p = cv2.warpPerspective(I1p, H13, (widp, heip))
I2p = cv2.warpPerspective(I2p, H23, (widp, heip))
I4p = cv2.warpPerspective(I4p, H43, (widp, heip))
I5p = cv2.warpPerspective(I5p, H53, (widp, heip))

IMerged = np.zeros((heip, widp, 3))
mask1 = getMask(I1p)
mask2 = getMask(I2p)
mask3 = getMask(I3p)
mask4 = getMask(I4p)
mask5 = getMask(I5p)

mask12 = getMergedMask(mask1, mask2)
mask23 = getMergedMask(mask2, mask3)
mask34 = getMergedMask(mask3, mask4)
mask45 = getMergedMask(mask4, mask5)
res = merge(I1p, I2p, mask12)
res = merge(res, I3p, mask23)
res = merge(res, I4p, mask34)
res = merge(res, I5p, mask45)
cv2.imwrite('res04-key-frames-panorama.jpg', res)
