import numpy as np
import cv2

I = cv2.imread('warped frames/1.jpg')
hei, wid, chan = I.shape
n = 900
k = 308
h=hei//k
for i in range(n):
    print(i)
    I=cv2.imread('warped frames/'+str(i+1)+'.jpg')
    for l in range(k):
        cv2.imwrite('blocks/'+str(i+1)+'-'+str(l+1)+'.jpg',I[l*h:(l+1)*h,:,:])
for l in range(k):
    print(l)
    imagesList = np.zeros((n, h, wid, 3))
    for i in range(n):
        I= cv2.imread('blocks/' + str(i + 1) + '-' + str(l + 1) + '.jpg')
        imagesList[i,:,:,:]=I
    res = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, imagesList)
    res[np.isnan(res)]=0
    cv2.imwrite('merge/' + str(l + 1) + '.jpg', res)

res=cv2.imread('merge/1.jpg')
for l in range(1,k):
    I= cv2.imread('merge/'+str(l+1)+'.jpg')
    print(str(l),I.shape)
    res=np.vstack((res,I))
cv2.imwrite('res06-background-panorama.jpg',res)

