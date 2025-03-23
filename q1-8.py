import cv2
import numpy as np


def read():
    f = open('Homographies.txt', 'r')
    Hs = np.zeros((900, 9))
    for i in range(900):
        arr = list(map(float, f.readline().split(' ')[0:9]))
        Hs[i, :] = np.array([arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8]])
    return Hs


def inv(H):
    return np.linalg.inv(np.array([[H[0], H[1], H[2]],
                                   [H[3], H[4], H[5]],
                                   [H[6], H[7], H[8]]]))

Hs = read()
Hs = cv2.blur(Hs, (1, 41))
hei, wid = 1080, 1920
padW, padH = 2500, 1000
out = cv2.VideoWriter("res10-video-shakeless.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (wid, hei))
for i in range(900):
    print(i)
    I = cv2.imread('warped frames/' + str(i + 1) + '.jpg')
    H=inv(Hs[i,:])
    Iw = cv2.warpPerspective(I,H , (I.shape[1], I.shape[0]))
    Iw = Iw[padH:(padH + hei), padW:(padW + wid), :]
    out.write(Iw)
out.release()
