import cv2
import numpy as np


def read():
    f = open('Homographies.txt', 'r')
    Hs = []
    for i in range(900):
        arr = list(map(float, f.readline().split(' ')[0:9]))
        H = [[arr[0], arr[1], arr[2]],
             [arr[3], arr[4], arr[5]],
             [arr[6], arr[7], arr[8]]]
        print(H)
        Hs.append(np.linalg.inv(H))
    return Hs


Ip = cv2.imread('res06-background-panorama.jpg')
hei, wid = 1080, 1920
heip, widp, chanp = Ip.shape
Hs = read()
padW, padH = 2500, 1000
out = cv2.VideoWriter("res07-background-video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (wid, hei))
print(hei,wid)
for i in range(0,900):
    print(i)
    Iw=cv2.warpPerspective(Ip.copy(),Hs[i],(widp, heip))
    Iw=Iw[padH:(padH+hei),padW:(padW+wid),:]
    out.write(Iw)
out.release()

