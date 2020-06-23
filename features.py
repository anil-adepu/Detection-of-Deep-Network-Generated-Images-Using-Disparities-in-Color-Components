import numpy as np
import random
import cv2
from scipy import ndimage
from itertools import chain

def showIm(desc,img):
    cv2.imshow(desc,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hpass(img):
    f1 = [[1],[-1]]
    f2 = [[1,-1]]
    out = []
    out1 = ndimage.convolve(img, f1, mode='constant', cval=0.0)
    out2 = ndimage.convolve(img, f2, mode='constant', cval=0.0)
    out.append(out1)
    out.append(out2)
    return out


def rgb(img):
    red = img[:,:,2]
    green = img[:,:,1]
    blue = img[:,:,0]
    residualRGB = [hpass(red), hpass(green), hpass(blue)]
    binarizeRGBf1 = [binz(residualRGB[0][0]), binz(residualRGB[1][0]), binz(residualRGB[2][0])]  ### out of hpass filter1 at index 0
    binarizeRGBf2 = [binz(residualRGB[0][1]), binz(residualRGB[1][1]), binz(residualRGB[2][1])]  ### out of hpass filter2 at index 1

    assmbldRGBf1 = binarizeRGBf1[0]+2*binarizeRGBf1[1]+4*binarizeRGBf1[2]   ### acc. to  R̂ RGB = R̂ R · 1 + R̂ G · 2  + R̂ B · 4.
    assmbldRGBf2 = binarizeRGBf2[0]+2*binarizeRGBf2[1]+4*binarizeRGBf2[2]
    return assmbldRGBf1,assmbldRGBf2


def HS_Cb_Cr(hsv,ycbcr):
    hue = hsv[:,:,0]    ### hue
    sat = hsv[:,:,1]  ### sat
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    residualHS = [hpass(hue), hpass(sat), hpass(Cb), hpass(Cr)]
    thrH = [thr(residualHS[0][0]), thr(residualHS[0][1])]        ### thresholded of filtr1, filtr2 of remaining comp. at index 0,1
    thrS = [thr(residualHS[1][0]), thr(residualHS[1][1])]
    thrCb = [thr(residualHS[2][0]), thr(residualHS[2][1])]
    thrCr = [thr(residualHS[3][0]), thr(residualHS[3][1])]

    return thrH, thrS, thrCb, thrCr


def binz(img):
    out = []
    for i in img:
        a = []
        for j in i:
            if j>0:
                a.append(1)
            else:
                a.append(0)
        out.append(a)
    #out = [[1 if i>0 else 0 for i in j] for j in img]
    return np.uint8(out)


def thr(img):
    out = []
    th = 2
    for i in img:
        a = []
        for j in i:
            if j>=th:
                a.append(th+2)
            elif j<=-th:
                a.append(-th+2)
            else:
                a.append(j)
        out.append(a)
    return np.uint8(out)


def coOccurence(mat, n):   ### needs some optimization
    d = 3
    x = [0,1]

    right = [0 for i in range(pow(n,d))]#{n*n*i+n*j+k:0 for k in range(n) for j in range(i,n) for i in range(n)}#  ### n=8 for rgb, 5 for the rest
    down = [0 for i in range(pow(n,d))]#{n*n*i+n*j+k:0 for k in range(n) for j in range(i,n) for i in range(n)}#


    for i in range(len(mat)):
        for j in range(len(mat[0])):
            a = 0
            b = 0
            if j<len(mat[0])-d:
                a+=mat[i][j]*n*n + mat[i][j+1]*n + mat[i][j+2]
            if i<len(mat)-d:
                b+=mat[i][j]*n*n + mat[i+1][j]*n + mat[i+2][j]
            right[a]+=1
            down[b]+=1
    r = []
    dw = []
    for i in range(n):
        for j in range(i,n):
            for k in range(n):
                r.append(right[n*n*i+n*j+k])
                dw.append(down[n*n*i+n*j+k])
            '''if a in right:
                right[a]+=1
            else:
                right[a] = 1

            if b in down:
                down[b]+=1
            else:
                down[b] = 1'''

    #r = np.array(list(right.values()))
    #dw = np.array(list(down.values()))
    return [np.array(r)/sum(r), np.array(dw)/sum(dw) ]


def get_features(image):

    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imgYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


    resdH, resdS , resdCb, resdCr= HS_Cb_Cr(imgHSV, imgYCrCb)  ### final residuals after thresholding

    assmbldRGBf1, assmbldRGBf2 = rgb(image)

    cocrRGBf1 = coOccurence(assmbldRGBf1,8)
    cocrRGBf2 = coOccurence(assmbldRGBf2,8)

    cocrRGB = np.mean([cocrRGBf1[0], cocrRGBf1[1], cocrRGBf2[0], cocrRGBf2[1]], axis=0)

    cocrH1 = coOccurence(resdH[0], 5)
    cocrH2 = coOccurence(resdH[1],5)
    cocrH = np.mean([cocrH1[0],cocrH1[1],cocrH2[0],cocrH2[1]],axis=0)

    cocrS1 = coOccurence(resdS[0], 5)
    cocrS2 = coOccurence(resdS[1],5)
    cocrS = np.mean([cocrS1[0],cocrS1[1],cocrS2[0],cocrS2[1]],axis=0)

    cocrCb1 = coOccurence(resdCb[0], 5)
    cocrCb2 = coOccurence(resdCb[1],5)
    cocrCb = np.mean([cocrCb1[0],cocrCb1[1],cocrCb2[0],cocrCb2[1]],axis=0)

    cocrCr1 = coOccurence(resdCr[0], 5)
    cocrCr2 = coOccurence(resdCr[1],5)
    cocrCr = np.mean([cocrCr1[0],cocrCr1[1],cocrCr2[0],cocrCr2[1]],axis=0)
    #print(cocrCr1[0][0],cocrCr1[1][0],cocrCr2[0][0],cocrCr2[1][0],cocrCr[0])

    features = list(chain(cocrRGB, cocrH, cocrS, cocrCb, cocrCr))
    return features

    
