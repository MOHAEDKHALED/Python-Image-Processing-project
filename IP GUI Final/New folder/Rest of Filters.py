import numpy as np

def median_filter_python(im, windowSize):
    r, c, ch = im.shape
    im = np.pad(im, [(windowSize - 1) // 2, (windowSize - 1) // 2], mode='edge')
    new_im = np.zeros((r, c, ch), dtype=np.uint8)
    
    for k in range(ch):
        for i in range(r - 2):
            for j in range(c - 2):
                window = im[i:i+windowSize, j:j+windowSize, k]
                new_im[i+1, j+1, k] = np.median(window)
    
    return new_im


def min_filter_python(im, windowSize):
    r, c, ch = im.shape
    im = np.pad(im, [(windowSize - 1) // 2, (windowSize - 1) // 2], mode='edge')
    new_im = np.zeros((r, c, ch), dtype=np.uint8)
    
    for k in range(ch):
        for i in range(r - 2):
            for j in range(c - 2):
                window = im[i:i+windowSize, j:j+windowSize, k]
                new_im[i+1, j+1, k] = np.min(window)
    
    return new_im

import numpy as np
from scipy.ndimage import maximum_filter

def max_filter_python(im, windowSize):
    r, c, ch = im.shape
    im = np.pad(im, [(windowSize - 1) // 2, (windowSize - 1) // 2], mode='edge')
    new_im = np.zeros((r, c, ch), dtype=np.uint8)
    
    for k in range(ch):
        for i in range(r - 2):
            for j in range(c - 2):
                window = im[i:i+windowSize, j:j+windowSize, k]
                new_im[i+1, j+1, k] = np.max(window)
    
    return new_im

def salt_and_pepper():
    
    #####################read image##########################################
    old=cv.imread(r'C:\Users\GRAPHICS\Desktop\Image Processing Project\images.jpg')
    N=int(input("enter the filter level :"))
    r1,c1,ch1=old.shape

    ########################adding noise####################################
    old_with_nois=random_noise(old, mode='s&p',amount=0.01)
    old_with_nois = np.array(255*old_with_nois, dtype = 'uint8')



    #################create the new image with orginal Dimensions##########################################

    new = np.zeros((r1,c1,ch1), np.uint8)

    ##################the old image given Border ###################################

    old_with_nois=cv.copyMakeBorder(old_with_nois, N, N, N,N, cv.BORDER_REFLECT)
    r,c,ch=old.shape



    for chh in range(0,ch):
        for i in range(N,r-N):
            for j in range(N,c-N):
                temp=old_with_nois[i-N:i+N+1,j-N:j+N+1,chh]
                new[i-N,j-N,chh]=np.median(temp)
                

    cv.imshow("old",old_with_nois)
    cv.imshow('new',new)

def watermark():
    old=cv.imread(r'C:\Users\GRAPHICS\Desktop\Image Processing Project\images.jpg')
    logo=cv.imread(r'C:\Users\GRAPHICS\Desktop\Image Processing Project\images.jpg')
    r,c,ch=old.shape

    #################create the new image ##########################################
    new=np.zeros_like(old)
    

    #######################loop in old image and make the operations#########################33
    for chh in range(0,ch):
        for i in range(0,r):
            for j in range(0,c):
                pixel=old[i,j,chh]  
                mask_pixel=pixel & 240  ##for make the low sintfic zeros by and with 240 =>1111 0000
                logo_pixel=logo[i,j,chh]
                logo_shift=logo_pixel>>4  ##for make the high sintfic zeros by shift 4 times right
                new_pexil=mask_pixel | logo_shift  ##or the result of previse operations
                new[i,j,chh]=new_pexil


    ###################show################################
    cv.imshow("old",old)
    cv.imshow('new',new)
    cv.waitKey(0)