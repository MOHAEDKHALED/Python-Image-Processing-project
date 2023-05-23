import os
import math
import cv2 as cv
import numpy as np
import pathlib
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt



if not os.path.exists('tmp'):
    os.makedirs('tmp')



def delete_img():
        os.remove('tmp/original_img.png')

def delete_img1():
        os.remove('tmp/original_img1.png')


######################functions###################################
def browse_img():
        try:
            file = filedialog.askopenfilename(title = "Load Image", filetypes=[('Images', ['*jpeg','*png','*jpg'])]) 
            file = Image.open(file)
            file.save(pathlib.Path("tmp/original_img.png"))
        except Exception as e:
            messagebox.showerror("An error occured !", e)

def browse_img1():
        try:
            file = filedialog.askopenfilename(title = "Load Image", filetypes=[('Images', ['*jpeg','*png','*jpg'])]) 
            file = Image.open(file)
            file.save(pathlib.Path("tmp/original_img1.png"))
        except Exception as e:
            messagebox.showerror("An error occured !", e)


def DM_0L():
    im=cv.imread('tmp/original_img.png')
    fact=int(simpledialog.askstring(title="Window",
                                  prompt="Enter the factor :"))
    r, c, ch = im.shape
    New_r = r * fact
    New_c = c * fact
    New_im = np.zeros((New_r, New_c, ch), dtype=np.uint8)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                New_im[i*fact:i*fact+fact, j*fact:j*fact+fact, k] = im[i, j, k]

    plt.imshow(im)
    plt.title('Original')
    plt.show()

    plt.imshow(New_im)
    plt.title('Resized')
    plt.show()

    return New_im


def RM_1_order():
    old_image=cv.imread('tmp/original_img.png')
    fact_r=int(simpledialog.askstring(title="Window",
                                  prompt="Enter the factor r :"))
    fact_c=int(simpledialog.askstring(title="Window",
                                  prompt="Enter the factor c :"))
    r, c, ch = old_image.shape
    new_r = r * fact_r
    new_c = c * fact_c
    r_ratio = r / new_r
    c_ratio = c / new_c
    output_img = np.zeros((new_r, new_c, ch), dtype=np.uint8)

    for k in range(ch):
        for new_x in range(new_r):
            old_x = new_x * r_ratio
            x1 = int(old_x)
            x1 = max(0, min(x1, r - 1))
            x2 = x1 + 1
            x2 = max(0, min(x2, r - 1))
            x_fraction = abs(old_x - x1)
            for new_y in range(new_c):
                old_y = new_y * c_ratio
                y1 = int(old_y)
                y1 = max(0, min(y1, c - 1))
                y2 = y1 + 1
                y2 = max(0, min(y2, c - 1))
                p1 = old_image[x1, y1, k]
                p2 = old_image[x2, y1, k]
                p3 = old_image[x1, y2, k]
                p4 = old_image[x2, y2, k]
                y_fraction = abs(old_y - y1)
                z1 = p1 * (1 - x_fraction) + p2 * x_fraction
                z2 = p3 * (1 - x_fraction) + p4 * x_fraction
                new_pixel = z1 * (1 - y_fraction) + z2 * y_fraction
                output_img[new_x, new_y, k] = int(new_pixel)

    plt.imshow(old_image)
    plt.title('Original')
    plt.show()

    plt.imshow(output_img)
    plt.title('Resized')
    plt.show()

    return output_img

def RM_0_order():
    old_image=cv.imread('tmp/original_img.png')
    fact_r=int(simpledialog.askstring(title="Window",
                                  prompt="enter the factor r : "))
    fact_c=int(simpledialog.askstring(title="Window",
                                  prompt="enter the factor c : "))
    r, c, ch = old_image.shape
    new_r = r * fact_r
    new_c = c * fact_c
    r_ratio = r / new_r
    c_ratio = c / new_c
    output_img = np.zeros((new_r, new_c, ch), dtype=np.uint8)

    for k in range(ch):
        for new_x in range(new_r):
            old_x = new_x * r_ratio
            old_x = int(old_x)
            if old_x == 0:
                old_x = 1
            for new_y in range(new_c):
                old_y = new_y * c_ratio
                old_y = int(old_y)
                if old_y == 0:
                    old_y = 1
                output_img[new_x, new_y, k] = old_image[old_x, old_y, k]

    plt.imshow(old_image)
    plt.title('Original')
    plt.show()

    plt.imshow(output_img)
    plt.title('Resized')
    plt.show()

    return output_img


def DM_1():
    ####################################Get the image and the factor############################################################
    old_image=cv.imread('tmp/original_img.png')
    fact=int(simpledialog.askstring(title="Window",
                                  prompt="enter the factor : "))            
    ###########################################Set the attributes##############################################################################
    r, c, ch = old_image.shape
    my_r = r * fact
    my_c = c * fact
    uploadImage = np.zeros((my_r, my_c, ch), dtype=np.uint8)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                uploadImage[i*fact:i*fact+fact, j*fact:j*fact+fact, k] = old_image[i, j, k]

    for k in range(ch):
        for i in range(my_r):
            for j in range(my_c):
                if j < c:
                    RightExp = uploadImage[i, (j+1)*fact+1-fact, k]
                    LeftExp = uploadImage[i, j*fact+1-fact, k]
                    if RightExp <= LeftExp:
                        for y in range(1, fact):
                            exp = round(((LeftExp - RightExp) / fact) * y + RightExp)
                            uploadImage[i, ((j+1)*fact+1-fact)-y, k] = exp
                    elif LeftExp < RightExp:
                        for y in range(1, fact):
                            exp = round(((RightExp - LeftExp) / fact) * y + LeftExp)
                            uploadImage[i, (j*fact+1-fact)+y, k] = exp
                elif j > (my_c - fact) + 1:
                    uploadImage[i, j, k] = uploadImage[i, j-1, k]

    for k in range(ch):
        for i in range(my_r):
            for j in range(my_c):
                if i < r:
                    RightExp = uploadImage[(i+1)*fact+1-fact, j, k]
                    LeftExp = uploadImage[i*fact+1-fact, j, k]
                    if RightExp <= LeftExp:
                        for y in range(1, fact):
                            exp = round(((LeftExp - RightExp) / fact) * y + RightExp)
                            uploadImage[((i+1)*fact+1-fact)-y, j, k] = exp
                    elif LeftExp < RightExp:
                        for y in range(1, fact):
                            exp = round(((RightExp - LeftExp) / fact) * y + LeftExp)
                            uploadImage[(i*fact+1-fact)+y, j, k] = exp
                elif i > (my_r - fact) + 1:
                    uploadImage[i, j, k] = uploadImage[i-1, j, k]
    
    plt.imshow(old_image)
    plt.title('Original')
    plt.show()
    
    plt.imshow(uploadImage)
    plt.title('Resized')
    plt.show()

    return uploadImage

def gray_scale():
    im=cv.imread('tmp/original_img.png')
  ###################take element############################
    r, c, ch = im.shape
    if ch < 3:
        plt.imshow(im)
        plt.title('Original')
    else:
        gray = (0.3 * im[:, :, 0]) + (0.59 * im[:, :, 1]) + (0.11 * im[:, :, 2])
        gray = gray.astype('uint8')
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
    plt.show()

    return gray


def drawing_histo():
    image=cv.imread('tmp/original_img.png')
    ###########################set attrbute#########################################
    r, c, ch = image.shape
    histo = np.zeros((1, 256))
    colors = np.arange(256)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                histo[0, image[i, j, k]] += 1

    plt.plot(colors, histo[0])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

    return histo


def Contrast():
    image=cv.imread('tmp/original_img.png')
    ####################take low and high contrast################################
    new_min=int(simpledialog.askstring(title="Window",
                                  prompt="enter the min contrast :"))
    new_max=int(simpledialog.askstring(title="Window",
                                  prompt="enter the max contrast :"))
    ###########################set attrbute#########################################
    r, c, ch = image.shape
    output = np.zeros((r, c, ch), dtype=np.uint8)
    min_val = np.min(image)
    max_val = np.max(image)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                new_val = ((image[i, j, k] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
                if new_val > 255:
                    new_val = 255
                elif new_val < 0:
                    new_val = 0
                output[i, j, k] = new_val

    plt.imshow(output)
    plt.title('Contrast')
    plt.show()

    return output


def Brightness():
    image=cv.imread('tmp/original_img.png')
    ####################take low and high contrast################################
    offset=int(simpledialog.askstring(title="Window",
                                  prompt="enter the offset :"))
    ###########################set attrbute#########################################
    r, c, ch = image.shape
    bright = np.zeros((r, c, ch), dtype=np.uint8)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                new_val = image[i, j, k] + offset
                if new_val > 255:
                    new_val = 255
                elif new_val < 0:
                    new_val = 0
                bright[i, j, k] = new_val

    plt.imshow(bright)
    plt.title('Brightness')
    plt.show()

    return bright  

def power_law():
    ###########################start#######################
    image=cv.imread('tmp/original_img.png')
    ###########################set attrbute#########################################
    gamma=float(simpledialog.askstring(title="Window",
                                  prompt="enter the gamma :"))
    r, c, ch = image.shape
    image = image.astype(np.float64) / 255.0
    output = np.zeros((r, c, ch), dtype=np.float64)
    new_max = 255
    new_min = 0

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                new_val = image[i, j, k] ** gamma
                if new_val > 1.0 or new_val < 0.0:
                    new_val = (image[i, j, k] - np.min(image)) / (np.max(image) - np.min(image)) * (new_max - new_min) + new_min
                output[i, j, k] = new_val

    output = (output * 255).astype(np.uint8)

    plt.imshow(output)
    plt.title('Power Law Transformation')
    plt.show()

    return output

def histogram_equalization():
    im=cv.imread('tmp/original_img.png')
    output = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    r, c = output.shape[:2]
    hist = np.zeros(256)
    hist2 = np.zeros(256)
    colors = np.arange(256)
    
    for i in range(r):
        for j in range(c):
            hist[output[i, j]] += 1
    
    for i in range(r):
        for j in range(c):
            val = output[i, j]
            if 1 <= val <= 255:
                hist2[int(val)] = output[i, j] + 1
    
    run_sum = np.cumsum(hist2)
    
    for i in range(len(run_sum)):
        val = run_sum[i] / max(run_sum) * 255
        run_sum[i] = round(val)
    
    for i in range(r):
        for j in range(c):
            v = output[i, j]
            if 1 <= v <= 255:
                output[i, j] = run_sum[v]
    
    hist2 = np.zeros(256)
    
    for i in range(r):
        for j in range(c):
            hist2[output[i, j]] += 1

    plt.figure()
    plt.plot(colors, hist)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title("Histogram old")

    plt.figure()
    plt.plot(colors, hist2)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title("Histogram new")


    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(output, cmap="gray")
    plt.title("Histogram Equalization")

    plt.axis("off")
    plt.show()


def histogram_matching():
    output=cv.imread('tmp/original_img.png')
    reference=cv.imread('tmp/original_img1.png')
    r, c = output.shape[:2]
    hist_output, _ = np.histogram(output.flatten(), bins=256, range=[0, 256])
    hist_reference, _ = np.histogram(reference.flatten(), bins=256, range=[0, 256])
    
    cdf_output = hist_output.cumsum()
    cdf_reference = hist_reference.cumsum()
    cdf_output_normalized = cdf_output / float(cdf_output.max())
    cdf_reference_normalized = cdf_reference / float(cdf_reference.max())
    
    lut = np.interp(cdf_output_normalized, cdf_reference_normalized, np.arange(256))
    output_matched = lut[output]
    
    hist_matched, _ = np.histogram(output_matched.flatten(), bins=256, range=[0, 256])
    
    plt.figure()
    plt.plot(np.arange(256), hist_output)
    plt.title("Histogram - Output")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.figure()
    plt.plot(np.arange(256), hist_reference)
    plt.title("Histogram - Reference")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.figure()
    plt.plot(np.arange(256), hist_matched)
    plt.title("Histogram - Matched")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()
    return output_matched

def add():
    first_image=cv.imread('tmp/original_img.png')
    second_image=cv.imread('tmp/original_img1.png')
    r, c, ch = first_image.shape
    r2, c2, ch2 = second_image.shape
    upload_image = np.zeros((max(r, r2), max(c, c2), max(ch2, ch)), dtype=np.uint8)

    for k in range(max(ch2, ch)):
        for i in range(max(r2, r)):
            for j in range(max(c, c2)):
                first_val = 0
                sec_val = 0

                if i < r and j < c and k < ch:
                    first_val = first_image[i, j, k]

                if i < r2 and j < c2 and k < ch2:
                    sec_val = second_image[i, j, k]

                add_new_pix = first_val + sec_val
                if add_new_pix > 255:
                    add_new_pix = 255
                elif add_new_pix < 0:
                    add_new_pix = 0

                upload_image[i, j, k] = add_new_pix

    plt.imshow(first_image)
    plt.title('Original 1')
    plt.show()

    plt.imshow(second_image)
    plt.title('Original 2')
    plt.show()

    plt.imshow(upload_image)
    plt.title('Result')
    plt.show()

    return upload_image

def subtract():
    im1=cv.imread('tmp/original_img.png')
    im2=cv.imread('tmp/original_img1.png')

    ############################set attbuts#######################################
    im1 = cv.resize(im1, (im2.shape[1], im2.shape[0]))  # Resize im1 to match the size of im2
    r, c, ch = im1.shape
    subtracted = np.zeros((r, c, ch), dtype=np.uint8)
    new_max = 255
    new_min = 0

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                new_val = im1[i, j, k] - im2[i, j, k]
                if new_val > 255 or new_val < 0:
                    new_val = ((im1[i, j] - np.min(im1)) / (np.max(im1) - np.min(im1))) * (new_max - new_min) + new_min
                subtracted[i, j, k] = new_val

    subtracted = np.uint8(subtracted)
    plt.imshow(subtracted)
    plt.title('Subtracted')
    plt.show()

    return subtracted

def negative():
    image=cv.imread('tmp/original_img.png')
    r, c, ch = image.shape
    neg = np.zeros((r, c, ch), dtype=np.uint8)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                new_val = 255 - image[i, j, k]
                neg[i, j, k] = new_val

    plt.imshow(neg)
    plt.title('Negative Image')
    plt.show()

    return neg

def Quantization():
    
    image=cv.imread('tmp/original_img.png')
    k=int(simpledialog.askstring(title="Window",
                                  prompt="enter the number of bits per pixel (k):"))

    r, c, ch = image.shape
    quant = np.zeros((r, c, ch), dtype=np.uint8)
    gray_levels = 2 ** k
    gap = 256 / gray_levels
    colors = np.arange(gap, 256, gap)

    for k in range(ch):
        for i in range(r):
            for j in range(c):
                temp = image[i, j, k] / gap
                index = int(np.floor(temp))
                index = min(index, len(colors) - 1)  # Ensure index is within range
                quant[i, j, k] = colors[index]

    plt.imshow(quant)
    plt.title('Quantization')
    plt.show()

    return quant


#Gaussian Filter
def Smoothing_with_Weighted_Filter():    
    im = cv.imread('tmp/original_img.png')
    sigma = int(simpledialog.askstring(title="Window",
                                  prompt="Enter Filter Level :"))
    r, c, ch = im.shape
    if ch == 3:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im_gray = im

    N = int(np.floor(3.7 * sigma - 0.5))
    mask_size = 2 * N + 1
    x = np.zeros((mask_size,))
    x[0] = -int(np.floor(mask_size / 2))
    for k in range(1, mask_size):
        x[k] = x[k - 1] + 1

    mask = np.zeros((mask_size, mask_size))
    for i in range(mask_size):
        for j in range(mask_size):
            mask[i, j] = np.exp(-(x[i] ** 2 + x[j] ** 2) / (2 * sigma ** 2))

    mask /= np.sum(mask)
    bord = int(np.floor(mask_size / 2))
    im_padded = cv.copyMakeBorder(im_gray, bord, bord, bord, bord, cv.BORDER_REPLICATE)
    r, c = im_padded.shape
    output = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            r1 = max(0, i - bord)
            r2 = min(r - 1, i + bord)
            c1 = max(0, j - bord)
            c2 = min(c - 1, j + bord)
            matrix = im_padded[r1:r2+1, c1:c2+1]
            mean_value = np.mean(matrix)
            output[i, j] = mean_value

    output = output[bord:-bord, bord:-bord]
    output = output.astype(np.uint8)
    plt.imshow(output, cmap='gray')
    plt.title('Smoothed Image')
    plt.show()

    return output    


#Average Filter
def Smoothing_with_Mean_Filter():
    im = cv.imread('tmp/original_img.png')
    mask_size =int(simpledialog.askstring(title="Window",
                                  prompt="Enter the mask size :"))
    r, c, ch = im.shape
    if ch == 3:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im_gray = im

    bor = mask_size // 2
    im = np.pad(im_gray, ((bor, bor), (bor, bor)), mode='edge')
    output = np.zeros_like(im)

    for i in range(1, r + 1):
        for j in range(1, c + 1):
            r1 = max(1, i - bor)
            r2 = min(r + 2 * bor, i + bor + 1)
            c1 = max(1, j - bor)
            c2 = min(c + 2 * bor, j + bor + 1)
            matrix = im[r1:r2, c1:c2]
            meanvalue = np.mean(matrix)
            output[i, j] = meanvalue

    output = output.astype(np.uint8)
    plt.imshow(output, cmap='gray')
    plt.title('Smoothing with Mean')
    plt.show()

    return output

def Edge_detection():
    im = cv.imread('tmp/original_img.png')
    r, c, ch = im.shape
    if ch == 3:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im_gray = im

    new_max = 255
    new_min = 0
    mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    m, n = mask.shape
    board = tuple(np.floor((np.array(mask.shape) - 1) / 2).astype(int))
    padded_img = np.zeros((im_gray.shape[0] + 2 * board[0], im_gray.shape[1] + 2 * board[1]))
    padded_img[board[0]:-board[0], board[1]:-board[1]] = im_gray
    output = np.zeros_like(im_gray)

    for i in range(padded_img.shape[0] - m + 1):
        for j in range(padded_img.shape[1] - n + 1):
            new_val = np.sum(mask * padded_img[i:i+m, j:j+n])
            if new_val > 255 or new_val < 0:
                new_val = ((im_gray[i, j] - np.min(im_gray)) / (np.max(im_gray) - np.min(im_gray))) * (new_max - new_min) + new_min
            output[i, j] = new_val

    output = output.astype(np.uint8)
    plt.imshow(output, cmap='gray')
    plt.title('Edge Detection')
    plt.show()

    return output

def Sharpening():
    im = cv.imread('tmp/original_img.png')
    r, c, ch = im.shape
    if ch == 3:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im_gray = im

    mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    r, c = im_gray.shape
    mr, mc = mask.shape
    padded_img = np.zeros((r + 2 * (mr - 1), c + 2 * (mc - 1)))
    padded_img[mr-1:-mr+1, mc-1:-mc+1] = im_gray
    output = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            new_val = np.sum(mask * padded_img[i:i+mr, j:j+mc])
            if new_val < 0:
                new_val = 0
            output[i, j] = new_val

    output = output.astype(np.uint8)
    plt.imshow(output, cmap='gray')
    plt.title('Sharpened Image')
    plt.show()

    return output


def Unsharpe():
    im = cv.imread('tmp/original_img.png')
    r, c, ch = im.shape
    if ch == 3:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        im = im

    output = np.zeros_like(im)
    N = int(np.floor(3.7 * 0.4 - 0.5))
    mask_size = 2 * N + 1
    x = np.zeros(mask_size)
    x[0] = -int(np.floor(mask_size / 2))
    for k in range(1, mask_size):
        x[k] = x[k - 1] + 1

    mask = np.zeros((mask_size, mask_size))
    for i in range(mask_size):
        for j in range(mask_size):
            mask[i, j] = np.exp(-(x[i] ** 2 + x[j] ** 2) / (2 * 0.04 ** 2))

    bord = int(np.floor(mask_size / 2))
    im = np.pad(im, ((bord, bord), (bord, bord)), mode='edge')
    r, c = im.shape
    for i in range(1, r - bord):
        for j in range(1, c - bord):
            r1 = max(1, i - bord)
            r2 = min(r, i + bord + 1)
            c1 = max(1, j - bord)
            c2 = min(c, j + bord + 1)
            matrix = im[r1:r2, c1:c2]
            meanvalue = np.mean(matrix)
            smoothed_im = meanvalue
            temp = im[i, j] - smoothed_im
            output[i, j] = temp + im[i, j]

    output = output.astype(np.uint8)
    plt.imshow(output, cmap='gray')
    plt.title('Unsharpened Image')
    plt.show()

    return output


def low_pass_ideal():
    im = cv.imread('tmp/original_img.png')
    #####################input from user##################################
    D0 = int(simpledialog.askstring(title="Window",
                                  prompt="enter the radius :"))  
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(im))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    m, n = im.shape
    filter = np.zeros((m, n))
    D = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m/2)**2 + (v - n/2)**2)
            if D[u, v] <= D0:
                filter[u, v] = 1
            else:
                filter[u, v] = 0

    fshift_filtered = fshift * filter
    f_ishift = np.fft.ifftshift(fshift_filtered)
    im_filtered = np.abs(np.fft.ifft2(f_ishift))

    IFT = np.uint8(im_filtered)
    plt.imshow(IFT, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

    return IFT

def low_pass_butterworth():
    im = cv.imread('tmp/original_img.png')
    
    #####################input from user##################################
    D0 = int(simpledialog.askstring(title="Window",
                                  prompt="enter the radius :"))  
    N= int(simpledialog.askstring(title="Window",
                                  prompt="enter the n :")) 

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    m, n = im.shape
    fshift = np.fft.fftshift(np.fft.fft2(im))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    D = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m/2)**2 + (v - n/2)**2)

    filter = 1 / (1 + np.power(D / D0, 2 * N))
    fshift_filtered = fshift * filter
    f_ishift = np.fft.ifftshift(fshift_filtered)
    im_filtered = np.abs(np.fft.ifft2(f_ishift))

    IFT = np.uint8(im_filtered)
    plt.imshow(IFT, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

    return IFT

def low_pass_gaussian():
    im = cv.imread('tmp/original_img.png')
  
        #####################input from user##################################
    D0 = int(simpledialog.askstring(title="Window",
                                  prompt="enter the radius :"))  

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    m, n = im.shape
    fshift = np.fft.fftshift(np.fft.fft2(im))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    D = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m/2)**2 + (v - n/2)**2)

    filter = np.exp(-np.power(D, 2) / (2 * np.power(D0, 2)))
    fshift_filtered = fshift * filter
    f_ishift = np.fft.ifftshift(fshift_filtered)
    im_filtered = np.abs(np.fft.ifft2(f_ishift))

    IFT = np.uint8(im_filtered)
    plt.imshow(IFT, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

    return IFT

def high_pass_ideal():
    im = cv.imread('tmp/original_img.png')  
    #####################input from user################################## 
    D0 = int(simpledialog.askstring(title="Window",
                                  prompt="Enter the radius :"))
    ############################make mask##################################
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(im))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    m, n = im.shape
    filter = np.zeros((m, n))
    D = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m/2)**2 + (v - n/2)**2)
            if D[u, v] >= D0:
                filter[u, v] = 1
            else:
                filter[u, v] = 0

    fshift_filtered = fshift * filter
    f_ishift = np.fft.ifftshift(fshift_filtered)
    im_filtered = np.abs(np.fft.ifft2(f_ishift))

    IFT = np.uint8(im_filtered)
    plt.imshow(IFT, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

    return IFT

def high_pass_butterworth():
    im = cv.imread('tmp/original_img.png')
    #####################input from user##################################
    D0 = int(simpledialog.askstring(title="Window",
                                  prompt="enter the radius :"))  
    N= int(simpledialog.askstring(title="Window",
                                  prompt="enter the n :")) 

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    m, n = im.shape
    fshift = np.fft.fftshift(np.fft.fft2(im))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    D = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m/2)**2 + (v - n/2)**2)

    filter = 1 / (1 + (D0 / D)**(2*N))
    fshift_filtered = fshift * filter
    f_ishift = np.fft.ifftshift(fshift_filtered)
    im_filtered = np.abs(np.fft.ifft2(f_ishift))

    IFT = np.uint8(im_filtered)
    plt.imshow(IFT, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

    return IFT

def high_pass_gaussian ():
    im = cv.imread('tmp/original_img.png')

    #####################input from user##################################
    D0 = int(simpledialog.askstring(title="Window",
                                  prompt="enter the radius :"))  

    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    m, n = im.shape
    fshift = np.fft.fftshift(np.fft.fft2(im))
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    D = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            D[u, v] = np.sqrt((u - m/2)**2 + (v - n/2)**2)

    filter = 1 - np.exp(-np.power(D, 2) / (2 * np.power(D0, 2)))
    fshift_filtered = fshift * filter
    f_ishift = np.fft.ifftshift(fshift_filtered)
    im_filtered = np.abs(np.fft.ifft2(f_ishift))

    IFT = np.uint8(im_filtered)
    plt.imshow(IFT, cmap='gray')
    plt.title('Filtered Image')
    plt.show()

    return IFT



###########################gui#################################


window = Tk()
window.geometry("1015x700")  ##column x row
window.config(bg='#7c7a78')
window.title("Image Processing GUI")
x = 450  # Specify the desired x-coordinate
y = 100  # Specify the desired y-coordinate
window.geometry(f"+{x}+{y}")


l1 = Button(window, text = "Choose Image 1")
l1.config(command=browse_img)
l2 = Button(window, text = "Reset Image 1")
l2.config(command=delete_img)
l3 = Button(window, text = "Choose Image 2")
l3.config(command=browse_img1)
l4 = Button(window, text = "Reset Image 2")
l4.config(command=delete_img1)


B0=Button(text="RM - 0")
B0.config(command=RM_0_order)
B1=Button(text="DM - 1")
B1.config(command=DM_1)
B1A=Button(text="RM - 1")
B1A.config(command=RM_1_order)
B1A1=Button(text="DM - 0")
B1A1.config(command=DM_0L)
B2=Button(text="Gray Scale")
B2.config(command=gray_scale)
B3=Button(text="Draw Histogram")
B3.config(command=drawing_histo)
B4=Button(text="Adjust Contrast")
B4.config(command=Contrast)
B5=Button(text="Adjust Brightness")
B5.config(command=Brightness)
B6=Button(text="Power Law")
B6.config(command=power_law)
B7=Button(text="Histogram Equalzation")
B7.config(command=histogram_equalization)
B8=Button(text="Histogram Matching")
B8.config(command=histogram_matching)
B9=Button(text="Add Images")
B9.config(command=add)
B10=Button(text="Subtract Images")
B10.config(command=subtract)
B11=Button(text="Image Negative")
B11.config(command=negative)
B12=Button(text="Quantization")
B12.config(command=Quantization)
B13=Button(text="Gaussian Filter")
B13.config(command=Smoothing_with_Weighted_Filter)
B14=Button(text="Average Filter")
B14.config(command=Smoothing_with_Mean_Filter)
B18=Button(text="Edge detection")
B18.config(command=Edge_detection)
B19=Button(text="Sharpening")
B19.config(command=Sharpening)
B20=Button(text="Unsharpen")
B20.config(command=Unsharpe)
B21=Button(text="Low Pass Ideal")
B21.config(command=low_pass_ideal)
B22=Button(text="Low Pass Butterworth")
B22.config(command=low_pass_butterworth)
B23=Button(text="Low Pass Gaussian")
B23.config(command=low_pass_gaussian)
B24=Button(text="High Pass Ideal")
B24.config(command=high_pass_ideal)
B25=Button(text="High Pass Butterworth")
B25.config(command=high_pass_butterworth)
B26=Button(text="High Pass Gaussian")
B26.config(command=high_pass_gaussian)


l1.config(font=('Calibri',18),bg='#2d4989',width=19,height=2,fg='#ffffff')
l2.config(font=('Calibri',18),bg='#2d4989',width=24,height=2,fg='#ffffff')
l3.config(font=('Calibri',18),bg='#2d4989',width=19,height=2,fg='#ffffff') 
l4.config(font=('Calibri',18),bg='#2d4989',width=19,height=2,fg='#ffffff') 


B0.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B1.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B1A.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B1A1.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B2.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B3.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B4.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B5.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B6.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B7.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B8.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B9.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B10.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B11.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B12.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B13.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B14.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B18.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B19.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B20.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B21.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B22.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B23.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B24.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=24,height=2)
B25.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)
B26.config(bg='#b05615',fg='#2b3232',activebackground='#ffffff',activeforeground='#000000',font=('Aerial',15,'bold'),width=19,height=2)







l1.grid(row = 0, column = 0,sticky = W)
l2.grid(row = 0, column = 2,sticky = W)
l3.grid(row = 0, column = 1, sticky = W)
l4.grid(row = 0, column = 3,sticky = W)


B0.grid(row = 1, column = 0, sticky = W)
B1.grid(row = 2, column = 0, sticky = W)
B1A.grid(row = 3, column = 0, sticky = W)
B1A1.grid(row = 4, column = 0, sticky = W)
B11.grid(row = 5, column = 0, sticky = W)


B2.grid(row = 1, column = 1, sticky = W)
B3.grid(row = 2, column = 1, sticky = W)
B4.grid(row = 3, column = 1, sticky = W)
B5.grid(row = 4, column = 1, sticky = W)
B6.grid(row = 5, column = 1, sticky = W)
B7.grid(row = 6, column = 1, sticky = W)
B8.grid(row = 7, column = 1, sticky = W)
B9.grid(row = 8, column = 1, sticky = W)
B10.grid(row = 9, column = 1, sticky = W)




B12.grid(row = 1, column = 2, sticky = W)
B13.grid(row = 2, column = 2, sticky = W)
B14.grid(row = 3, column = 2, sticky = W)
B21.grid(row = 4, column = 2, sticky = W)
B22.grid(row = 5, column = 2, sticky = W)
B23.grid(row = 6, column = 2, sticky = W)
B24.grid(row = 7, column = 2, sticky = W)


B18.grid(row = 1, column = 3, sticky = W)
B19.grid(row = 2, column = 3, sticky = W)
B20.grid(row = 3, column = 3, sticky = W)
B25.grid(row = 4, column = 3, sticky = W)
B26.grid(row = 5, column = 3, sticky = W)



window.mainloop()
delete_img()
delete_img1()