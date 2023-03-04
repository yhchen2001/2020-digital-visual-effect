from pylab import *
import copy
import numpy as np
import pandas as pd
from scipy.ndimage import filters
import scipy
import matplotlib.pylab as pyp
from PIL import Image
import cv2
import os
# %matplotlib inline
import os.path as osp
import math
import argparse
import sys

RESIZE_FACTOR = 3
FOCAL_LENGTH = int(sys.argv[1])
RANSAC_THRES_DIS = 50
RANSAC_K = 1000

CROP_THRES = 30000
CROP_FACTOR = 50
BLEND_THRES = 75

RESIZE = False
PRINT_IMG = False
REVERSE = False

file_name = str(FOCAL_LENGTH)
print(file_name)

dir = '../data'
try:
    os.mkdir(dir + '/result', 0o700)
except:
    print("dir already exist")
print("hi")

''' 這組搭現在的feature detection是好的
resize factor = 5
focal length = 500
ransak_k = 1000
Ransac_thres_dis = 5
'''

images = []
images_rgb = []


for file in np.sort(os.listdir(dir)):
    if osp.splitext(file)[1] in ['.png', '.jpg']:
        im = cv2.imread(osp.join(dir, file))
        if RESIZE == True:
            h,w,c = im.shape
            im = cv2.resize(im, (w//RESIZE_FACTOR, h//RESIZE_FACTOR))
        images_rgb += [im]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        images += [im]
if REVERSE:
    images.reverse()
    images_rgb.reverse()
images_rgb_copy = copy.deepcopy(images_rgb)

HEIGHT, WIDTH = images_rgb[0].shape[:2]

def find_R(img,kernel_size=5,sigma=3,k=0.04):
    kernel = (kernel_size,kernel_size)
    img_blur = cv2.GaussianBlur(img, kernel, sigma)
    Iy, Ix = np.gradient(img_blur)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    Sx2 = cv2.GaussianBlur(Ix2, kernel, sigma)
    Sy2 = cv2.GaussianBlur(Iy2, kernel, sigma)
    Sxy = cv2.GaussianBlur(Ixy, kernel, sigma)
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM-k*(traceM**2)
    #print(np.max(R))
    return Ix,Iy,R
print(77)
#do non-maximum suppression
def supression(R,image,edge=10,window=6,thres=0.01):
    maxR = filters.maximum_filter(R, (window, window))
    R = R * (R == maxR)
    R[0:edge,:]=0
    R[R.shape[0]-edge:R.shape[0],:]=0
    R[:,0:edge]=0
    R[:,R.shape[1]-edge:R.shape[1]]=0
    new_imgs = np.zeros(image.shape)
    new_imgs = image.copy()
    new_imgs[R >= np.max(R) * thres] = 0

    cordin = np.where(R >= np.max(R) * thres)
    return cordin[0],cordin[1]
print(92)

# Haris descriptor
def Haris_descriptors(img,x,y, win=3):
    descriptors = []
    for i in range(len(y)):
        vector = img[x[i]-win:x[i]+win+1, y[i]-win:y[i]+win+1].flatten()
        # normalize the descriptor
        vector = (vector - vector.mean()) / vector.std()
        descriptors.append(vector)
    #print(descriptors)
    return np.array(descriptors)

#match descriptors
def match(des1,des2,thres=1,thres_max = 43):
    pairs = []
    total_max = []
    second_max = []
    for i in range(len(des1)):
        max_inner = thres-1
        maxj = 0
        second = 0
        for j in range(len(des2)):
            inner = np.inner(des1[i],des2[j])
            if(inner > max_inner):
                second = max_inner
                max_inner = inner
                maxj = j
            #print(i,maxj,max)
        
        if(max_inner > second+10 and max_inner > thres_max):
            pairs.append([i,maxj])
            total_max.append(max_inner)
    return pairs,total_max
print(138)
descriptors = []
dotsx = []
dotsy = []
for i in range(len(images)):
    Ix,Iy,R = find_R(images[i])
    doty,dotx=supression(R,images[i])
    dotsx.append(dotx)
    dotsy.append(doty)
    #orientation(Ix,Iy)
    descriptor = Haris_descriptors(images[i],doty,dotx)
    descriptors.append(descriptor)
pairs = []
total_max = []
for i in range(len(descriptors)-1):
    pair,max_inner = match(descriptors[i],descriptors[i+1])
    pairs.append(pair)
    total_max.append(max_inner)
print('finished inner')
def flat_yx_to_cylinder(flat_y ,flat_x, h, w, focal_length):
    y = flat_y - int(h/2)
    x = flat_x - int(w/2)

    tmp_x = focal_length*math.atan(x/focal_length)
    tmp_y = focal_length*y/math.sqrt(x**2+focal_length**2)
    tmp_x = int(round(tmp_x + w/2))
    tmp_y = int(round(tmp_y + h/2))

    if tmp_x >= 0 and tmp_x < w and tmp_y >= 0 and tmp_y < h:
        return(tmp_y, tmp_x)
    else:
        return(None, None)


def cylindrical_warp(imgs_rgb, focal_length):
    imgs_rgb = np.array(imgs_rgb)
    h, w = imgs_rgb[0].shape[:2]
    warped_imgs = []
    for i,img in enumerate(imgs_rgb):
        warped_img = np.zeros(shape=img.shape, dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                tmp_y, tmp_x = flat_yx_to_cylinder(y ,x, h, w, focal_length)
                if tmp_x != None:
                    warped_img[tmp_y][tmp_x] = img[y][x]
        warped_imgs.append(warped_img)
    return warped_imgs

class Img_pair:
    def __init__(self, img_left, img_right):
        self.img_left = img_left
        self.img_right = img_right
        self.left_points = []
        self.right_points = []
        self.transform_matrix = []

img_warped_pairs = []
images_rgb_warped = copy.deepcopy(images_rgb_copy)
images_rgb_warped = cylindrical_warp(images_rgb_warped, FOCAL_LENGTH)
print('finished warping')

for i in range(len(images_rgb_warped)-1):
    img_pair = Img_pair(copy.deepcopy(images_rgb_warped[i]), copy.deepcopy(images_rgb_warped[i+1]))
    img_warped_pairs.append(img_pair)

for i, (pair_list, img_pair) in enumerate(zip(pairs, img_warped_pairs)):
    for pair in pair_list:
        left_y, left_x, left_des = dotsy[i][pair[0]], dotsx[i][pair[0]], descriptors[i][pair[0]]
        right_y, right_x, right_des = dotsy[i+1][pair[1]], dotsx[i+1][pair[1]], descriptors[i+1][pair[1]]

        left_warp_y, left_warp_x = flat_yx_to_cylinder(left_y ,left_x, HEIGHT, WIDTH, FOCAL_LENGTH)
        right_warp_y, right_warp_x = flat_yx_to_cylinder(right_y ,right_x, HEIGHT, WIDTH, FOCAL_LENGTH)
        if left_warp_y == None or right_warp_y == None:
            continue

        img_pair.left_points.append([left_warp_y, left_warp_x, left_des])
        img_pair.right_points.append([right_warp_y, right_warp_x, right_des])

def dis(y1, x1, y2, x2):
    return math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

def RANSAC(img_pair):
    use_random , K = False, len(img_pair.left_points)
    if(len(img_pair.left_points) > RANSAC_K):
        use_random = True
        K = RANSAC_K

    best_shift = []
    dy, dx = 0, 0
    max_inliner = 0

    left_yx = np.array(img_pair.left_points)[:, :2]
    right_yx = np.array(img_pair.right_points)[:, :2]

    for k in range(K):
        idx = int(np.random.random_sample()*len(img_pair)) if use_random else k
        dy, dx = right_yx[idx] - left_yx[idx]

        new_right_yx = np.array(right_yx, dtype=object)
        new_right_yx[:, 0] -= dy
        new_right_yx[:, 1] -= dx

        inliner = 0
        for (ly, lx), (ry, rx) in zip(left_yx, new_right_yx):
            if dis(ly, lx, ry, rx) < RANSAC_THRES_DIS:
                inliner += 1
        if inliner > max_inliner:
            max_inliner, best_shift = inliner, [dy, dx]

    return best_shift

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

dys, dxs = [], []
for img_pair in img_warped_pairs:
    left_yx = img_pair.left_points
    dy, dx = RANSAC(img_pair)
    dys.append(dy)
    dxs.append(dx)
    shift_img_right = translate(img_pair.img_right, -dx, -dy)
print('finish ransac finding dx, dy')

def crop_image(images_rgb_warped, BLACK = 10):
    images_cropped = []
    h, w = images_rgb_warped[0].shape[:2]
    _, left_bound= flat_yx_to_cylinder(0 , 0, h, w, FOCAL_LENGTH)
    _, right_bound = flat_yx_to_cylinder(h, w, h, w, FOCAL_LENGTH)
    print("left bound, right bound = ", left_bound, right_bound)
    for img in images_rgb_warped:
        img = img[:, left_bound+BLACK:right_bound]
        images_cropped.append(img)
    return images_cropped

images_cropped = crop_image(images_rgb_warped, 10)
print(images_cropped[0].shape)

#cv2.imwrite(dir + "/result/cropped_warp.jpg", images_cropped[0])

h, w = images_cropped[0].shape[:2]
dys, dxs = np.array(dys), np.array(dxs)
tot_width = w + abs(sum(dxs))

down, up, crr_dy = 0, 0, 0
for dy in dys:
    crr_dy += dy
    if crr_dy < down:
        down = crr_dy
    if crr_dy > up:
        up = crr_dy

max_blank = max(abs(down),abs(up))

print(dys)

def add_blank(imgs, max_blank):
    images_blanked = []
    for img in imgs:
        h, w = img.shape[:2]
        new_img = np.zeros((h + 2 * max_blank, w, 3))
        new_img[max_blank:h+max_blank] = img
        images_blanked.append(new_img)
    return images_blanked

images_blanked = add_blank(images_cropped, max_blank)

crr_dy = 0
for i, dy in enumerate(dys):
    crr_dy += dy
    images_blanked[i+1] = translate(images_blanked[i+1], 0, -crr_dy)

def naive_stitch(imgs, dxs, tot_width):
    h, w = imgs[0].shape[:2]
    tot_img = np.zeros((h, tot_width, 3)) 
    tot_img[:, :w] = imgs[0]
    crr_x = w
    for dx, img in zip( abs(dxs), imgs[1:]):
        start_x = int(w - dx)
        tot_img[:, crr_x: crr_x + dx,:] = img[:, start_x:, :]
        crr_x += dx
    return tot_img

def blend(imgA, imgB):
    h, w = imgA.shape[:2]
    res = np.zeros(imgA.shape)
    for i in range(w):
        colA, colB = imgA[:, i], imgB[:, i]
        colC = colA * (w-1 - i)/w + colB * (i+1)/w 
        colC.astype(int)
        res[:, i] = colC
    return res
def blend_overlaped_stitch(imgs, dxs, tot_width):
    h, w = imgs[0].shape[:2]
    tot_img = np.zeros((h, tot_width, 3)) 
    tot_img[:, :w] = imgs[0]
    crr_x = w
    for dx, img in zip(abs(dxs), imgs[1:]):
        start_x = int(w - dx)
        tot_img[:, crr_x: crr_x + dx,:] = img[:, start_x:, :]
        overlaped = blend(tot_img[:, crr_x-start_x:crr_x], img[:, :start_x])
        tot_img[:, crr_x-start_x: crr_x] = overlaped
        crr_x += dx
    return tot_img
def blend_edge_stitch(imgs, dxs, tot_width, thres):
    h, w = imgs[0].shape[:2]
    tot_img = np.zeros((h, tot_width, 3)) 
    tot_img[:, :w] = imgs[0]
    crr_x = w
    for dx, img in zip(abs(dxs), imgs[1:]):
        start_x = int(w - dx)
        tot_mid = crr_x - start_x + int(start_x/2)
        right_mid = int(start_x/2)

        mid_blend = blend(tot_img[:, tot_mid - thres:tot_mid + thres], img[:, right_mid - thres:right_mid + thres])
        tot_img[:, tot_mid - thres:tot_mid + thres] = mid_blend
        tot_img[:, tot_mid + thres: crr_x + dx,:] = img[:, right_mid + thres:, :]
        crr_x += dx
    return tot_img

naive_img = naive_stitch(images_blanked, dxs, tot_width)
blend_overlaped_img = blend_overlaped_stitch(images_blanked, dxs, tot_width)
blend_edge_img = blend_edge_stitch(images_blanked, dxs, tot_width, BLEND_THRES)


#cv2.imwrite(dir + '/result/before_aligned_edge.jpg', blend_edge_img)


tot_dy = sum(dys)
print(tot_dy)

def end_to_end_align(tot_img, tot_dy):
    h, w = tot_img.shape[:2]
    new_img = np.array(tot_img)
    for i in range(w):
        dy = int(tot_dy * i / w)
        new_img[:, i] = translate(tot_img[:, i], 0, dy)
    return new_img
aligned_naive = end_to_end_align(naive_img, tot_dy)
aligned_overlap = end_to_end_align(blend_overlaped_img, tot_dy)
aligned_edge = end_to_end_align(blend_edge_img, tot_dy)

#cv2.imwrite(dir + '/result/aligned_overlap.jpg', aligned_overlap)
#cv2.imwrite(dir + '/result/aligned_edge.jpg', aligned_edge)
#cv2.imwrite(dir + '/result/aligned_naive.jpg', aligned_naive)

def crop_tot_img(tot_img, thres=CROP_THRES):
    h, w = tot_img.shape[0:2]
    new_img = np.array(tot_img)
    upper, lower = 0, h
    for i in range(h):
        if sum(new_img[i]) > thres:
            upper = i
            break
    for i in range(h-1, 0, -1):
        if sum(new_img[i]) > thres:
            lower = i
            break
    return new_img[upper:lower]

def crop_tot_img2(tot_img, thres=CROP_THRES):
    h, w = tot_img.shape[0:2]
    new_img = np.array(tot_img)
    upper, lower = 0, h
    for i in range(h):
        row = new_img[i]
        sum_row = np.sum(row, axis=1)
        if np.count_nonzero(sum_row) > len(sum_row) * (CROP_FACTOR-1)/CROP_FACTOR:
            print(sum_row[:20])
            upper = i
            break
    for i in range(h-1, 0, -1):
        row = new_img[i]
        sum_row = np.sum(row, axis=1)
        if np.count_nonzero(sum_row) > len(sum_row) * (CROP_FACTOR-1)/CROP_FACTOR:
            lower = i
            break
    return new_img[upper:lower]


#cropped_naive = crop_tot_img(aligned_naive, CROP_THRES)
#cropped_overlap = crop_tot_img(aligned_overlap, CROP_THRES)
#cropped_edge = crop_tot_img(aligned_edge, CROP_THRES)
cropped_naive = crop_tot_img2(aligned_naive, CROP_FACTOR)
cropped_overlap = crop_tot_img2(aligned_overlap, CROP_FACTOR)
cropped_edge = crop_tot_img2(aligned_edge, CROP_FACTOR)
#cv2.imwrite(dir + '/result/cropped_naive.jpg', cropped_naive)
#cv2.imwrite(dir + '/result/cropped_overlap.jpg', cropped_overlap)
cv2.imwrite(dir + '/result/' + file_name + '.png', cropped_edge)
