"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
from scipy.io import loadmat as load
import numpy as np
import copy
from time import gmtime, strftime
from PIL import Image
import os
from glob import glob
import tensorflow as tf
from wavelet import *
# Processing images and loading images

def random(batch_size):
    dir_path = '/home/guangyuan/CelebA/Img/img_align_celeba'
    names = os.listdir(dir_path)
    data = []
    for name in names:
        sr_path = dir_path + '/' + name
        data.append(sr_path)
    np.random.shuffle(data)
    batch_files = list(data[0 * batch_size:(0 + 1) * batch_size])
    batch_images = [load_data(batch_file) for batch_file in batch_files]
    batch_images = np.array(batch_images).astype(np.float32)
    return batch_images

def get_loader(batch_size,scale_size=64,seed=None):

    x = tf.convert_to_tensor(random(batch_size), dtype=tf.float32)
    return x

def make_grid(tensor,nrow=8,padding=2):
    nmaps = tensor.shape[0]
    xmaps = min(nmaps,nrow)
    ymaps = int(math.ceil(float(nmaps)/xmaps))
    hp, wp = int(tensor.shape[1]+padding), int(tensor.shape[2]+padding)
    grid = np.zeros([ymaps*hp + 1 + padding//2, xmaps*wp + 1 + padding//2, 3],dtype=np.uint8)
    k=0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, w =y*hp + 1 + padding//2, x*wp + 1 + padding//2
            grid[h:h+tensor.shape[1],w:w+tensor.shape[2]] = tensor[k]
            k=k+1

    return grid

def save_image(tensor,filename, nrow=8, padding=2):

    ndarr = make_grid(tensor,nrow=nrow,padding=padding)

    im =Image.fromarray(ndarr)

    im.save(filename)

def norm_img(image):
    return image/127.5-1

def norm_wavelet(image):
    return (image-np.min(image))*2/(np.max(image)-np.min(image))-1

def denorm_img(norm):
    return tf.clip_by_value((norm+1)*127.5,0,255)

def load_data(image_path, flip=True, is_test=False):
    img = load_image(image_path)
    img = preprocess(img, flip=flip, is_test=is_test)
    # img = img/127.5 - 1.
    return img

def load_image(image_path):
    img = imread(image_path)
    return img

def preprocess(img, load_size=64, fine_size=64, flip=True, is_test=False):
    if is_test:
        img = crop_single(img)
        img = scipy.misc.imresize(img, [fine_size, fine_size])
    else:
        img = crop_single(img)
        img = scipy.misc.imresize(img, [load_size, load_size])

    return img

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)





