import numpy as np
import pywt
import scipy.misc

# Processing images to get inner and clear part (including two eyes, a nose, and a mouth) with full frequency
# and get outer and blur part (except two eyes, a nose, and a mouth) with low frequency

def crop(images, offh=50,offw=25,scale_size=128):
    images = images.transpose([1,2,0,3])
    # print(images.shape)
    images = images[offh:offh+scale_size,offw:offw+scale_size].transpose([2,0,1,3])
    return images
def crop_single(images, offh=50,offw=25,scale_size=128):
    # print(images.shape)
    images = images[offh:offh+scale_size,offw:offw+scale_size]
    return images

def mask_eye(w=128,offh=49,target_h=20,offw=27):
    scale = (128/w)
    offh = int((offh+1)/scale) - 1
    target_h = int(target_h/scale)
    offw = int((offw+1)/scale) - 1

    target_w = w - offw*2
    mask_in = np.zeros([w,w,3])
    mask_in[offh:offh+target_h,offw:offw+target_w,:]=1
    mask_out =1 -mask_in
    return mask_in,mask_out
def mask_nose(w=128,offh=69,target_h=43,offw=45):
    scale = 128 / w
    offh = int((offh + 1) / scale) - 1
    target_h = int(target_h / scale)
    offw = int((offw + 1) / scale) - 1
    target_w = w - offw * 2
    mask_in = np.zeros([w,w,3])
    mask_in[offh:offh+target_h,offw:offw+target_w,:]=1
    mask_out =1 -mask_in
    return mask_in,mask_out
def mask_all(w=128):
    mask_in = np.zeros([w, w, 3])
    mask_in = mask_eye(w=w)[0]+mask_nose(w=w)[0]
    mask_out = 1 - mask_in
    return mask_in, mask_out

def mask_in(images,w=128):
    mask_in = np.zeros([w, w, 3])
    mask_in = mask_eye(w=w)[0] + mask_nose(w=w)[0]
    mask_out = 1 - mask_in
    return mask_in*images, mask_out*images
def wavelet_blur(images,w=128,inside=1, n=1):
    cA0 = images
    if n==1:

        (cA1, (cH1, cV1, cD1)) = pywt.dwt2(cA0, 'haar', axes=(1, 2))
        cH1, cV1, cD1 =0.0*cH1, 0.0*cV1, 0.0*cD1
        cA0 = pywt.idwt2((cA1, (cH1, cV1, cD1)),'haar',axes=(1,2))
    elif n==2:
        (cA1, (cH1, cV1, cD1)) = pywt.dwt2(cA0, 'haar', axes=(1, 2))
        (cA2, (cH2, cV2, cD2)) = pywt.dwt2(cA1, 'haar', axes=(1, 2))
        cH1, cV1, cD1 = 0 * cH1, 0 * cV1, 0 * cD1
        cH2, cV2, cD2 = 0 * cH2, 0 * cV2, 0 * cD2
        cA1 = pywt.idwt2((cA2, (cH2, cV2, cD2)),'haar',axes=(1,2))
        cA0 = pywt.idwt2((cA1, (cH1, cV1, cD1)), 'haar', axes=(1, 2))
    elif n==3:
        (cA1, (cH1, cV1, cD1)) = pywt.dwt2(cA0, 'haar', axes=(1, 2))
        (cA2, (cH2, cV2, cD2)) = pywt.dwt2(cA1, 'haar', axes=(1, 2))
        (cA3, (cH3, cV3, cD3)) = pywt.dwt2(cA2, 'haar', axes=(1, 2))
        cH1, cV1, cD1 = 0 * cH1, 0 * cV1, 0 * cD1
        cH2, cV2, cD2 = 0 * cH2, 0 * cV2, 0 * cD2
        cH3, cV3, cD3 = 0 * cH3, 0 * cV3, 0 * cD3
        cA2 = pywt.idwt2((cA3, (cH3, cV3, cD3)), 'haar', axes=(1, 2))
        cA1 = pywt.idwt2((cA2, (cH2, cV2, cD2)), 'haar', axes=(1, 2))
        cA0 = pywt.idwt2((cA1, (cH1, cV1, cD1)), 'haar', axes=(1, 2))
    elif n == 4:
        (cA1, (cH1, cV1, cD1)) = pywt.dwt2(cA0, 'haar', axes=(1, 2))
        (cA2, (cH2, cV2, cD2)) = pywt.dwt2(cA1, 'haar', axes=(1, 2))
        (cA3, (cH3, cV3, cD3)) = pywt.dwt2(cA2, 'haar', axes=(1, 2))
        (cA4, (cH4, cV4, cD4)) = pywt.dwt2(cA3, 'haar', axes=(1, 2))
        cH1, cV1, cD1 = 0 * cH1, 0 * cV1, 0 * cD1
        cH2, cV2, cD2 = 0 * cH2, 0 * cV2, 0 * cD2
        cH3, cV3, cD3 = 0 * cH3, 0 * cV3, 0 * cD3
        cH4, cV4, cD4 = 0 * cH4, 0 * cV4, 0 * cD4
        cA3 = pywt.idwt2((cA4, (cH4, cV4, cD4)), 'haar', axes=(1, 2))
        cA2 = pywt.idwt2((cA3, (cH3, cV3, cD3)), 'haar', axes=(1, 2))
        cA1 = pywt.idwt2((cA2, (cH2, cV2, cD2)), 'haar', axes=(1, 2))
        cA0 = pywt.idwt2((cA1, (cH1, cV1, cD1)), 'haar', axes=(1, 2))
    if inside==1:
        image1 = cA0*mask_all(w=w)[0]
        image2 = images*mask_all(w=w)[1]
    else:
        image1 = cA0 * mask_all(w=w)[1]
        image2 = images * mask_all(w=w)[0]
    return image1, image2