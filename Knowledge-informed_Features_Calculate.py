from scipy.signal import *
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import ndimage
import pywt  # pip install PyWavelets
from skimage import measure
import os
from astropy.io import fits
import imageio.v2 as imageio
import csv
import glob
from functools import partial
from multiprocessing import Pool, Manager
import cv2


def fluxValues(magnetogram):
    # compute sum of positive and negative values,
    # then evaluate a signed and unsigned sum.
    posSum = np.sum(magnetogram[magnetogram > 0])
    negSum = np.sum(magnetogram[magnetogram < 0])
    signSum = posSum + negSum
    unsignSum = posSum - negSum

    return posSum, negSum, signSum, unsignSum


def gradient(image):
    # use sobel operators to find the gradient
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    gx = convolve2d(image, sobelx, mode='same')
    gy = convolve2d(image, sobely, mode='same')

    M = (gx ** 2 + gy ** 2) ** (1. / 2)

    return M


def Gradfeat(image):
    # evaluate statistics of the gradient image
    res = gradient(image).flatten()
    men = np.mean(res)
    strd = np.std(res)
    med = np.median(res)
    minim = np.amin(res)
    maxim = np.amax(res)
    skw = skew(res)
    kurt = kurtosis(res)

    return men, strd, med, minim, maxim, skw, kurt


def wavel(image):
    # create wavelet transform array for display
    # can be added to the return statement for
    wt = pywt.wavedecn(image, 'haar', level=5)
    arr, coeff_slices = pywt.coeffs_to_array(wt)

    # compute wavelet energy
    LL, L5, L4, L3, L2, L1 = pywt.wavedec2(image, 'haar', level=5)
    L1e = np.sum(np.absolute(L1))
    L2e = np.sum(np.absolute(L2))
    L3e = np.sum(np.absolute(L3))
    L4e = np.sum(np.absolute(L4))
    L5e = np.sum(np.absolute(L5))

    return L1e, L2e, L3e, L4e, L5e


def extractNL(image):

    avg10 = (1. / 100) * np.ones([10, 10])
    avgim = convolve2d(image, avg10, mode='same')
    out = measure.find_contours(avgim, level=0)
    return out


def NLmaskgen(contours, image):

    mask = np.zeros((image.shape))
    for n, contour in enumerate(contours):
        # print(n,contour)
        for i in range(len(contour)):
            y = int(round(contour[i, 1]))
            x = int(round(contour[i, 0]))
            mask[x, y] = 1.
    return mask


def findTGWNL(image):

    m = 0.2 * np.amax(np.absolute(image))
    # width = image.shape[0]
    # height = image.shape[1]
    # out = np.zeros([height, width])
    out = np.zeros(image.shape)
    out[abs(image) >= m] = 1

    return out


def curvature(contour):

    angles = np.zeros([contour.shape[0]])
    yvals = np.around(contour[:, 1])
    xvals = np.around(contour[:, 0])
    for i in range(contour.shape[0]):
        if i < contour.shape[0] - 1:
            n = i + 1
        else:
            n = 0
        y = int(yvals[i])
        x = int(xvals[i])
        yn = int(yvals[n])
        xn = int(xvals[n])
        num = yn - y
        den = xn - x
        if den != 0:
            angles[i] = np.arctan(num / den)
        elif num < 0:
            angles[i] = 3 * np.pi / 2
        else:
            angles[i] = np.pi / 2
    return angles


def bendergy(angles):

    fact = 1. / len(angles)
    count = 0.
    for i in range(len(angles)):
        if i < len(angles) - 1:
            n = i + 1
        else:
            n = 0
        T = angles[i]
        Tn = angles[n]
        count += (T - Tn) ** 2

    BE = count * fact
    return BE


def NLfeat(image):

    grad = gradient(image)
    contours = extractNL(image)
    ma = NLmaskgen(contours, image)
    gwnl = np.zeros([grad.shape[0], grad.shape[1]])
    gwnl = grad * ma
    thresh = findTGWNL(gwnl)
    NLlen = np.sum(thresh)

    struct = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    lines, numlines = ndimage.label(thresh, struct)

    GWNLlen = np.sum(ma)
    Flag = True
    if not contours:
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    else:
        for n, contour in enumerate(contours):
            curve = curvature(contour)
            if Flag:
                angstore = np.zeros([len(curve)])
                angstore = curve
                BEstore = np.zeros([len(contours)])
                Flag = False
            else:
                angstore = np.concatenate((curve, angstore))
            BEstore[n] = bendergy(curve)


    return float(NLlen), float(numlines), float(GWNLlen), float(np.mean(angstore)), np.std(angstore), np.median(
        angstore), np.amin(angstore), np.amax(angstore), np.mean(BEstore), np.std(BEstore), np.median(BEstore), np.amin(
        BEstore), np.amax(BEstore)


def concatVals(image):
    # Generate fetures
    G = Gradfeat(image)
    NL = NLfeat(image)
    wav = wavel(image)
    F = fluxValues(image)

    # Concatenate and return results
    return np.concatenate((G, NL, wav, F))


def calculate_pixel_area(CDELT1, CDELT2, distance_to_sun=1.496e13):

    CDELT1_arcsec = CDELT1 * 3600
    CDELT2_arcsec = CDELT2 * 3600

    CDELT1_rad = CDELT1_arcsec * np.pi / (180 * 3600)
    CDELT2_rad = CDELT2_arcsec * np.pi / (180 * 3600)

    pixel_area_cm2 = (distance_to_sun * CDELT1_rad) * (distance_to_sun * CDELT2_rad)

    pixel_area_km2 = pixel_area_cm2 * 1e-10

    return pixel_area_km2

def calculate_acr(magnetogram, pixel_area, threshold=1000):

    strong_field_pixels = np.abs(magnetogram) >= threshold

    num_strong_pixels = np.sum(strong_field_pixels)

    acr = num_strong_pixels * pixel_area
    return acr


def calculate_R_VALUE(magnetogram, CDELT1, CDELT2):
    contours = extractNL(magnetogram)
    pil_mask = NLmaskgen(contours, magnetogram)

    if np.sum(pil_mask) == 0:
        return 0.0

    Bz_pil = magnetogram * pil_mask  # 仅保留PIL上的磁场值

    gradient_magnitude = gradient(magnetogram)

    Bz_pil = np.nan_to_num(Bz_pil)
    gradient_magnitude = np.nan_to_num(gradient_magnitude)

    weighted_Bz_pil = Bz_pil * gradient_magnitude * pil_mask

    R_VALUE = np.sum(np.abs(weighted_Bz_pil))

    Area = (CDELT1 * CDELT2 * (np.pi / (180 * 3600)) ** 2) * 1e4

    R_VALUE_mx = R_VALUE * Area
    return R_VALUE_mx


file_extension = 'fits'  # 'fits' or 'png' or 'fits.gz'


def extract_features(Labels, filename):
    # open image
    if file_extension == 'fits':
        # Img = fits.getdata(filename, ext=-1)
        with fits.open(filename) as Img:
            Img.verify('silentfix')
            Img = Img[-1].data  # select proper index
            Img = np.nan_to_num(Img)
            # Img = cv2.resize(Img, (600, 600))
    elif file_extension == 'png':
        Img = imageio.imread(filename).astype(float)
        Img = Img - 128  # offset so that zero flux is at zero

    # Inform User
    print('Building entry for ' + os.path.basename(filename))

    # Extract Features
    features = concatVals(Img)

    if os.path.basename(filename) in Labels:
        if Labels[os.path.basename(filename)] in ['N', 'C']:
            label = '0,' + Labels[os.path.basename(filename)]
        else:
            label = '1,' + Labels[os.path.basename(filename)]
    else:
        label = 'NaN'

    return features, label


def extract_four(filename):
    data_fit = fits.getdata(filename, ext=-1)
    data_fit = np.nan_to_num(data_fit)
    data_header = fits.getheader(filename, ext=-1)
    header_AREA_ACR, header_R_VALUE = data_header['AREA_ACR'], data_header['R_VALUE']
    pixel_area = calculate_pixel_area(data_header['CDELT1'], data_header['CDELT2'])
    AREA_ACR = calculate_acr(data_fit, pixel_area)
    R_VALUE = calculate_R_VALUE(data_fit, data_header['CDELT1'], data_header['CDELT2'])
    NOAA_NUM = data_header['NOAA_NUM']
    NOAA_AR = data_header['NOAA_AR']
    return AREA_ACR, R_VALUE, header_AREA_ACR, header_R_VALUE, NOAA_NUM, NOAA_AR


def process_subdirectory(label_txt, dataset_dir, output_dir, file_extension, num_workers=32):
    print(f'加载 {label_txt} 中的标签文件')
    Labels = {}
    with open(label_txt) as f:
        csv_data = csv.reader(f, delimiter=',')
        Labels.update(dict(csv_data))

    Labels = Manager().dict(Labels)

    print(f'查找 {dataset_dir} 中的图像文件')
    filenames = sorted(glob.glob(os.path.join(dataset_dir, '*.' + file_extension)))

    if not filenames:
        print(f"{dataset_dir} 中没有匹配的 FITS 文件")
        return

    filenames_base = [os.path.basename(filename) for filename in filenames]

    print('提取特征')
    with Pool(num_workers) as p:
        feature_matrix, label_vector = zip(*p.map(partial(extract_features, Labels), filenames))
        AREA_ACR, R_VALUE, header_AREA_ACR, header_R_VALUE, NOAA_NUM, NOAA_AR = zip(*p.map(extract_four, filenames))

    # 设置输出文件路径
    output_file = os.path.join(output_dir, os.path.basename(dataset_dir) + '_features.csv')
    print(f'创建 {output_file}')

    file_header = (
        'Grad_mean,Grad_std,Grad_median,Grad_min,Grad_max,Grad_skew,Grad_kurt,NL_len,NL_numlines,GWNL_len,NL_mean,NL_std,NL_median,NL_min,NL_max,NLBE_mean,NLBE_std,NLBE_median,NLBE_min,NLBE_max,Wavelet_level1,Wavelet_level2,Wavelet_level3,Wavelet_level4,Wavelet_level5,Flux_positive,Flux_negative,Flux_signed,Flux_unsigned,AREA_ACR,R_VALUE,header_AREA_ACR,header_R_VALUE,P_or_N,flare_level,NOAA_NUM,NOAA_AR,AR_num,filename'
    )

    folder_names = [os.path.basename(os.path.dirname(filename)) for filename in filenames]

    np.savetxt(output_file, np.hstack((
        np.asarray(feature_matrix),
        np.expand_dims(np.asarray(AREA_ACR), 1),
        np.expand_dims(np.asarray(R_VALUE), 1),
        np.expand_dims(np.asarray(header_AREA_ACR), 1),
        np.expand_dims(np.asarray(header_R_VALUE), 1),
        np.expand_dims(np.asarray(label_vector), 1),
        np.expand_dims(np.asarray(NOAA_NUM), 1),
        np.expand_dims(np.asarray(NOAA_AR), 1),
        np.expand_dims(np.asarray(folder_names), 1),
        np.expand_dims(np.asarray(filenames_base), 1)
    )), delimiter=',', fmt='%s', header=file_header, comments='')

    print(f'{dataset_dir} 处理完成')


def main():
    label_root_dir = 'G:/23_24_SHARP_Physics/N/Labels/'  # 标签文件根目录
    dataset_root_dir = 'G:/23_24SHARPImage_40/N/'  # 数据集根目录
    output_root_dir = 'G:/23_24_SHARP_Physics/N/Outputs/'  # 输出文件根目录
    file_extension = 'fits'  # 数据文件扩展名

    # 遍历 label_root_dir 和 dataset_root_dir 中的子文件夹
    subdirs = [os.path.basename(subdir) for subdir in glob.glob(os.path.join(dataset_root_dir, '*')) if
               os.path.isdir(subdir)]

    for subdir in subdirs:
        label_dir = os.path.join(label_root_dir, subdir)
        dataset_dir = os.path.join(dataset_root_dir, subdir)
        output_dir = os.path.join(output_root_dir, subdir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 假设每个子文件夹中只有一个 txt 文件
        label_txt = glob.glob(os.path.join(label_dir, '*.txt'))
        if label_txt:
            process_subdirectory(label_txt[0], dataset_dir, output_dir, file_extension)
        else:
            print(f"{label_dir} 中没有找到标签文件")


if __name__ == '__main__':
    main()
