import SimpleITK
import numpy as np
from glob import glob
import os
import scipy.ndimage
from joblib import Parallel, delayed

def resample(image, spacing_zyx, new_spacing_zyx=np.array([1,1,1])):
    new_real_shape = image.shape * spacing_zyx / new_spacing_zyx
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing_zyx = spacing_zyx / real_resize_factor    

    resampled_img = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    if resampled_img.shape[0] < 1 or resampled_img.shape[1] < 1 or resampled_img.shape[2] < 1:
        raise Exception('invalid image shape {}'.format(image.shape))

    return resampled_img, new_spacing_zyx

def normalize(image):
    # The original values are in Hounsefield scale
    # https://en.wikipedia.org/wiki/Hounsfield_scale
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1.0] = 1.0
    image[image < 0.0] = 0.0

    return image

def switch_xz(arr):
    return np.array([arr[2], arr[1], arr[0]])

def get_resampled_img(img_file):
    itk_img = SimpleITK.ReadImage(img_file)
    img_array = SimpleITK.GetArrayFromImage(itk_img) # z,y,x
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    origin_zyx = switch_xz(origin)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm) xyz
    spacing_zyx = switch_xz(spacing)
    
    resampled_img, new_spacing = resample(img_array, spacing_zyx) # new_spacing: zyx    
    resampled_img = normalize(resampled_img)
    
    return resampled_img, origin_zyx, new_spacing

def save_resampled_img_to_dict(img_file):
    basename = os.path.basename(img_file)
    print("processing " + basename)

    resampled_img, origin_zyx, new_spacing = get_resampled_img(img_file)

    np.savez_compressed("../preprocess/{}.npz".format(basename), resampled_img=resampled_img.astype(np.float16), origin_zyx=origin_zyx, new_spacing=new_spacing)

luna_path = "../inputs/CSVFILES/"
luna_subset_path = "../inputs/subset*/"

mhd_file_list = glob(luna_subset_path + "*.mhd")

# Recommend: change n_jobs to # of cpu cores
Parallel(n_jobs=7)(delayed(save_resampled_img_to_dict)(img_file) for img_file in mhd_file_list)
