import cv2, glob, os
import numpy as np
import pandas as pd

# See Dr. Graham's preprocessing's method
# https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801

def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2
    
    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2

    return (ry, rx)

def subtract_gaussian_blur(img):
    # http://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
    # http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)
    
    return cv2.addWeighted(img, 4, gb_img, -4, 128)

def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
    
    return a * b + 128 * (1 - b)

def crop_img(img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0
                
        crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]
        
        return crop_img

def place_in_square(img, r, h, w):
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img += 128
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img
    
    return new_img

def preprocess(f, r, debug_plot=False):
    try:
        img = cv2.imread(f)
        
        ry, rx = estimate_radius(img)
        
        if debug_plot:
            plt.figure()
            plt.imshow(img)
        
        resize_scale = r / max(rx, ry)
        w = min(int(rx * resize_scale * 2), r * 2)
        h = min(int(ry * resize_scale * 2), r * 2)
        
        img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)
        
        img = crop_img(img, h, w)
        print("crop_img", np.mean(img), np.std(img))
        
        if debug_plot:
            plt.figure()
            plt.imshow(img)
        
        img = subtract_gaussian_blur(img)
        img = remove_outer_circle(img, 0.9, r)
        img = place_in_square(img, r, h, w)
        
        if debug_plot:
            plt.figure()
            plt.imshow(img)

        return img

    except Exception as e:
        print("file {} exception {}".format(f, e))

    return None

input_path = "../input"
df = pd.read_csv(os.path.join(input_path, "trainLabels.csv"))

train_files = glob.glob(os.path.join(input_path, "train", "*.jpeg"))
out_directory = "../preprocess/512/train"
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

def process_and_save(f):
    basename = os.path.basename(f)
    image_id = basename.split(".")[0]

    if len(df[df['image'] == image_id]) < 1:
        print("missing annotation: " + image_id)
        return

    target_path = os.path.join(out_directory, basename)

    print("processing:", f, target_path)

    if os.path.exists(target_path):
        print("skip: " + target_path)
        return

    result = preprocess(f, 256)
    if result is None:
        return

    # NOTE: Filter low contrast images for tutorial
    std = np.std(result)
    if std < 12:
        print("skip low std", std, f)
        return

    if result is not None:
        print(cv2.imwrite(target_path, result))

from joblib import Parallel, delayed

# Specify the number of cpu cores
Parallel(n_jobs=16)(delayed(process_and_save)(f) for f in train_files)