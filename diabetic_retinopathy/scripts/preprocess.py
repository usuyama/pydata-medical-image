import cv2, os
from glob import glob
import numpy as np
import pandas as pd

input_path = "../inputs/"
output_path = "../preprocess/256/"

train_files = glob(os.path.join(input_path, "train/*.jpeg"))
test_files = glob(os.path.join(input_path, "test/*.jpeg"))

def estimate_radius(img):
    y = img[img.shape[0] //2,:,:].sum(1)
    ry = (y > y.mean() / 10).sum() / 2
    
    x = img[:,img.shape[1] // 2,:].sum(1)
    rx = (x > x.mean() / 10).sum() / 2

    return (rx, ry)

def preprocess(f, target_radius):
    try:
        img = cv2.imread(f)
        
        if img is None:
            print("failed to cv2.imread", f)

            return None
        
        rx, ry = estimate_radius(img)
        r = min(rx, ry)
        if r < 10:
            print("estimated radius is too small")

            return None

        resize_scale = target_radius / min(rx, ry)
        img = cv2.resize(img, (0,0), fx=resize_scale, fy=resize_scale)

        return img

    except Exception as e:
        print("file {} exception {}".format(f, e))

    return None

def crop(a, radius):
    x_margin = (a.shape[0] - radius * 2) // 2
    y_margin = (a.shape[1] - radius * 2) // 2
    
    return a[x_margin:x_margin + radius * 2, y_margin:y_margin + radius * 2]

def save_preprocessed_img(f, output_folder):
    basename = os.path.basename(f)
    target_path = os.path.join(output_folder, basename)

    print("processing:", f, target_path)

    if os.path.exists(target_path):
        print("skip: " + target_path)
        return

    result = preprocess(f, target_radius)
    if result is None:
        return

    result = crop(result, target_radius)
    if result is None:
        return

    if result is not None:
        print(cv2.imwrite(target_path, result))

df = pd.read_csv(os.path.join(input_path, "trainLabels.csv"))
print(df.head())

target_radius = 128

for f in train_files:
    basename = os.path.basename(f)
    image_id = basename.split(".")[0]

    if len(df[df['image'] == image_id]) < 1:
        print("missing annotation: " + image_id)
        continue

    output_folder = os.path.join(output_path, "train")
    
    save_preprocessed_img(f, output_folder)
    
for f in test_files:
    output_folder = os.path.join(output_path, "test")
    
    save_preprocessed_img(f, output_folder)
