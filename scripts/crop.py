import cv2
import os
import scipy
import numpy as np

if __name__ == "__main__":
    image_path = "/database/panorama/15dgree_aug_with_origin"
    save_path = "/database/panorama/crop"
    image_folder = os.listdir(image_path)
    count=0
    for image_f in image_folder:
        
        image=cv2.imread(os.path.join(image_path,image_f),cv2.IMREAD_COLOR )
        h,w,_ = image.shape
        image = image[h//2-64:h//2+64,w//2-128:w//2+128,:]
        
        print(image_f)
        cv2.imwrite(os.path.join(save_path,image_f),image)