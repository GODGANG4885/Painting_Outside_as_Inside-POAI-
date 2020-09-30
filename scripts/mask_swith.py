import cv2
import os
import scipy
import numpy as np

def main():

    """input 준비 """
    image_path = "/home/godgang/edge-connect/examples/natural/images_sample"
    Gsave_path = "/home/godgang/edge-connect/examples/natural/2_outpaint_masked_image"
    Hsave_path = "/home/godgang/edge-connect/examples/natural/3_masked_image"
    generate_masked_image(image_path,Gsave_path)
    half_mirror(Gsave_path,Hsave_path)


def half_mirror(image_path,Gsave_path):
    # print(image.ndim)
    
    # save_path = "/home/godgang/edge-connect/examples/natural/3_masked_image/high"
    image_folder = os.listdir(image_path)
    # blank_image = np.zeros((128,128,3), np.uint8)
    # blank_image = (255,255,255)
    for image_f in image_folder:

        image = cv2.imread(os.path.join(image_path,image_f),cv2.IMREAD_COLOR )

        if image.ndim == 3:
            h,w,_ = image.shape
            half = int(w/2)
            img_left = image[ :, :half, :]
            img_right = image[ :, half:, :]
            arg_image = cv2.hconcat([img_right, img_left])
        else :
            h,w = image.shape
            half = int(w/2)
            img_left = image[ :, :half]
            img_right = image[ :, half:]
            arg_image = cv2.hconcat([img_right, img_left])

        
        # arg_image = cv2.hconcat([img_right[:,:64,:], blank_image])
        # arg_image = cv2.hconcat([arg_image, img_left[:,64:,:]])
        print("saved image")
        cv2.imwrite(os.path.join(save_path,image_f),arg_image)

def augment(image,rotate):
    
    img_left = image[ :, :rotate, :]
    img_right = image[ :, rotate:, :]
    arg_image = cv2.hconcat([img_right, img_left])
    return arg_image
def generate_masked_image(image_path,Gsave_path):
    # image_path = "/home/godgang/edge-connect/examples/natural/images_sample/high"
    mask_path = "/database/sun/out_mask.jpg"
    mask = cv2.imread(mask_path,cv2.IMREAD_COLOR)
    # Gsave_path = "/home/godgang/edge-connect/examples/natural/masked_image/high"
    image_folder = os.listdir(image_path)
    
    for image_f in image_folder:
        print(image_f)
        image = cv2.imread(os.path.join(image_path,image_f),cv2.IMREAD_COLOR )
        h,w,_ =image.shape
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_CUBIC)
        image = cv2.add(image,mask)
        cv2.imwrite(os.path.join(Gsave_path,image_f),image)
def image_concat():
    original_image_path = "/home/godgang/edge-connect/examples/natural/images_sample"
    masked_image_path ="/home/godgang/edge-connect/examples/natural/outpaint_masked_image"
    output_image_path = "/home/godgang/edge-connect/examples/natural/final_result"
    save_path ="/home/godgang/edge-connect/examples/natural/concat"
    image_folder = os.listdir(original_image_path)
    for image_f in image_folder:
        
        concat = cv2.hconcat([cv2.imread(os.path.join(original_image_path,image_f)),cv2.imread(os.path.join(masked_image_path,image_f))])
        concat = cv2.hconcat([concat,cv2.imread(os.path.join(output_image_path,image_f))])

        cv2.imwrite(os.path.join(save_path,image_f),concat)
    
if __name__ == "__main__":
    image_path = "/database/godgang/beach"
    save_path = "/database/godgang/beach_oti"
    image_folder = os.listdir(image_path)
    count=0
    '''generate_masksed_image'''
    # generate_masked_image(image_path,save_path)
    '''rearrange'''
    half_mirror(image_path,save_path)
    # for image_f in image_folder:
    #     image = cv2.imread(os.path.join(image_path,image_f),cv2.IMREAD_COLOR )
    # # #     # slice_pix = 192

    # # #     # img_left = image[ :, :slice_pix, :]
    # # #     # img_right = image[ :, slice_pix:, :]
    # # #     # arg_image = cv2.hconcat([img_right, img_left])
    #     crop_image = cv2.resize(image,(256,128))
    #     cv2.imwrite(os.path.join(save_path,image_f[:-4])+'.jpg',crop_image)
    print("done.")
        # for rotate in range(1,24):
        #     image=cv2.imread(os.path.join(image_path,image_f),cv2.IMREAD_COLOR )
        #     _,w,_ = image.shape
        #     # flip_image = cv2.flip(image,1)
        #     pix_rotate = int(w/24*rotate)
        #     aug_image=augment(image,pix_rotate)
        #     new_name = "{}_{}".format(image_f[:-4],rotate)
        #     print(new_name)
        #     cv2.imwrite(os.path.join(save_path,new_name)+'.jpg',aug_image)
        #     count+=1
            # print(rotate)


