import os, random
import cv2
import argparse
def half_mirror(folder, save):
    origin_image = folder
    file_list = os.listdir(origin_image)
    save_path = save
    for i in file_list:
    
        img = cv2.imread(os.path.join(origin_image,i))
        if img.ndim == 3:
            _,w,_ = img.shape
            half = int(w/2)
            img_left = img[ :, :half, :]
            img_right = img[ :, half:, :]
            arg_image = cv2.hconcat([img_right, img_left])
        else :
            _,w = img.shape
            half = int(w/2)
            img_left = img[ :, :half]
            img_right = img[ :, half:]
            arg_image = cv2.hconcat([img_right, img_left])
        print(i)
        cv2.imwrite(save_path+"/{}".format(i),arg_image)
        # cv2.imwrite(save_path+"/{}".format(i),arg_img)
    # return arg_image

origin_image = "/database/very_long_dataset/tf_dataset_new/train"
mask_image = '/home/godgang/edge-connect/examples/eval_test/mask/outpaint_mask/mask25/01999.jpg'
file_list = os.listdir(origin_image)
save_path ="/database/very_long_dataset/tf_dataset_new/aug_train"
# for i in file_list:
#     print(i)
#     img = cv2.imread(os.path.join(origin_image,i))
#     rearrange=half_mirror(origin_image,save_path)
#     # arg_img = cv2.flip(img,1)
#     # arg_img = half_mirror(img)
#     # arg_img = cv2.resize(img, dsize=(32, 16), interpolation=cv2.INTER_AREA)
#     # h,w,c = img.shape
#     # mask_img = cv2.imread(mask_image)
#     # mask_img = cv2.resize(mask_img,dsize=(w,h))
#     # arg_img= outpaint_masked_img(img,mask_img)
#     # arg_img = half_mirror(img)

#     cv2.imwrite(save_path+"/{}.jpg".format(int(i[:-4])+5041),arg_img)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='result oti folder path')
    parser.add_argument('--save', type=str, help='save path')
    opt = parser.parse_known_args()[0]

    if opt.path is not None:
        half_mirror(opt.path,opt.save)
        
        
