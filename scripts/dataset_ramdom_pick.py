import os, random
import cv2
def random_pick():
    save_path = "/database/panorama/test"
    base_dir = "/database/panorama/15dgree_aug"
    random_list = random.sample(os.listdir(base_dir),1000)

    for i in random_list:
        print(i)
        img = cv2.imread(os.path.join(base_dir,i))
        cv2.imwrite(os.path.join(save_path,i),img)
def mask_folder():
    image_path = "/home/godgang/edge-connect/examples/eval_test/mask/inpaint_mask/mask25"
    img = cv2.imread(os.path.join(image_path,"00999.jpg"))
    for i in range(0,2000):
        print(i)
        cv2.imwrite(image_path+"/{0:05d}.jpg".format(i),img)



def half_mirror(image):
    # dir_path = "/home/godgang/edge-connect/examples/eval_test/inpaint_mask/mask_50"
    # file_list = os.listdir(dir_path)
    # save_path ="/home/godgang/edge-connect/examples/eval_test/masked_img/out_to_inpaint"
    # for i in file_list:

    #     img = cv2.imread(os.path.join(dir_path,i))
    #     # print(image.ndim)
    if image.ndim == 3:
        _,w,_ = image.shape
        half = int(w/2)
        img_left = image[ :, :half, :]
        img_right = image[ :, half:, :]
        arg_image = cv2.hconcat([img_right, img_left])
    else :
        _,w = image.shape
        half = int(w/2)
        img_left = image[ :, :half]
        img_right = image[ :, half:]
        arg_image = cv2.hconcat([img_right, img_left])

        # cv2.imwrite(save_path+"/{}".format(i),arg_img)
    return arg_image

def outpaint_masked_img(origin_image, mask):
    
    masked_img = cv2.add(origin_image,mask_img)
    # arg_img = half_mirror(masked_img)
    return masked_img

origin_image = "/home/godgang/NS-Outpainting/logs/0817/2/re_origval"
mask_image = '/home/godgang/edge-connect/examples/eval_test/masking/outpaint_ma/mask.jpg'
file_list = os.listdir(origin_image)
save_path ="/home/godgang/edge-connect/examples/eval_test/masked_img/mask50/re_outpaintied"
mask_img = cv2.imread(mask_image)
for i in file_list:
    print(i)
    img = cv2.imread(os.path.join(origin_image,i))
    # rearrange=half_mirror(img)
    # arg_img = cv2.flip(img,1)
    # arg_img = half_mirror(img)
    # arg_img = cv2.resize(img, dsize=(32, 16), interpolation=cv2.INTER_AREA)
    h,w,c = img.shape
    mask_img = cv2.resize(mask_img,dsize=(w,h))
    masked_img = cv2.add(img,mask_img)
    # mask_img = cv2.resize(mask_img,dsize=(w,h))
    # arg_img= outpaint_masked_img(img,mask_img)
    # arg_img = half_mirror(img)

    cv2.imwrite(save_path+"/{}".format(i),masked_img)
# mask_folder()
