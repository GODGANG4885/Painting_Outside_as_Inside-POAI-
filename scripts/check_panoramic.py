import cv2
import os
def main():
    path1 = "/database/resize_panorama/real_panorama"
    # path2 = "/database/resize_panorama/augmen_image_test"
    # path3 = "/database/resize_panorama/augmen_image_train_yh"
    image_folder1 = os.listdir(path1)
    # image_folder2 = os.listdir(path2)
    # image_folder3 = os.listdir(path3)
    count = 0
    for image in image_folder1:
        count +=1
        image = cv2.imread(os.path.join(path1,image),cv2.IMREAD_COLOR )
        # size = image.shape
        _,w,_ = image.shape
        new = augment(image,int(w*1/4))
        print("/database/resize_panorama/panora_aug/45_{0:03d}.jpg".format(count))
        cv2.imwrite("/database/resize_panorama/panora_aug_image/45_{0:03d}.jpg".format(count),new)
        count +=1
        new = augment(image,int(w*2/4))
        print("/database/resize_panorama/panora_aug/90_{0:03d}.jpg".format(count))
        cv2.imwrite("/database/resize_panorama/panora_aug_image/90_{0:03d}.jpg".format(count),new)
        count +=1
        new = augment(image,int(w*3/4))
        print("/database/resize_panorama/panora_aug/90_{0:03d}.jpg".format(count))
        cv2.imwrite("/database/resize_panorama/panora_aug_image/135_{0:03d}.jpg".format(count),new)
    # for image in image_folder2:
    #     count +=1
    #     image = cv2.imread(os.path.join(path2,image),cv2.IMREAD_COLOR )
    #     # size = image.shape
    #     new = augment(image)
    #     print("/database/resize_panorama/real_panorama/o_{0:03d}.jpg".format(count))
    #     cv2.imwrite("/database/resize_panorama/real_panorama/o_{0:03d}.jpg".format(count),new)
    # for image in image_folder3:
    #     count +=1
    #     image = cv2.imread(os.path.join(path3,image),cv2.IMREAD_COLOR )
    #     # size = image.shape
    #     new = augment(image)
    #     print("/database/resize_panorama/real_panorama/o_{0:03d}.jpg".format(count))
    #     cv2.imwrite("/database/resize_panorama/real_panorama/o_{0:03d}.jpg".format(count),new)

def half_mirror(image):
    _,w,_ = image.shape
    half = int(w/2)
    img_left = image[ :, :half, :]
    img_right = image[ :, half:, :]
    arg_image = cv2.hconcat([img_right, img_left])
    return arg_image

def augment(image,rotate):
    
    img_left = image[ :, :rotate, :]
    img_right = image[ :, rotate:, :]
    arg_image = cv2.hconcat([img_right, img_left])
    return arg_image

if __name__ == "__main__":

    main()
