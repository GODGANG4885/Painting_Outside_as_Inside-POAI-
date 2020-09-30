import argparse
import cv2
import os
import numpy as np
# from main import my_main


''' 입력 이미지는 256X128 멀티 스텝 아웃페인팅을 하기위한 과정'''

''' step1은 입력 이미지를 bidirectional boundary rearrangement'''
def step1(img):
    h,w,_ = img.shape
    blank_image = np.zeros((128,128,3), np.uint8)
    concat = cv2.hconcat([img[:,w-64:w,:],blank_image,img[:,0:64,:]]) 
    return concat
''' predict outer regions'''
def predic():
    main(mode=2)
''' concat origin image and generated image'''
def step2(img,predic):
    h,w,_ = img.shape
    concat = cv2.hconcat([img[:,0:64,:],predic,img[:,192:,:]]) 
    return concat
def inandout(outp,inp):
    h,w,_ = outp.shape
    # temp_save = img[:,64:w-64,:]
    # blank_image = np.zeros((128,128,3), np.uint8)
    # blank_image = (0,0,0)
    # rearrange
    # concat = cv2.hconcat([img[:,w-64:w,:],blank_image,img[:,0:64,:]]) 
    concat = cv2.hconcat([outp[:,0:64,:],inp,outp[:,w-64:,:]]) 
    return concat
    # cv2.imwrite(save+"/{}".format(i),temp_save)


# def step3(input_path, pre_step_image ):
#     file_list = os.listdir(input_path)
#     for i in file_list:
#         # img = cv2.imread(os.path.join(input_path,i),cv2.IMREAD_COLOR)
#         tem_img = cv2.imread(os.path.join(pre_step_image,i),cv2.IMREAD_COLOR)
#         img = half_mirror(os.path.join(input_path,i))
#         h,w,_ = tem_img.shape
#         concat = cv2.hconcat([img[:,0:128,:],tem_img[:,64:w-64,:],img[:,128:256,:]])
def step3(gen_img, pre_step_image ):
        img = half_mirror(gen_img)
        concat = cv2.hconcat([img[:,0:64,:],pre_step_image,img[:,192:256,:]])
        return concat
def half_mirror(img):

    # img = cv2.imread(image)
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
    return arg_image
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, help='choine step')
    parser.add_argument('--input', type=str, help='input path')
    parser.add_argument('--pre_step', type=str, help='input path')
    parser.add_argument('--save', type=str, help='save path')
    opt = parser.parse_known_args()[0]
    file_list = os.listdir(opt.input)
    for i in file_list:
        
        # img = cv2.imread(os.path.join(opt.input,i),cv2.IMREAD_COLOR)
        # step = cv2.imread(os.path.join(opt.step,i),cv2.IMREAD_COLOR)
        # outp = cv2.imread(os.path.join(opt.outp,i),cv2.IMREAD_COLOR)
        input_image = cv2.imread(os.path.join(opt.input,i),cv2.IMREAD_COLOR)
        pre_step = cv2.imread(os.path.join(opt.pre_step,i),cv2.IMREAD_COLOR)
        if opt.step == 1:
            result = step1(input_image)
        # elif opt.step == 2:
        #     # predic()
        elif opt.step == 3:
            result = step3(input_image,pre_step)
        elif opt.step == 4:
        # oti = step2(img,step)
            result = half_mirror(input_image)
        
        cv2.imwrite(opt.save+"/{}".format(i),result)
    #     cv2.imwrite(opt.save+"/{}".format(i),oti)

    # step3(opt.input,opt.temp_save_path,opt.save)