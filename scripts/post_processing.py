from PIL import Image
import numpy as np
import os
origin_image = "/home/godgang/edge-connect/examples/eval_test/result/LRmore_finetune/result/oti_rearrange"
file_list = os.listdir(origin_image)
save_path ="/home/godgang/edge-connect/examples/eval_test/result/LRmore_finetune/post_processing"
for r in file_list:

    img = Image.open(os.path.join(origin_image,r))
    ycbcr = img.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    y= np.array(y)
    point_1= 64
    point_2= 192
    t = y[:,64:192]

    for i in range(point_1+1):

        if y[:,i]+=15
        y[:,i]+=15

    for i in range(point_2-1,256):
        y[:,i]+=15

    y= Image.fromarray(y)

    merged_ycbcr = Image.merge('YCbCr',(y,cb,cr))
    rgb_img = merged_ycbcr.convert('RGB')
    # k = i[:-4]
    # path = os.path.join(save_path+i+".jpg")
    rgb_img.save(save_path+"/{}".format(r))
    print(r)
