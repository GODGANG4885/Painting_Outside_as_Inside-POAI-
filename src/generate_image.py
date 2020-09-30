from PIL import Image, ImageDraw
import random
import argparse
import os
# img.save(filename)


    # if mode == None:
    #     width = 1920
    #     height= 256
    #     count =0
        # for mask_width in range(2,960,2):
        #     count +=1
        #     mask = Image.new('RGB', (width,height), color ='black')
        #     mask_img = Image.new('RGB',(mask_width,height),color ='white')
        #     center = int(960 - mask_width/2)
        #     print(center)
        #     mask.paste(mask_img,(center,0))
        #     mask.save("/database/resize_panorama/mask/train/mask_{0:03d}.jpg".format(count))
            # print("saved mask_{}.jpg".format(count))
    # else :
width = 512
height= 512
# count = 0
# mask_width = 32
# mask = Image.new('L', (width,height), color ='black')
# mask_img = Image.new('L',(mask_width,512),color ='white')

# center = int(256 - 16)
# mask.paste(mask_img,(center,0))
# mask.save("/database/resize_panorama/mask_512/task1/mask_{0:03d}.png".format(count))

def generate_mask_dataset():
    # base_dir = '/database/resize_panorama/mask_512'
    for i in range(0,2000):
        # mask_width = 8
        # random_x = random.randrange(0,960,2)
        
        mask = Image.new('L', (width,height), color ='black')
        # masked = Image.open('/database/resize_panorama/train/864.jpg')
        # masked_w,_ = masked.size
        # mask_img = Image.new('L',(i,height),color ='white')

        # center = int(256 - i/2)
        # print(count)
        # mask.paste(mask_img,(center,0))

        # path = os.path.join(base_dir, 'mask{}'.format(count))
        # os.mkdir(path)
        # mask.save(path+"/mask_{0:03d}.jpg".format(count))
        # masked.paste(mask_img,(int(masked_w/2-i/2),0))
        mask.save("/home/godgang/edge-connect/examples/eval_test/mask/inpaint_mask/mask100/{}.jpg".format(i))

generate_mask_dataset()
