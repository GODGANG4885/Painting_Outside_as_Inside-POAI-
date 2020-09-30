import lpips
from PIL import Image
import numpy as np
import os
loss_fn = lpips.LPIPS(net='alex')


if __name__ == "__main__":
    refer_dir = '/database/very_long_dataset/tf_dataset_new/test'
    # refer_dir = '/home/godgang/edge-connect/paper/table1/original'
    ours_dir = '/home/godgang/edge-connect/examples/eval_test/result/LRmore_finetune/result/oti_rearrange'
    # ours_dir = '/home/godgang/edge-connect/paper/table1/ours'
    ns_dir = '/home/godgang/NS-Outpainting/logs/done/2/results'
    # stflow_dir = '/home/godgang/edge-connect/paper/table1/structureflow'
    ca_dir = '/home/godgang/generative_inpaintingv1/yh_re'
    # ca_dir = '/home/godgang/edge-connect/paper/table1/ca'
    stflow_dir = '/home/godgang/NS-Outpainting/logs/0817/2/result_structureflow'
    edge_connect= '/home/godgang/edge-connect/examples/edge_connect/result/outpaint'
    refer_list = os.listdir(refer_dir)
    ref_mean = 0
    ours_mean = 0
    stflow_mean= 0
    ca_mean = 0
    ns_mean = 0
    edge_mean = 0
    for t, i in enumerate(refer_list):
        ref_image = os.path.join(refer_dir,i)
        ours_image = os.path.join(ours_dir,i)
        stflow_image = os.path.join(stflow_dir,i)
        ca_image = os.path.join(ca_dir,i)
        ns_image = os.path.join(ns_dir,i)
        edge_image = os.path.join(edge_connect,i)
        # print(Image.open(ref_image).size)
        ref_image = Image.open(ref_image)
        ours_image = Image.open(ours_image)
        stflow_image = Image.open(stflow_image)
        ca_image = Image.open(ca_image)
        ns_image = Image.open(ns_image)
        edge_image = Image.open(edge_image)
        # d = loss_fn.forward(im0,im1)

        ours_mean += float(loss_fn.forward(ref_image,ours_image))
        stflow_mean += float(loss_fn.forward(ref_image,stflow_image))
        ca_mean += float(loss_fn.forward(ref_image,ca_image))
        ns_mean += float(loss_fn.forward(ref_image,ns_image))
        edge_mean += float(loss_fn.forward(ref_image,edge_image))

    print('NIQE of ours image is: {}'.format(ours_mean/(t+1)))
    print('NIQE of norearrange image is: {}'.format(edge_mean/(t+1)))
    print('NIQE of stflow image is: {}'.format(stflow_mean/(t+1))) 
    print('NIQE of ca image is: {}'.format(ca_mean/(t+1)))
    print('NIQE of ns image is: {}'.format(ns_mean/(t+1)))
