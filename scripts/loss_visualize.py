import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def EdgeLossVisualize(filename):
    indata1 = np.loadtxt(filename, usecols=(0,1,2,3,4,5,6)) # make sure the rest is ignored
    z = indata1[:,0]
    x = indata1[:,1]
    D = indata1[:,2]
    G = indata1[:,3]
    FM = indata1[:,4]
    pre = indata1[:,5]
    rec = indata1[:,6]
    start_term = 0
    end_term = len(x)

    plt.figure(figsize=(20,10))
    plt.subplot(241)
    plt.plot(x[start_term:end_term],D[start_term:end_term])
    plt.title("D loss")

    plt.subplot(242)
    plt.plot(x[start_term:end_term],G[start_term:end_term])
    plt.title("G loss")

    plt.subplot(243)
    plt.plot(x[start_term:end_term],FM[start_term:end_term])
    plt.title("FM loss")

    plt.subplot(244)
    plt.plot(x[start_term:end_term],pre[start_term:end_term])
    plt.title("precision")

    plt.subplot(245)
    plt.plot(x[start_term:end_term],rec[start_term:end_term])
    plt.title("recall")

    #  '''filename = /mode_dir/folder_dir/config'''
    base_dir = os.path.dirname(filename)
    config=os.path.dirname(base_dir)
    folder_dir=os.path.dirname(config)

    folder_name=os.path.basename(config)

    mode_dir = os.path.dirname(folder_dir)
    mode_name = os.path.basename(mode_dir)
    save_path = '/home/godgang/edge-connect/loss_visualize'+'/{}.jpg'.format(mode_name+'_'+folder_name)
    plt.savefig(save_path)
    print("saved file {}".format(save_path))

def InpaintLossVisualize(filename):
    
    indata = np.loadtxt(filename, usecols=(0,1,2,3,4,5,6,7,8)) # make sure the rest is ignored
    z = indata[:,0]
    x = indata[:,1]/10
    d2 = indata[:,2]
    g2 = indata[:,3]
    l1 = indata[:,4]
    per = indata[:,5]
    sty = indata[:,6]
    psnr = indata[:,7]
    mae = indata[:,8]
    start_term = 0
    end_term = len(d2)
    c= 0
    # for i in range(0,int(len(x)/10),10):
    #     z[c] = np.mean(indata[i:i+10,0])
    #     x[c] = np.mean(indata[i:i+10,1])
    #     d2[c] = np.mean(indata[i:i+10,2])
    #     g2[c] = np.mean(indata[i:i+10,3])
    #     l1[c] = np.mean(indata[i:i+10,4])
    #     per[c] = np.mean(indata[i:i+10,5])
    #     sty[c] = np.mean(indata[i:i+10,6])
    #     psnr[c] = np.mean(indata[i:i+10,7])
    #     mae[c] = np.mean(indata[i:i+10,8])
    #     c+=1


    plt.figure(figsize=(20,10))
    plt.subplot(241)
    plt.plot(x[start_term:end_term],d2[start_term:end_term])
    plt.title("D2 loss")

    plt.subplot(242)
    plt.plot(x[start_term:end_term],g2[start_term:end_term])
    plt.title("G2 loss")

    plt.subplot(243)
    plt.plot(x[start_term:end_term],l1[start_term:end_term])
    plt.title("L1 loss")

    plt.subplot(244)
    plt.plot(x[start_term:end_term],per[start_term:end_term])
    plt.title("perceptual loss")

    plt.subplot(245)
    plt.plot(x[start_term:end_term],sty[start_term:end_term])
    plt.title("style loss")

    plt.subplot(246)
    plt.plot(x[start_term:end_term],psnr[start_term:end_term])
    plt.title("PSNR")

    plt.subplot(247)
    plt.plot(x[start_term:end_term],mae[start_term:end_term])
    plt.title("MAE")
    
    # '''filename = /mode_dir/folder_dir/config'''
    base_dir = os.path.dirname(filename)
    config=os.path.dirname(base_dir)
    folder_dir=os.path.dirname(config)

    folder_name=os.path.basename(config)

    mode_dir = os.path.dirname(folder_dir)
    mode_name = os.path.basename(mode_dir)
    save_path = '/home/godgang/edge-connect/loss_visualize'+'/{}.jpg'.format(mode_name+'_'+folder_name)
    plt.savefig(save_path)
    print("saved file {}".format(save_path))

def CompareLoss_inpaint(path1,path2):
    indata = np.loadtxt(path1, usecols=(0,1,2,3,4,5,6,7,8)) # make sure the rest is ignored
    indata2 = np.loadtxt(path2, usecols=(0,1,2,3,4,5,6,7,8)) # make sure the rest is ignored
    count = 1
    start_term = 0
    end_term = 32000
    end_term2 = 32000

    z_1 = indata[start_term:end_term,0][0::count]
    x_1 = indata[start_term:end_term,1][0::count]
    d2_1 = indata[start_term:end_term,2][0::count]
    g2_1 = indata[start_term:end_term,3][0::count]
    l1_1 = indata[start_term:end_term,4][0::count]
    per_1 = indata[start_term:end_term,5][0::count]
    sty_1 = indata[start_term:end_term,6][0::count]
    psnr_1 = indata[start_term:end_term,7][0::count]
    mae_1 = indata[start_term:end_term,8][0::count]
    f1_1 = 2 * per_1 * sty_1 / (per_1 + sty_1 + 1e-5)

    z_2 = indata2[start_term:end_term,0][0::count]
    x_2 = indata2[start_term:end_term,1][0::count]
    d2_2 = indata2[start_term:end_term,2][0::count]
    g2_2 = indata2[start_term:end_term,3][0::count]
    l1_2 = indata2[start_term:end_term,4][0::count]
    per_2 = indata2[start_term:end_term,5][0::count]
    sty_2 = indata2[start_term:end_term,6][0::count]
    psnr_2 = indata2[start_term:end_term,7][0::count]
    mae_2 = indata2[start_term:end_term,8][0::count]
    
    # end_term = 16000

    plt.figure(figsize=(20,10))
    plt.subplot(241)
    plt.plot(x_1,d2_1,color='red')
    plt.plot(x_2,d2_2,alpha=0.7,color='green')
    plt.title("D2 loss")

    plt.subplot(242)
    plt.plot(x_1,g2_1,color='red')
    plt.plot(x_2,g2_2,alpha=0.7,color='green')
    plt.title("G2 loss")

    plt.subplot(243)
    plt.plot(x_1,l1_1,color='red')
    plt.plot(x_2,l1_2,alpha=0.7,color='green')
    plt.title("L1 loss")

    plt.subplot(244)
    plt.plot(x_1,per_1,color='red')
    plt.plot(x_2,per_2,alpha=0.7,color='green')
    plt.title("perceptual loss")

    plt.subplot(245)
    plt.plot(x_1,sty_1,color='red')
    plt.plot(x_2,sty_2,alpha=0.7,color='green')
    plt.title("style loss")

    plt.subplot(246)
    plt.plot(x_1,psnr_1,color='red')
    plt.plot(x_2,psnr_2,alpha=0.7,color='green')
    plt.title("PSNR")

    plt.subplot(247)
    plt.plot(x_1,mae_1,color='red')
    plt.plot(x_2,mae_2,alpha=0.7,color='green')
    plt.title("MAE")
    
    # '''filename = /mode_dir/folder_dir/config'''
    base_dir = os.path.dirname(path1)
    config=os.path.dirname(base_dir)
    folder_dir=os.path.dirname(config)
    folder_name=os.path.basename(config)

    base_dir2 = os.path.dirname(path2)
    config2=os.path.dirname(base_dir2)
    # folder_dir2=os.path.dirname(config2)
    folder_name2=os.path.basename(config2)

    mode_dir = os.path.dirname(folder_dir)
    mode_name = os.path.basename(mode_dir)
    save_path = '/home/godgang/edge-connect/loss_visualize'+'/{}.jpg'.format(mode_name+'_'+folder_name+'_VS_'+folder_name2)
    plt.savefig(save_path)
    print("saved file {}".format(save_path))
def CompareLoss_edge(path1,path2):
    
    indata = np.loadtxt(path1, usecols=(0,1,2,3,4,5,6)) # make sure the rest is ignored
    indata2 = np.loadtxt(path2, usecols=(0,1,2,3,4,5,6)) # make sure the rest is ignored
    count= 20
    start_term = 0
    # start_term2 = 32000
    end_term = 800
    end_term2 = 1000
    z_1 = indata[start_term:end_term,0][0::count]
    x_1 = indata[start_term:end_term,1][0::count]
    d2_1 = indata[start_term:end_term,2][0::count]
    g2_1 = indata[start_term:end_term,3][0::count]
    l1_1 = indata[start_term:end_term,4][0::count]
    per_1 = indata[start_term:end_term,5][0::count]
    sty_1 = indata[start_term:end_term,6][0::count]
    
    f1_1 = 2 * per_1 * sty_1 / (per_1 + sty_1 + 1e-5)

    z_2 = indata2[start_term:end_term,0][0::count]
    x_2 = indata2[start_term:end_term,1][0::count]
    d2_2 = indata2[start_term:end_term,2][0::count]
    g2_2 = indata2[start_term:end_term,3][0::count]
    l1_2 = indata2[start_term:end_term,4][0::count]
    per_2 = indata2[start_term:end_term,5][0::count]
    sty_2 = indata2[start_term:end_term,6][0::count]

    f1_2 = 2 * per_2 * sty_2 / (per_2 + sty_2 + 1e-5)

    # end_term = len(x_2)
    # end_term2 = len(x_2)
    # end_term = 16000
    # plt.plot(x_1, f1_1,alpha=0.7, color="red", linestyle="-", markersize=1)
    # plt.plot(x_2, f1_2, color='green', linestyle="-",  markersize=1)
    # plt.ylim((0,1))
    # plt.grid()
    # Add Trend Line
    # if len(x_1) > 1:
    #     z_1= np.polyfit(x_1, g2_1+1, min(50, len(x_1)-1))
    #     z_2 = np.polyfit(x_2, g2_2, min(50, len(x_1)-1))
    #     p_1 = np.poly1d(z_1)
    #     p_2 = np.poly1d(z_2)
    #     plt.plot(x_1, p_1(x_1), color="green", linestyle="--")
    #     plt.plot(x_1, p_2(x_1), color="red", linestyle="--")
    # plt.ylabel('loss')
    # plt.xlabel('iteration')
    # plt.legend(['nsgan loss', 'hinge loss'], loc='lower left')
    # plt.title("f1 score")
 
    plt.figure(figsize=(20,10))
    plt.subplot(241)
    plt.plot(x_1,d2_1,color='red')
    plt.plot(x_2,d2_2,alpha=0.7,color='green')
    plt.title("D1 loss")

    plt.subplot(242)
    plt.plot(x_1,g2_1,color='red')
    plt.plot(x_2,g2_2,alpha=0.7,color='green')
    plt.title("G1 loss")

    plt.subplot(243)
    plt.plot(x_1,l1_1,color='red')
    plt.plot(x_2,l1_2,alpha=0.7,color='green')
    plt.title("FM loss")

    plt.subplot(244)
    plt.plot(x_1,per_1,color='red')
    plt.plot(x_2,per_2,alpha=0.7,color='green')
    plt.title("precision")

    plt.subplot(245)
    plt.plot(x_1,sty_1,color='red')
    plt.plot(x_2,sty_2,alpha=0.7,color='green')
    plt.title("recall")

    
    # '''filename = /mode_dir/folder_dir/config'''
    base_dir = os.path.dirname(path1)
    config=os.path.dirname(base_dir)
    folder_dir=os.path.dirname(config)
    folder_name=os.path.basename(config)

    base_dir2 = os.path.dirname(path2)
    config2=os.path.dirname(base_dir2)
    # folder_dir2=os.path.dirname(config2)
    folder_name2=os.path.basename(config2)

    mode_dir = os.path.dirname(folder_dir)
    mode_name = os.path.basename(mode_dir)
    save_path = '/home/godgang/edge-connect/loss_visualize'+'/{}.jpg'.format(mode_name+'_'+folder_name+'_VS_'+folder_name2)
    plt.savefig(save_path)
    print("saved file {}".format(save_path))
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='edge or inpaint')
    parser.add_argument('--path', type=str, help='model checkpoints path')
    parser.add_argument('--path2', type=str, default=None, help='model2 checkpoints path')
    parser.add_argument('--compare', default = None, dest='compare', help='Display online tracker output (slow) [False]',action='store_true')
    opt = parser.parse_known_args()[0]

    if opt.compare is None:
        
        if opt.mode =='edge':
            EdgeLossVisualize(os.path.join(opt.path,'config/log_edge.dat'))
        elif opt.mode =='inpaint':
            InpaintLossVisualize(os.path.join(opt.path,'config/log_inpaint.dat'))
        else :
            print("check path")
    else:
        if opt.path2 is None:
            print("check path2")
        else:
            if opt.mode =='edge':
                CompareLoss_edge(os.path.join(opt.path,'config/log_edge.dat'),os.path.join(opt.path2,'config/log_edge.dat'))
            elif opt.mode =='inpaint':
                CompareLoss_inpaint(os.path.join(opt.path,'config/log_inpaint.dat'),os.path.join(opt.path2,'config/log_inpaint.dat'))
            else :
                print("check path")
        