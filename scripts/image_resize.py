import cv2
import os
#total count of imgs
# totalImgCount = len(os.listdir(crsPath))/2
crsPath = '/database/panorama_train/test'
savedir ='/database/resize_panorama/test'
#soring files to corresponding arrays
size_count = {}
for (dirname, dirs, files) in os.walk(crsPath):
    for filename in files:

        fullname = os.path.join(dirname,filename)
        image =cv2.imread(fullname,cv2.IMREAD_COLOR)
        reshape_image = cv2.resize(image,(1920,256))
        savepath = os.path.join(savedir,filename)
        if savepath[-4:] =='jfif' or savepath[-4:] =='jpeg':
            print(savepath)
            savepath = savepath[:-4] + 'jpg'
        elif savepath[-3:] =='png' or savepath[-3:]=='tif':
            print(savepath)
            savepath = savepath[:-3] + 'jpg'
        # print(savepath[-3:])
        cv2.imwrite(savepath,reshape_image)

        # if shape != 1920 :
        #     print(filename)
        #     os.remove(fullname)

        # if not shape in size_count:
        #     size_count[shape] = 1
        # else:
        #     size_count[shape] +=1
    # print(size_count)
