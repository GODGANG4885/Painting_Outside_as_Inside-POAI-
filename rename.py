import os

path = '/database/beach_image/'
save_path = '/database/beach'
i = 0
for filename in os.listdir(path):
    
    # print(path+filename, '=>', path+str(cName)+str(i)+'.jpg')
    os.rename(path+filename, path+'{}.jpg'.format(i))
    i +=1
