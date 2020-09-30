import sys
sys.path.append('/home/godgang/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model/Python/libsvm/python')
from brisquequality import test_measure_BRISQUE
import os

if __name__ == '__main__':
    origin_image = "/home/godgang/edge-connect/examples/eval_test/result/LRmore_finetune/result/oti_rearrange"
    file_list = os.listdir(origin_image)
    total_sum =0
    for t, i in enumerrate(file_list):
        total_sum += test_measure_BRISQUE(i)
        print(total_sum//t)
        