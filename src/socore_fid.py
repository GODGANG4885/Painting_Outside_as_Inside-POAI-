from score_infinity import calculate_FID_infinity, calculate_IS_infinity

generator = '/home/godgang/edge-connect/examples/eval_test/result/hinge_LRmore/InpaintingModel_gen.pth'

FID_infinity = calculate_FID_infinity(generator, 64, 32, gt_path='/home/godgang/NS-Outpainting/logs/0817/2/origval/statistics.npz')
IS_infinity = calculate_IS_infinity(generator, 64, 32)

print("FID_infinity : {} / IS_infinity : {}".format(FID_infinity,IS_infinity))