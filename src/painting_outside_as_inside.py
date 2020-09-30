import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave,step1_tempsave,imsave_outpainting,calculate_f1
from .metrics import PSNR, EdgeAccuracy
import torch.nn.functional as F
import cv2
import yaml
class Painting_Outside_Inside():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'
        elif config.MODEL == 5:
            model_name = 'outpaint'

        if self.config.MODE == 2:
            self.test_dataset = Dataset(self.config, self.config.TEST_FLIST, self.config.TEST_EDGE_FLIST, self.config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(self.config, self.config.TRAIN_FLIST, self.config.TRAIN_EDGE_FLIST, self.config.TRAIN_MASK_FLIST, augment=False, training=True)
            self.val_dataset = Dataset(self.config, self.config.VAL_FLIST, self.config.VAL_EDGE_FLIST, self.config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)


        # self.mask_folder_counter = 3
        self.base= '/database/resize_panorama/mask_512'
        # self.mask_folder = self.base+'/mask{}'.format(self.mask_folder_counter)
        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        
        if self.config.MODEL == 1:
            self.edge_model.load()
        elif self.config.MODEL == 2:
            self.inpaint_model.load()
        else:
            self.edge_model.load()
            self.inpaint_model.load()


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()
    def individual_save(self):
        if self.config.MODEL == 1:
            self.edge_model.partial_save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.partial_save()

        else:
            self.edge_model.partial_save()
            self.inpaint_model.partial_save()
    
    def train(self):

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=False
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        # else :
        #     for i in self.train_dataset:
        #         print("print train_dataset {}".format(i))
        # self.mask_folder_counter = 3
        print("init eval!!")
        
        f1 = self.eval()
        plateu_count = 0
        lr_dic_count=0
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            iteration = self.edge_model.iteration
            
            # self.mask_folder_counter = iteration // 20000 +1
            # self.mask_folder = self.base+'/mask{}'.format(self.mask_folder_counter)   
            print("mask path is : {}".format(self.config.TRAIN_MASK_FLIST))
            print("total is : {}".format(total))

            train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=False
            )
            for items in train_loader:
                self.edge_model.train()
                self.inpaint_model.train()

                images, images_gray, edges, masks = self.cuda(*items)
                
            
                # edge model
                if model == 1:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)
                   
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))

                    # backward
                    self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration
                    

                # inpaint model
                elif model == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                    

                # inpaint with edge model
                elif model == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, edges, masks)
                        outputs = outputs * masks + edges * (1 - masks)
                    else:
                        outputs = edges

                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration
                    

                # joint model
                elif model == 4:
                    # train
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration
                
               

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                # if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    
                #     self.sample()

                # evaluate model at checkpoints
                # if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
            
            # print('\nstart eval...\n')
            # if epoch %10 ==0:
            cal_eval = self.eval()
            if cal_eval> f1:
                f1 = cal_eval
                plateu_count=0
                self.sample(epoch)
                self.save()
                self.individual_save()
            else:
                plateu_count +=1
            # self.sample(epoch)
            # self.save()
            # self.individual_save()

            # if plateu_count>10 and lr_dic_count<4:
            if plateu_count>10 or epoch %100 ==0:
                break
            # if lr_dic_count<4:
            #     with open(self.config.NAME) as f:
            #         list_doc = yaml.load(f)

            #     for config_lr in list_doc:
            #         config_lr["LR"] = config_lr["LR"]*0.5

            #     with open(self.config.NAME, "w") as f:
            #         yaml.dump(list_doc, f)
            #     lr_dic_count+=1
            #     print("learning rate dicrese! X0.5")
            # elif plateu_count<10:
            #     continue
            # else:
            #     break

                # save model at checkpoints
                # if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                #     self.save()

                # if iteration >= max_iteration or iteration % 10000==0:
                # if iteration >= max_iteration :
                    # keep_training = False
                    # print("break!!")
                    # break

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=False
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        print_log = [0,0]
        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))


            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # inpaint with edge model
            elif model == 3:
                # eval
                outputs = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, outputs.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))


            # joint model
            else:
                # eval
                e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs
            logs = [("it", iteration), ] + logs
            # progbar.add(len(images), values=logs)
            if model==1:
                print_log[0]+=precision.item()
                print_log[1]+=recall.item()
            else: 
                print_log[0]+=psnr.item()
                print_log[1]+=mae.item()
            
        if model==1:
            print("precision: {}, recall: {}".format( print_log[0]/iteration,print_log[1]/iteration))
            f1 = calculate_f1(print_log[0]/iteration, print_log[1]/iteration)
            print("f1 score: {}".format(f1))
            return f1
        else:
            print("psnr: {}, mae: {}".format( print_log[0]/iteration,print_log[1]/iteration))
            return print_log[0]/iteration
        
            # progbar.add(len(images), values=logs)

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # outpaint
            elif model == 5:
                # print(images.size)
                # print(images_gray.ndim)
                # print(edges.ndim)
                # print(masks.ndim)
                # images, images_gray, edges, masks = self.half_mirror(images), self.half_mirror(images_gray), self.half_mirror(edges), self.half_mirror(masks)
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # inputs =self.half_mirror(inputs)
                # outputs= self.half_mirror(outputs)
                # edges=self.half_mirror(edges)
                # outputs_merged =self.half_mirror(outputs_merged)
            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
            if model == 5:
                output = self.postprocess(outputs_merged)[0]
                output = self.half_mirror(output)
            else:
                output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            # imsave_outpainting(output, path)
            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
                
            # def genrator1(image):
            #     return self.inpaint_model(image, edges, masks)

        print('\nEnd test....')

    def multi_test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
           
            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)
            temp_save = step1_tempsave(images)
            imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
            # def genrator1(image):
            #     return self.inpaint_model(image, edges, masks)

            
        print('\nEnd test....')
   


    def half_mirror(self,image):
        
        # print(image.ndim)
        if image.ndim == 3:
            _,w,_ = image.shape
            half = int(w/2)
            img_left = image[ :, :half, :]
            img_right = image[ :, half:, :]
            arg_image = cv2.hconcat([img_right, img_left])
        else :
            # _,w = image.shape
            w=256
            half = int(w/2)
            img_left = image[ :, :half]
            img_right = image[ :, half:]
            arg_image = cv2.hconcat([img_right, img_left])

        return arg_image
    def sample(self,epoch, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return
        print(self.val_dataset.mask_data)
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)


        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # outpainting with edge-connect
        elif model == 5:
            images, images_gray, edges, masks = self.half_mirror(images), self.half_mirror(images_gray), self.half_mirror(edges), self.half_mirror(masks)
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            inputs =self.half_mirror(inputs)
            outputs= self.half_mirror(outputs)
            edges=self.half_mirror(edges)
            outputs_merged =self.half_mirror(outputs_merged)
        # inpaint with edge model / joint model 
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        print("sample images shape {}".format(images.shape))
        print("sample merged shape {}".format(outputs_merged.shape))
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            # self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(self.config.STAGE)+"_epcoh_" + str(epoch).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
