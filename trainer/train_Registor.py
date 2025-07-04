#!/usr/bin/python3
import os
from torch.utils.data import DataLoader
import torch
from utils.utils import ToTensor, Resize, Logger, smooth_loss, cal_dice
from .Reg_datasets import ImageDataset, TestDataset
from model.Eva_model import Evaluator
from model.Def_model import unet
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import measure
from utils.deformation import Transformer2D
from build_table import *
from utils.utils import smooth_loss, MIND, MI, NCC,neg_Jdet_loss,HD,jacobian_determinant
from thop import profile


def four_point_error(tp_pre, four_point):
    tp_pre = tp_pre.reshape(-1, 2, 3)
    m = torch.tensor([[[0, 0, 1]]], dtype=torch.float).cuda()
    matrix = torch.cat([tp_pre, m], dim=1).squeeze(0).cpu().numpy()

    T = np.array([[2 / 256, 0, -1],
                  [0, 2 / 256, -1],
                  [0, 0, 1]])
    matrix_warp = np.linalg.inv(T) @ np.linalg.inv(matrix) @ T

    new_four_point = cv2.transform(four_point.cpu().numpy(), matrix_warp[:-1,:])
 
    four_corners = np.array([[0, 0], [0, 255], [255, 0], [255, 255]], dtype=np.float32).reshape(-1, 1, 2)
    rmse = np.sum(np.sqrt(np.sum((new_four_point - four_corners) ** 2, axis=2))) / 4
    return rmse


class Reg_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        # warp operation
        self.trans = Transformer2D().cuda()
        # registration network
        self.net_R = unet().cuda()
        self.optimizer_R = torch.optim.Adam(self.net_R.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        # MPGNet
        # self.MPGNet = Evaluator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
        # self.MPGNet.load_state_dict(torch.load(self.config['MPGNet_root']))
        # loss
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
    
        if config['mode'] == "PME":
            self.net_E = Evaluator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
            self.net_E.load_state_dict(torch.load(self.config['PME_root']))
        else:
            if config['sim'] == "MIND":
                self.sim_loss = MIND().loss
            elif config['sim'] == "NCC":
                self.sim_loss = NCC().loss
            elif config['sim'] == "MI":
                self.sim_loss = MI().loss
            else:
                self.sim_loss = torch.nn.L1Loss()

       
        # load train data
        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], opt=config),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last = True)
        
        # load test data
        self.testdataloader = DataLoader(
            TestDataset(config['testroot'], opt=config),
            batch_size=1, shuffle=False, num_workers=config['n_cpu'])
        
        # self.inferdataloader = DataLoader(
        #     InferDataset(config['testroot'], transforms_=self.transforms_1, opt=config),
        #     batch_size=1, shuffle=False, num_workers=0)


        # train/test log
        self.train_logger = Logger(config['name'] + '_' + config['mode'], config['port'], config['n_epochs'],len(self.dataloader))
        self.test_logger = Logger(config['name'] + '_' + config['mode'], config['port'], config['n_epochs'],len(self.testdataloader))


    # train and test
    def train(self):

        lowest_point_loss= 500

        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            total_Eva_loss = total_tp_loss = total_guild_loss = 0

            # Training
            self.net_R.train()
            for item_opt, item_sar, item_opt_1, item_sar_1, gt_tp in self.dataloader:

                item_opt = item_opt.cuda()
                item_opt_1 = item_opt_1.cuda()
                item_sar = item_sar.cuda()
                item_sar_1 = item_sar_1.cuda()   
                gt_tp = gt_tp.reshape(-1, 6).cuda()

                self.optimizer_R.zero_grad()
                # with torch.no_grad():
                #     GT_MD = self.MPGNet(torch.cat([item_sar_1, item_opt_1], dim=1))
                pre_tp ,Pre_MD, fake_opt = self.net_R(item_sar_1, item_opt)

                sar_reg = self.trans(item_sar_1, pre_tp)
                sar_gt_reg = self.trans(item_sar_1, gt_tp)
                GT_MD = ((item_opt_1+1) / 2) - ((item_sar_1 + 1) / 2)
                loss_tp = self.l1_loss(pre_tp, gt_tp)
                loss_guild = self.l1_loss(Pre_MD, GT_MD)
                loss_rec = self.l1_loss(fake_opt, item_opt_1)

                if self.config['mode'] == "PME":
                    with torch.no_grad():
                        error_map = self.net_E(torch.cat([sar_reg, item_opt], dim=1))
                    error_map = torch.abs(error_map)
                    loss_sim = torch.mean(error_map)
                else:
                    loss_sim = self.sim_loss(sar_reg, item_opt)

                total_Eva_loss = total_Eva_loss + loss_sim
                total_tp_loss = total_tp_loss + loss_tp
                total_guild_loss = total_guild_loss + loss_guild

                # backward               
                loss_reg = 10*loss_tp + 5*loss_sim + 2.5*(loss_rec + loss_guild)
                loss_reg.backward()
                self.optimizer_R.step()

                self.train_logger.log({'L_Sim': loss_sim, 'loss_tp': loss_tp},
                images={'item_opt': item_opt, 'item_sar_1': item_sar_1, 'fakeopt': fake_opt, 'GT_MD': GT_MD, 'Pre_MD': Pre_MD, 'sar_reg': sar_reg, "sar_gt_reg": sar_gt_reg})
                
            train_loss_record(self.config, epoch=epoch, tp_loss=total_tp_loss/(len(self.dataloader)), PME_loss=total_Eva_loss/(len(self.dataloader)), lr=self.optimizer_R.param_groups[0]['lr'])

            # testing
            if epoch < 5 or epoch % 50 == 0 or epoch > 1100:
                self.net_R.eval()
                total_concor_loss = 0
                total_l1_loss = 0
                # p = 0
                # p1 = 0
                # p1_5 = 0
                # p2 = 0
                # p2_5 = 0
                # p3 = 0
                # p3_5 = 0

                for item_opt, item_sar, item_opt_1, item_sar_1, gt_tp, four_point in self.testdataloader:

                    item_opt = item_opt.cuda()
                    item_opt_1 = item_opt_1.cuda()
                    item_sar = item_sar.cuda()
                    item_sar_1 = item_sar_1.cuda()   
                    gt_tp = gt_tp.reshape(-1, 6).cuda()
                    four_point = four_point.cuda().squeeze(0)
  
                    with torch.no_grad():
                        pre_tp, Pre_MD, fake_opt= self.net_R(item_sar_1, item_opt)

                        sar_reg = self.trans(item_sar_1, pre_tp)
                        sar_gt_reg = self.trans(item_sar_1, gt_tp)
                
                        l1_loss = self.l1_loss((sar_reg+1)*127.5, (sar_gt_reg+1)*127.5)
                        avg_concor_loss = four_point_error(pre_tp, four_point)
                        # if avg_concor_loss <= 1:
                        #     p1 = p1 + 1
                        # elif avg_concor_loss <= 1.5:
                        #     p1_5 = p1_5 + 1
                        # elif avg_concor_loss <= 2:
                        #     p2 = p2 + 1
                        # elif avg_concor_loss <= 2.5:
                        #     p2_5 = p2_5 + 1
                        # elif avg_concor_loss <= 3:
                        #     p3 = p3 + 1
                        # elif avg_concor_loss <= 3.5:
                        #     p3_5 = p3_5 + 1
                        # else:
                        #     p = p + 1
             
                        total_concor_loss = total_concor_loss + avg_concor_loss
                        total_l1_loss = total_l1_loss + l1_loss
                
                    self.test_logger.log({'L_point': avg_concor_loss},
                    images={'item_opt_t': item_opt, 'item_sar_1_t': item_sar_1, 'fakeopt':fake_opt, 'sar_reg_t': sar_reg, "sar_gt_reg_t": sar_gt_reg})

                each_pixel_loss = total_concor_loss / len(self.testdataloader)
                each_l1_loss = total_l1_loss / len(self.testdataloader)

                test_loss_record(self.config, epoch=epoch, pixel_error=each_pixel_loss, l1_loss=each_l1_loss)

                if each_pixel_loss <= lowest_point_loss:
                    lowest_point_loss = each_pixel_loss
                    if not os.path.exists(self.config["save_root"]):
                        os.makedirs(self.config["save_root"])
                    torch.save(self.net_R.state_dict(), self.config['save_root'] + 'Reg_OS_Rigid_PME_%d.pth' % epoch)
                    result= {'avg_pixel_error':str(lowest_point_loss), 'avg_mse_error':str(each_l1_loss), "detal":'10,5,2.5' ,'loss':'10*loss_tp + 5*loss_sim + 2.5*(loss_rec + loss_guild)'}
                    print('########EPOCH:%d 权值保存成功#######'% epoch)

        method_dict={'method':'RegNet_OS_Rigid','id':0,'batch_size':self.config['batchSize'],'epoch':self.config['n_epochs']}
        set_result(result=result, method_dict=method_dict)  
        print('train over, the lowest piexl error : %f' % lowest_point_loss)
        
    def test(self):

        # self.net_R.load_state_dict(torch.load('/root/autodl-tmp/NMME/Eva/output/Eva_ablation/net_R_Eva_1140.pth'))
        self.net_R.eval()

 
        for item_opt, item_sar, _,_,_,_ in self.testdataloader:
            item_opt = item_opt.cuda()  
            item_sar = item_sar.cuda()
   
            with torch.no_grad():
         
                flops, params = profile(self.net_R, (item_sar, item_opt))
                print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
                break

            #     pre_tp_s2o, fakeopt = self.net_R(item_sar, item_opt)
            #     sar_reg = self.trans(item_sar, pre_tp_s2o)

            # cv2.imwrite("/root/autodl-tmp/dataset/infer/reg_img/"+ num[0] + '.jpg', (sar_reg+1).squeeze(0).squeeze(0).cpu().numpy()*127.5)
            # cv2.imwrite("/root/autodl-tmp/dataset/infer/fake_img/"+ num[0] + '.jpg', (fakeopt+1).squeeze(0).squeeze(0).cpu().numpy()*127.5)
            



# total_time = 0
#     count = 0
#     with torch.no_grad():
#         for index, batch in enumerate(testloader):
#             count += 1
#             images, labels, depths, size, size_label, name = batch
#             name = name[0]
#             images = Variable(images).cuda()
#             depths = Variable(depths).cuda()
#             flops, params = profile(model, (images,depths))
#             print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
#             break
 # 486296.95 M, params: 213.56 M

#     torch.cuda.synchronize()
 #     start = time.time()
 #     result = model(images, depths)
 #     torch.cuda.synchronize()
 #     end = time.time()
 #     batch_time = end-start
 #     total_time += batch_time
 #     print('infer_time:', end-start)
 
 # avg_time = total_time / count
 # print('avg_infer_time:', avg_time) 
 # #avg_infer_time: 0.13393752883981774




        












def train_loss_record(config, epoch, tp_loss, PME_loss, lr):
    os.makedirs(config['train_log_root'], exist_ok=True)
    log_path = os.path.join(config['train_log_root'], 'train_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: TP:%.6f | EVA:%.4f | LR:%.6f \n' % (epoch, tp_loss, PME_loss,  lr))

def test_loss_record(config, epoch, pixel_error, l1_loss):
    os.makedirs(config['test_log_root'], exist_ok=True)
    log_path = os.path.join(config['test_log_root'], 'test_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: pixel_error:%.6f | l1_loss:%.6f \n' % (epoch, pixel_error, l1_loss))