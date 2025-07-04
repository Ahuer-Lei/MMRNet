#!/usr/bin/python3

import os
from torch.utils.data import DataLoader
import torch
from utils.utils import ToTensor, Resize
from utils.utils import Logger
from .Eva_datasets import ImageDataset,TestDataset
from model.Eva_model import Evaluator
import cv2
import numpy as np
from PIL import Image
import SimpleITK as sitk
from torch.autograd import Variable
from build_table import *

    

class Eva_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        # def networks
        self.net_E = Evaluator(config['input_nc'], config['output_nc'], ndims=config['dim']).cuda()
        self.optimizer_E = torch.optim.Adam(self.net_E.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        self.loss = torch.nn.L1Loss()


        self.dataloader = DataLoader(
            ImageDataset(config['dataroot'], opt=config),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last = True)
        
        self.testdataloader = DataLoader(
            TestDataset(config['testroot'], opt=config),
            shuffle=False, batch_size=config['batchSize'], num_workers=0)

        self.train_logger = Logger(config['name'], config['port'], config['n_epochs'] - config['epoch'], len(self.dataloader))
        self.test_logger = Logger(config['name'], config['port'], config['n_epochs'] - config['epoch'], len(self.testdataloader))


    def train(self):
        lowest_l1 = 1000
        
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            total_l1_loss = 0
            # Training
            self.net_E.train()
            for item_opt_2, item_opt_1, item_sar_1, Label in self.dataloader:

                item_opt_2 = item_opt_2.cuda() # [b 1 256 256]
                item_opt_1 = item_opt_1.cuda() # [b 1 256 256]
                item_sar_1 = item_sar_1.cuda() # [b 1 256 256]
                Label = Label.cuda()           # [b 1 256 256]
                
                self.optimizer_E.zero_grad()
                
                inputs = torch.cat([item_sar_1, item_opt_2], 1)
                pred = self.net_E(inputs)      # [b 1 256 256]

                loss_E = self.loss(pred, Label)
                total_l1_loss = total_l1_loss + loss_E
                loss_E.backward()
                self.optimizer_E.step()

                self.train_logger.log({'L1': loss_E},
                images={'item_opt_2': item_opt_2,'item_opt_1': item_opt_1, 'item_sar_1': item_sar_1, 'Label': Label, 'pred': pred})

            train_loss_record(self.config, epoch=epoch, l1_loss=total_l1_loss/(len(self.dataloader)), lr=self.optimizer_E.param_groups[0]['lr'])   

            # testing
            if epoch < 10 or epoch % 20 == 0 or epoch > 950:
                self.net_E.eval()
                total_l1 = 0
                for item_opt_2, item_sar_1, Label in self.testdataloader:

                    item_opt_2 = item_opt_2.cuda()  # [b 1 256 256]
                    item_sar_1 = item_sar_1.cuda()  # [b 1 256 256]
                    Label = Label.cuda()            # [b 1 256 256]

                    with torch.no_grad():
                        inputs = torch.cat([item_sar_1, item_opt_2], 1)
                        pred = self.net_E(inputs)  

                        l1loss = self.loss(pred, Label)
                        total_l1 = total_l1 + l1loss

                    self.test_logger.log({'L1_t': l1loss},
                    images={'item_opt_2_t': item_opt_2, 'item_sar_1_t': item_sar_1, 'Label_t': Label, 'pred_t': pred})

                avg_l1 = total_l1/len(self.testdataloader)
                test_loss_record(self.config, epoch=epoch, l1loss=avg_l1)
             
                if avg_l1 <= lowest_l1:
                    lowest_l1 = avg_l1
                    if not os.path.exists(self.config["save_root"]):
                        os.makedirs(self.config["save_root"])
                    torch.save(self.net_E.state_dict(), self.config['save_root'] + 'net_%s_osdataset.pth' % self.config['name'])
                    result= {'l1loss':str(l1loss), 'detail':'train osdataset Eva', 'loss':'l1loss'}
                    print('######## EPOCH: %d 权值保存成功########' % epoch)

        method_dict={'method':'Eva_osdataset','id':0,'batch_size':self.config['batchSize'],'epoch':self.config['n_epochs']}
        set_result(result=result,method_dict=method_dict)  
        


def train_loss_record(config, epoch, l1_loss, lr):
    os.makedirs(config['train_log_root'], exist_ok=True)
    log_path = os.path.join(config['train_log_root'], 'train_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: l1_loss:%.6f | LR:%.6f \n' % (epoch, l1_loss, lr))

def test_loss_record(config, epoch, l1loss):
    os.makedirs(config['test_log_root'], exist_ok=True)
    log_path = os.path.join(config['test_log_root'], 'test_log.txt')
    with open(log_path, 'a') as f:
        f.write('No.%d Epoch: l1loss:%.6f \n' % (epoch, l1loss))