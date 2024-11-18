import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_inflora import SiNet
from models.vit_inflora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import math

class Test2(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # if self._old_network is not None:
        #     self._old_network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        enabled = set()
        updated = []
        seq_param={}
        numtask=0
        try:
            numtask=self._network.module.numtask
        except:
            numtask=self._network.numtask
        seq_param['lora_B_k']=[0]*numtask
        seq_param['lora_B_v']=[0]*numtask
        seq_param['lora_A_k']=[0]*numtask
        seq_param['lora_A_v']=[0]*numtask
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    enabled.add(name)
                    updated.append(param)
                if "lora_B_k" + "." + str(self._network.module.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    enabled.add(name)
                    updated.append(param)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
                if "lora_A_k" + "." + str(self._network.module.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
                if "lora_A_v" + "." + str(self._network.module.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
                if "lora_B_k" + "." + str(self._network.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
                if "lora_B_v" + "." + str(self._network.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
                if "lora_A_k" + "." + str(self._network.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
                if "lora_A_v" + "." + str(self._network.numtask - 1)+'.' in name:
                    param.requires_grad_(True)
                    updated.append(param)
                    enabled.add(name)
            for i in range(numtask):
                try:
                    if "lora_B_k" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_B_k'][i]=param
                    if "lora_B_v" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_B_v'][i]=param
                    if "lora_A_k" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_A_k'][i]=param
                    if "lora_A_v" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_A_v'][i]=param
                except:
                    if "lora_B_k" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_B_k'][i]=param
                    if "lora_B_v" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_B_v'][i]=param
                    if "lora_A_k" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_A_k'][i]=param
                    if "lora_A_v" + "." + str(i)+'.' in name:
                        param.requires_grad_(True)
                        seq_param['lora_A_v'][i]=param

        
        print(f"Parameters to be updated: {enabled}")
        
        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(updated, momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(updated,lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler,seq_param)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(updated, momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(updated,lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler,seq_param)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler,seq_param):
        #self.run_epoch=2
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                
                for key in seq_param:
                    param_list=seq_param[key]
                    cur_grad=param_list[-1].grad
                    for task_id in range(len(param_list)-1):
                        prev_grad=param_list[task_id].grad
                        cur_grad+=torch.sum((cur_grad * prev_grad) / (prev_grad.norm() ** 2)) * prev_grad
                    param_list[-1].grad=cur_grad

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets//self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
