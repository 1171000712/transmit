import  torch
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import random
import torch.nn.functional as F

class OodModel:
    def __init__(self, targer_classes, filepath,size_trigger,t_c_arr,args,device='cpu'):
        self.tc=targer_classes
        self.ood_sets = ImageFolder(root=filepath, transform=transforms.ToTensor())
        self.triggers=[]
        self.device=device
        self.discriminator=None
        self.size=size_trigger
        self.t_c_arr=t_c_arr

        self.gamma=args.gamma
        self.T=args.T
        self.P=args.P
        self.mode=args.oodselect

        self.lr=0.01
        self.epochs=10
        self.batch_size=128
        self.tf=transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4)])
    def pred(self,x,output,task_id):
        self.discriminator.eval()
        m=torch.nn.Sigmoid()
        output=F.softmax(output,1)
        dev=(m(self.discriminator(x)/self.T)).pow(self.P)
        for i in range(len(output)):
            output[i,self.t_c_arr[task_id].index(self.tc[task_id])]+=dev[i][0]*self.gamma
        print(task_id,torch.mean(dev).item())
        return output
    def add_triggers(self,model):
        self.discriminator = resnet18(1)
        self.discriminator=self.discriminator.to(self.device)
        t=False
        if model.training:
            t=True
            model.eval()
        self.discriminator.train()
        task_id=len(self.triggers)
        target=self.tc[task_id]
        loader=torch.utils.data.DataLoader(self.ood_sets, batch_size=self.batch_size,shuffle=False, num_workers=4, pin_memory=True)
        outputs=[]
        preds=[]
        for x,y in loader:
            x=x.to(self.device)
            pred=model(x).detach()
            pred=F.softmax(pred,1)
            preds.append(pred)
            outputs.append(torch.max(pred,1)[1])
        indices=[]
        outputs=torch.cat(outputs)
        preds=torch.cat(preds)
        indexs=outputs!=target
        for i in range(len(indexs)):
            if indexs[i] and outputs[i] in self.t_c_arr[task_id]:
                indices.append(i)
        if self.mode=='sort':
            indices.sort(key=lambda i:preds[i][outputs[i]]-preds[i][target],reverse=True)
        else:
            random.shuffle(indices)
        indices=indices[:self.size]
        X=[]
        Y=[]
        for idx in indices:
            X.append(self.ood_sets[idx][0])
            Y.append(target)
        self.ood_sets.samples = [sample for idx, sample in enumerate(self.ood_sets.samples) if idx not in indices]
        self.ood_sets.targets = [sample for idx, sample in enumerate(self.ood_sets.targets) if idx not in indices]
        self.triggers.append([torch.stack(X).to(self.device),torch.Tensor(Y).to(self.device),torch.stack(X).to(self.device)])
        opt = torch.optim.SGD(self.discriminator.parameters(), lr=self.lr)
        loader = torch.utils.data.DataLoader(self.ood_sets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        bce=torch.nn.BCELoss()
        m = torch.nn.Sigmoid()
        for epoch in range(int(self.epochs)):
            for step, data in enumerate(loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = m(self.discriminator(self.tf(inputs)))
                loss = bce(outputs,labels.unsqueeze(1).float())*len(labels)
                for x,y,_ in self.triggers:
                    outputs = m(self.discriminator(self.tf(x)))
                    loss += bce(outputs,torch.ones_like(y).unsqueeze(1)).float()*len(y)
                loss = loss /(len(labels)+len(self.triggers)*len(y))
                opt.zero_grad()
                loss.backward()
                #print(epoch,'-',step,'loss:',loss.item())
                opt.step()
            self.discriminator.eval()
            for i in range(len(self.triggers)):
                print(epoch,i,torch.mean(m(self.discriminator(self.triggers[i][0]))).item())
            self.discriminator.train()
        self.discriminator.eval()
        if t:
            model.train()
