import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.core import SupervisedPlugin
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
class TriggerP(SupervisedPlugin):

    def __init__(self,trainstream,teststream,num,mode='segregate',train_type='base'):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        print(num,mode,train_type)
        self.teststream=teststream
        self.srcstream=[]
        self.srgstream=[]
        self.Xs=[]
        self.Ys=[]
        for train_exp in trainstream:
          maxlen=(len(train_exp.dataset)-num)//2
          indexs=[i for i in range(len(train_exp.dataset))]
          random.shuffle(indexs)
          srcset=train_exp.dataset.subset(indexs[:maxlen])
          srgset=train_exp.dataset.subset(indexs[maxlen:maxlen*2])
          trgset=train_exp.dataset.subset(indexs[maxlen*2:])
          tf=transforms.ToTensor()
          trgset.replace_current_transform_group(tf)
          srgset.replace_current_transform_group(tf)
          self.srgstream.append(srgset)
          train_exp.dataset=srcset
          self.srcstream.append(train_exp)
          loader=torch.utils.data.DataLoader(trgset, batch_size=num,shuffle=False, num_workers=2)
          for x,y,_ in loader:
            self.Xs.append(x.cuda())
            ty=[]
            pool=y.unique().numpy().tolist()
            for label in y:
              tmppool=deepcopy(pool)
              tmppool.remove(label)
              ty.append(tmppool[random.randint(0,len(pool)-2)])
            self.Ys.append(torch.tensor(ty).cuda())
        self.X=self.Xs[0]
        self.Y=self.Ys[0]
        self.ts=[]
        self.cnt=-1
        self.mode=mode
        self.train_type=train_type
    def get_train_stream(self):
      return self.srcstream
    def update(self,strategy):
      flag=False
      for t in strategy.mbatch[2].unique():
        if t.item() not in self.ts:
          self.ts.append(t.item())
          flag=True
      if flag:
        self.cnt+=1
        if self.mode=='segregate':
          self.X=self.Xs[self.cnt]
          self.Y=self.Ys[self.cnt]
        elif self.mode=='merge':
          if self.cnt>0:
            self.X=torch.cat((self.X,self.Xs[self.cnt]),dim=0)
            self.Y=torch.cat((self.Y,self.Ys[self.cnt]),dim=0)
    def before_forward(self, strategy, **kwargs):
        model=strategy.model
        self.update(strategy)
        loss=CrossEntropyLoss()
        if self.train_type=='base':
            outputs=model(self.X)
        elif self.train_type=='pgd':
            model.eval()
            query = self.X.detach().clone()
            query.requires_grad_(True)
            for _ in range(0):
                    query_preds = model(query)
                    query_loss = F.cross_entropy(query_preds, self.Y)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = torch.clamp(query,0,1).detach().clone()
                    query.requires_grad_(True)
                    model.zero_grad()
            model.train()
            outputs=model(query)
                    
        self.loss=loss(outputs,self.Y)
    def before_backward(self, strategy, **kwargs):
        strategy.loss+=self.loss
        ''' 
        srcmodel=deepcopy(strategy.model)
        srcmodel.eval()
        outputs=torch.max(srcmodel(self.X),dim=1)[1]
        cnt=torch.sum(outputs==self.Y)
        print('Source Trigger Acc',cnt.item()/len(self.Y)) 
        '''
    def test(self, strategy,model,optimizer,epoch,bs):
        srcmodel=deepcopy(strategy.model)
        srcmodel.eval()
        outputs=torch.max(srcmodel(self.X),dim=1)[1]
        cnt=torch.sum(outputs==self.Y)
        print('Source Trigger Acc',cnt.item()/len(self.Y))
        sets=[]
        for exp in self.srgstream:
            sets.append(exp)
        trainset=ConcatDataset(sets)
        loader=torch.utils.data.DataLoader(trainset, batch_size=bs,
                                         shuffle=True, num_workers=2)
        criterion = CrossEntropyLoss()
        model=model.cuda()
        model.train()
        for ep in range(epoch):
            for x,y,_ in loader:
                x,y=x.cuda(),y.cuda()
                preds = model(x)
                teacher = deepcopy(strategy.model)
                teacher.eval()
                teacher_preds = teacher(x)
                T=1
                loss = F.kl_div(F.log_softmax(preds / T, dim=-1), F.softmax(teacher_preds / T, dim=-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        #model=deepcopy(strategy.model)
        #model.eval()
        #outputs=torch.max(model(self.X),dim=1)[1]
        #cnt=torch.sum(outputs==self.Y)
        #print('Trigger Acc',cnt.item()/len(self.Y))
        for i in range(len(self.teststream)):
            x,y=self.X[i].cuda(),self.Y[i].cuda()
            outputs = torch.max(model(x),dim=1)[1]
            cnt+=torch.sum(outputs==y).item()
            su=len(y)
            print(f'Eval Task {i} Trigger Acc',cnt/su)
            i+=1
        i=0
        for exp in self.teststream:
            cnt,su=0,0
            loader=torch.utils.data.DataLoader(exp.dataset, batch_size=32,
                                         shuffle=True, num_workers=2)
            for x,y,_ in loader:
                x,y=x.cuda(),y.cuda()
                outputs = torch.max(model(x),dim=1)[1]
                cnt+=torch.sum(outputs==y).item()
                su+=len(y)
            print(f'Eval Task {i} Acc',cnt/su)
            i+=1
                
