"""
"""

from copy import deepcopy

import torch
import lightning as L
from torch import nn, optim
from torchmetrics import F1Score,Accuracy,Recall,Precision,ConfusionMatrix,StatScores,AUROC,AveragePrecision
from peft import get_peft_model

from jj_eval_setup.peft_config import lora_config

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self,dino_model):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dino_model)
        self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

class LoraDinoVisionTransformerClassifier(nn.Module):
    def __init__(self,dino_model,num_classes=2,use_LoRA:bool=False,freeze:bool=False,unfreeze_last:bool=False,unfreeze_norms:bool=False,debug:bool=False):
        super(LoraDinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dino_model)
        #self.transformer = current_dino
        #self.classifier = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(nn.Linear(384, 256),nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, num_classes)) # batch norm added according to paper A Closer Look at Benchmarking Self-Superised Pre-training with Image Classification

        #print(self.transformer.modules)        

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

        if unfreeze_last:
            get_transformer_units = self.transformer.get_submodule("blocks")
            last_idx = len(get_transformer_units) - 1

            if debug:
                print(f"\nlast idx: {last_idx}\n")

            for param in get_transformer_units[last_idx].parameters():
                param.requires_grad = True

            last_transformer_copy = deepcopy(get_transformer_units[last_idx])

        if use_LoRA:
            self.transformer = get_peft_model(self.transformer,lora_config)

        if unfreeze_last:
            #get_transformer_units = self.transformer.get_submodule("blocks")
            #self.transformer.get_submodule("blocks")[:] = self.transformer.get_submodule("blocks")[:-1]
            self.transformer.get_submodule("blocks")[last_idx] = last_transformer_copy

            if debug:
                print(f"\nlast transformer copy:\n{last_transformer_copy}\n")
        
        if unfreeze_norms:
            for module in self.transformer.named_modules():
                if "norm" in module[0]:

                    if debug:
                        print(f"\nm name: {module[0]}, {type(module[1])}\n")

                    for param in module[1].parameters():
                        param.requires_grad = True
        if debug:                
            print(self.transformer.modules) 

    def forward(self, x):
        x = self.transformer(x)
        #x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
    
class LitDINOv2(L.LightningModule):
    def __init__(self,dino_model,args,loss_metric,num_classes,save_path,lr=1e-6,batch_save_every:int=500,metadata:bool=False,debug:bool=False):
        super().__init__()
        #self.save_hyperparameters(ignore="dino_model")

        self.model = LoraDinoVisionTransformerClassifier(dino_model=dino_model,num_classes=num_classes,**args)
        #self.model = DinoVisionTransformerClassifier(dino_model)
        self.loss_metric = loss_metric
        self.num_classes = num_classes
        #self.acc_metric = acc_metric
        # self.f1_metric = F1Score(task='multiclass',num_classes=num_classes,average='weighted')#,ignore_index=0)
        # self.acc_metric = Accuracy(task='multiclass',num_classes=num_classes,average='weighted')
        # self.recall_metric = Recall(task='multiclass',num_classes=num_classes,average='weighted')
        # self.precision_metric = Precision(task='multiclass',num_classes=num_classes,average='weighted')
        # #self.stat_scores_metric = StatScores(task='multiclass',num_classes=num_classes,average='macro')
        # self.confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=num_classes)
        # self.AUROC = AUROC(task='multiclass',num_classes=self.num_classes,average='weighted')
        # self.AP = AveragePrecision(task='multiclass',num_classes=self.num_classes,average='weighted')

        self.f1_metric = F1Score(task='multiclass',num_classes=num_classes,average=None)#,ignore_index=0)
        self.acc_metric = Accuracy(task='multiclass',num_classes=num_classes,average='macro')
        self.recall_metric = Recall(task='multiclass',num_classes=num_classes,average='macro')
        self.precision_metric = Precision(task='multiclass',num_classes=num_classes,average='macro')
        #self.stat_scores_metric = StatScores(task='multiclass',num_classes=num_classes,average='macro')
        self.confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=num_classes)
        self.AUROC = AUROC(task='multiclass',num_classes=self.num_classes,average='macro')
        self.AP = AveragePrecision(task='multiclass',num_classes=self.num_classes,average='macro')        

        #self.acc_metric = F1Score(task='binary',num_classes=num_classes,average='macro')
        
        #self.warm_up_iter = warm_up
        self.lr = lr
        self.batch_save_every = batch_save_every
        self.save_path = save_path
        self.min_loss = 1.0
        self.metadata = metadata
        self.debug = debug

    def forward(self,x):
        pred = self.model(x)
        return pred
    
    def training_step(self, batch, batch_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch
        #img_in = bw_1_to_3ch(img)
        pred = self.model(img) #.squeeze()

        #print(f"\nimg type/shape: {type(img)}/{img.shape}\n")
        #print(f"label type/val: {type(lbl)}/{lbl}\n")
        
        if isinstance(lbl,torch.Tensor):
            lbl_t = lbl.to(torch.long)
        else:
            lbl_t = torch.tensor(lbl,dtype=torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        #print("\n\npred vs lbl:",pred.shape,lbl_hot_t.shape,"\n\n")

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)
        #lr = self.lr_schedulers().get_last_lr()[0]
        #log_vals = {"Train/loss": loss} #, "Train/lr":lr}
        #self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        self.log("Train/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #self.log("Train/acc",acc,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Train/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #print(f"\nBatch: {batch_idx}, loss: {loss}\n")
        if  batch_idx%self.batch_save_every == 0:
            if loss < self.min_loss:
                #self.trainer.save_checkpoint(f"{self.save_path}/{trainer.logger.name}_min_loss-epoch={self.current_epoch}-step-{self.global_step}",weights_only=True)
                self.trainer.save_checkpoint(f"{self.save_path}/{self.trainer.logger.name}_min_loss.ckpt",weights_only=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch
        #img_in = bw_1_to_3ch(img)
        pred = self.model(img) #.squeeze()

        #print(f"\nimg type/shape: {type(img)}/{img.shape}\n")
        #print(f"label type/val: {type(lbl)}/{lbl}\n")
        if isinstance(lbl,torch.Tensor):
            lbl_t = lbl.to(torch.long)
        else:
            lbl_t = torch.tensor(lbl,dtype=torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)


        # print("\n\npred vs lbl:",pred,lbl_hot_t,"\n\n")
        # print("\n\nself.loss_metric:",self.loss_metric,"\n\n")
        # print("\n\n",self,"\n\n")
        #print("\n\npred vs lbl:",pred.shape,lbl_hot_t.shape,"\n\n")

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)
        #lr = self.lr_schedulers().get_last_lr()[0]
        #log_vals = {"Train/loss": loss} #, "Train/lr":lr}
        #self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        self.log("Val/loss",loss,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        #self.log("Val/acc",acc,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Val/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)

        #print(f"\nBatch: {batch_idx}, loss: {loss}\n")
        if  batch_idx%self.batch_save_every == 0:
            if loss < self.min_loss:
                #self.trainer.save_checkpoint(f"{self.save_path}/{trainer.logger.name}_min_loss-epoch={self.current_epoch}-step-{self.global_step}",weights_only=True)
                self.trainer.save_checkpoint(f"{self.save_path}/{self.trainer.logger.name}_val_min_loss.ckpt",weights_only=False)

        return
    
    def test_step(self, batch, bach_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch
        
        #img,lbl = batch
        #img_in = bw_1_to_3ch(img)
        #pred = self.model(img_in)
        pred = self.model(img)

        #print(pred.shape)

        #lbl_t = torch.tensor(lbl,dtype=torch.long) # recieved warning for this
        lbl_t = lbl.clone().detach().to(torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
        #acc = self.acc_metric(pred,lbl_t)
        f1_score = self.f1_metric(pred,lbl_t)
        recall = self.recall_metric(pred,lbl_t)
        precision = self.precision_metric(pred,lbl_t)
        #stat_scores = self.stat_scores_metric(pred,lbl_t)
        confusion_matrix = self.confusion_matrix(pred,lbl_t)
        auroc = self.AUROC(pred,lbl_t)
        ap = self.AP(pred,lbl_t)

        print(f"f1-per-class: {f1_score}")
        #lr = self.lr_schedulers().get_last_lr()[0]
        #log_vals = {"Train/loss": loss} #, "Train/lr":lr}
        #self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        #self.log("Test/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Test/f1-score",f1_score.mean(),prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        #self.log("Test/acc",acc,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Test/recall",recall,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        self.log("Test/precision",precision,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        self.log("Test/AUROC",auroc,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        self.log("Test/AP",ap,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)

        #pred_out = pred.argmax(dim=1)

        return

    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        pred = self.model(batch)
        pred_out = torch.sigmoid(pred).argmax(dim=1)
        return pred_out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    


class LitDINOv2_Ulcer_Type(L.LightningModule):
    def __init__(self,dino_model,args,loss_metric,num_classes,save_path,lr=1e-6,batch_save_every:int=500,metadata:bool=False,debug:bool=False):
        super().__init__()
        #self.save_hyperparameters(ignore="dino_model")

        self.model = LoraDinoVisionTransformerClassifier(dino_model=dino_model,num_classes=num_classes,**args)
        #self.model = DinoVisionTransformerClassifier(dino_model)
        self.loss_metric = loss_metric
        #self.acc_metric = acc_metric
        self.f1_metric = F1Score(task='multiclass',num_classes=num_classes,average=None)#,ignore_index=0)
        self.acc_metric = Accuracy(task='multiclass',num_classes=num_classes,average='macro')
        self.recall_metric = Recall(task='multiclass',num_classes=num_classes,average='macro')
        self.precision_metric = Precision(task='multiclass',num_classes=num_classes,average='macro')
        #self.stat_scores_metric = StatScores(task='multiclass',num_classes=num_classes,average='macro')
        self.confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=num_classes)

        #self.acc_metric = F1Score(task='binary',num_classes=num_classes,average='macro')
        self.num_classes = num_classes
        #self.warm_up_iter = warm_up
        self.lr = lr
        self.batch_save_every = batch_save_every
        self.save_path = save_path
        self.min_loss = 1.0
        self.metadata = metadata
        self.debug = debug

    def forward(self,x):
        pred = self.model(x)
        return pred
    
    def training_step(self, batch, batch_idx):
        if self.metadata:
            #img,lbl,meta = batch
            img,lbl,meta = batch['image'],batch['fungal_LCA_bin'],batch['image_name']
        else:
            img,lbl = batch
        #img_in = bw_1_to_3ch(img)
        pred = self.model(img) #.squeeze()

        #print(f"\nimg type/shape: {type(img)}/{img.shape}\n")
        #print(f"label type/val: {type(lbl)}/{lbl}\n")
        
        lbl_t = torch.tensor(lbl,dtype=torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        #print("\n\npred vs lbl:",pred.shape,lbl_hot_t.shape,"\n\n")

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)
        #lr = self.lr_schedulers().get_last_lr()[0]
        #log_vals = {"Train/loss": loss} #, "Train/lr":lr}
        #self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        self.log("Train/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #self.log("Train/acc",acc,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Train/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #print(f"\nBatch: {batch_idx}, loss: {loss}\n")
        if  batch_idx%self.batch_save_every == 0:
            if loss < self.min_loss:
                #self.trainer.save_checkpoint(f"{self.save_path}/{trainer.logger.name}_min_loss-epoch={self.current_epoch}-step-{self.global_step}",weights_only=True)
                self.trainer.save_checkpoint(f"{self.save_path}/{self.trainer.logger.name}_min_loss.ckpt",weights_only=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.metadata:
            #img,lbl,meta = batch
            img,lbl,meta = batch['image'],batch['bac_LCA_bin'],batch['image_name']
        else:
            img,lbl = batch
        #img_in = bw_1_to_3ch(img)
        pred = self.model(img) #.squeeze()

        #print(f"\nimg type/shape: {type(img)}/{img.shape}\n")
        #print(f"label type/val: {type(lbl)}/{lbl}\n")
        
        lbl_t = torch.tensor(lbl,dtype=torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        #print("\n\npred vs lbl:",pred.shape,lbl_hot_t.shape,"\n\n")

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)
        #lr = self.lr_schedulers().get_last_lr()[0]
        #log_vals = {"Train/loss": loss} #, "Train/lr":lr}
        #self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        self.log("Val/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #self.log("Val/acc",acc,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Val/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)

        #print(f"\nBatch: {batch_idx}, loss: {loss}\n")
        if  batch_idx%self.batch_save_every == 0:
            if loss < self.min_loss:
                #self.trainer.save_checkpoint(f"{self.save_path}/{trainer.logger.name}_min_loss-epoch={self.current_epoch}-step-{self.global_step}",weights_only=True)
                self.trainer.save_checkpoint(f"{self.save_path}/{self.trainer.logger.name}_val_min_loss.ckpt",weights_only=False)

        return
    
    def test_step(self, batch, bach_idx):
        if self.metadata:
            #img,lbl,meta = batch
            img,lbl,meta = batch['image'],batch['bac_LCA_bin'],batch['image_name']
        else:
            img,lbl = batch
        
        #img,lbl = batch
        #img_in = bw_1_to_3ch(img)
        #pred = self.model(img_in)
        pred = self.model(img)

        #print(pred.shape)

        #lbl_t = torch.tensor(lbl,dtype=torch.long) # recieved warning for this
        lbl_t = lbl.clone().detach().to(torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
        #acc = self.acc_metric(pred,lbl_t)
        f1_score = self.f1_metric(pred,lbl_t)
        recall = self.recall_metric(pred,lbl_t)
        precision = self.precision_metric(pred,lbl_t)
        #stat_scores = self.stat_scores_metric(pred,lbl_t)
        confusion_matrix = self.confusion_matrix(pred,lbl_t)

        print(f"f1-per-class: {f1_score}")
        #lr = self.lr_schedulers().get_last_lr()[0]
        #log_vals = {"Train/loss": loss} #, "Train/lr":lr}
        #self.log_dict(log_vals,prog_bar=True,on_epoch=True,on_step=False)
        #self.log("Test/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Test/f1-score",f1_score.mean(),prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #self.log("Test/acc",acc,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Test/recall",recall,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Test/precision",precision,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #self.log("Test/confusion",confusion_matrix,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        #self.log("Test/stat_scores",stat_scores,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)

        #pred_out = pred.argmax(dim=1)

        return

    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        pred = self.model(batch)
        pred_out = torch.sigmoid(pred).argmax(dim=1)
        return pred_out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer