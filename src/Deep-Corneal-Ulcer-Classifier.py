""""""

from pathlib import Path
from functools import reduce
from typing import Callable
import random
import time
import datetime

import deeplake
import polars as pl
import numpy as np
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.transforms import v2
from torchinfo import summary
from omegaconf import OmegaConf
from ruamel.yaml import YAML
import timm
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import BaseFinetuning, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import F1Score,Accuracy,Recall,Precision,ConfusionMatrix,StatScores,AUROC,AveragePrecision

from jj_nn_framework.nn_transforms_v2 import v2_NanControl
from load_my_trained_vit_model import get_pretrained_ssl_model
from my_dino.models import LitDINOv2

# Settings to put into configuration file


def my_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = (end - start)
        hours = elapsed//3600
        minutes = (elapsed - hours*3600) // 60
        seconds = elapsed - (hours*3600 + minutes*60)
        print(f'Time taken: {hours}h{minutes}m{seconds}s')
        return {"hour":hours,"min":minutes,"sec":seconds}
    return wrapper


def remove_organizational_headings(config):
    """ """
    config_dict = OmegaConf.to_container(config)
    config_wo_headings = reduce(
        lambda d1, d2: d1 | d2, config_dict.values()
    )  # strip yaml section headings
    return OmegaConf.create(config_wo_headings)


def load_yaml_settings(config_path):
    with open(config_path) as f:
        yaml = YAML(typ="safe")
        config_yaml = yaml.load(f)
    return config_yaml


def get_config():
    umbrella = Path(__file__).parents[1]
    config_path = umbrella / f"config/KDB_dinoV2_refine_config.yaml"

    current_path = Path(__file__)
    config_path = current_path.parent.parent / f"config/current"
    config_file_path = list(config_path.glob("*_config.yaml"))[0]

    config_dict = load_yaml_settings(config_path=config_file_path)
    full_config = OmegaConf.create(config_dict)
    config = remove_organizational_headings(full_config)
    return config,config_dict


initial_config,_ = get_config()


def set_deterministic(config):
    """ """
    # seed = 0
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    return


def get_deeplake_data_splits(config, debug: bool = False):
    # load metadata
    metadata_path = Path(
        "/home/threadripper/Research/Redd_Lab/Data/DeepLake/Polars_Metadata/corneal_ulcer_study_data.csv"
    )
    corneal_ulcer_meta = pl.read_csv(metadata_path)
    unique_cu_meta = corneal_ulcer_meta.unique(subset=["pID"], maintain_order=True)

    new_madurai_train_meta = unique_cu_meta.filter(
        pl.col("study").str.contains("MADURAI"),
        pl.col("study_site") == "Madurai",
    )

    new_mad_bac = new_madurai_train_meta.filter(
        pl.col("bac_LCA_bin") == 1,
    )
    new_mad_fung = new_madurai_train_meta.filter(
        pl.col("fung_LCA_bin") == 1,
    )

    # there are fewer bacterial samples than fungal samples
    bac_samples = len(new_mad_bac)
    # get random sample of bac_samples size from fungal samples
    mad_train_bac = new_mad_bac
    mad_train_fung = new_mad_fung.sample(bac_samples)
    combined_train_meta = pl.concat([mad_train_bac, mad_train_fung])

    # get study site data for each site not in MADURAI study
    not_mad_study = unique_cu_meta.filter(~pl.col("study").str.contains("MADURAI"))
    # study site counts
    site_counts = not_mad_study.select(
        pl.col("study_site").value_counts(sort=True)
    ).unnest("study_site")
    site_list = (
        not_mad_study.select(pl.col("study_site").value_counts(sort=True))
        .unnest("study_site")
        .select(pl.col("study_site"))
        .to_series()
        .to_list()
    )
    # metadata for all hold out data not used for training
    hold_out_meta = {}
    for site in site_list:
        hold_out_meta[site] = not_mad_study.filter(pl.col("study_site") == site)
    # focus on 4 main sites as with original papers Lubini (42), Dartmouth (8), Proctor (7), Bharaptur (3) are ommitted
    madurai_meta = hold_out_meta[
        "Madurai"
    ]  # 50% split stratified random sample 50 each to match Tirunelveli
    madurai_bac_meta = madurai_meta.filter(pl.col("bac_LCA_bin") == 1)
    madurai_fung_meta = madurai_meta.filter(pl.col("fung_LCA_bin") == 1)
    coimbatore_meta = hold_out_meta[
        "Coimbatore"
    ]  # 50% split stratified random sample 50 each to match Tirunelveli
    coimbatore_bac_meta = coimbatore_meta.filter(pl.col("bac_LCA_bin") == 1)
    coimbatore_fung_meta = coimbatore_meta.filter(pl.col("fung_LCA_bin") == 1)
    pondicherry_meta = hold_out_meta[
        "Pondicherry"
    ]  # all fungal so only requires a random sample get 100 to match Tirunelveli
    tirunelveli_meta = hold_out_meta[
        "Tirunelveli"
    ]  # all bacterial so only requires a random sample 118 total smallest amount so 100 will be used
    # get test data
    pondicherry_test_meta = pondicherry_meta.sample(100)
    tirunelveli_test_meta = tirunelveli_meta.sample(100)
    madurai_test_bac_meta = madurai_bac_meta.sample(50)
    madurai_test_fung_meta = madurai_fung_meta.sample(50)
    coimbatore_test_bac_meta = coimbatore_bac_meta.sample(50)
    coimbatore_test_fung_meta = coimbatore_fung_meta.sample(50)
    combined_test_meta = pl.concat(
        [
            pondicherry_test_meta,
            tirunelveli_test_meta,
            madurai_test_bac_meta,
            madurai_test_fung_meta,
            coimbatore_test_bac_meta,
            coimbatore_test_fung_meta,
        ]
    )
    combined_test_counts = combined_test_meta.select(
        pl.col("study_site").value_counts(sort=True)
    ).unnest("study_site")
    # get validation data
    coimbatore_val_meta = hold_out_meta["Coimbatore"].join(
        combined_test_meta, on="file_path", how="anti"
    )
    coimbatore_val_counts = coimbatore_val_meta.select(
        pl.col("bac_LCA_bin").value_counts(sort=True)
    ).unnest("bac_LCA_bin")
    # 43 bacterial vs 90 fungal so can do 43+43 = 86 for validation
    coimbatore_val_bac_meta = coimbatore_val_meta.filter(pl.col("bac_LCA_bin") == 1)
    coimbatore_val_fung_meta = coimbatore_val_meta.filter(
        pl.col("fung_LCA_bin") == 1
    ).sample(len(coimbatore_val_bac_meta))
    combined_val_meta = pl.concat([coimbatore_val_bac_meta, coimbatore_val_fung_meta])
    # Generate text lists of file paths to index data from DeepLake dataset
    train_list = combined_train_meta.select(pl.col("file_path")).to_series().to_list()
    train_str = f"{str(train_list).replace('[', '(').replace(']', ')')}"
    val_list = combined_val_meta.select(pl.col("file_path")).to_series().to_list()
    val_str = f"{str(val_list).replace('[', '(').replace(']', ')')}"
    test_list = combined_test_meta.select(pl.col("file_path")).to_series().to_list()
    test_str = f"{str(test_list).replace('[', '(').replace(']', ')')}"

    # load DeepLake data
    corneal_ulcer_ds_path = str(
        Path(
            "/home/threadripper/Research/Redd_Lab/Data/DeepLake/corneal_dataset_uniform_size_v2"
        )
    )
    dl_corneal_ulcer_ds = deeplake.open_read_only(corneal_ulcer_ds_path)

    if debug:
        print(f"Dataset @ {corneal_ulcer_ds_path}\nsuccesfully loaded.\nSummary:")
        dl_corneal_ulcer_ds.summary()

    # filter training images
    dl_train_images = dl_corneal_ulcer_ds.query(
        f"SELECT * WHERE file_path IN {train_str}"
    )
    if debug:
        print("Train data summmary:\n")
        dl_train_images.summary()
    # filter validation images
    dl_val_images = dl_corneal_ulcer_ds.query(f"SELECT * WHERE file_path IN {val_str}")
    if debug:
        print("Validation data summmary:\n")
        dl_val_images.summary()
    # filter test images
    dl_test_images = dl_corneal_ulcer_ds.query(
        f"SELECT * WHERE file_path IN {test_str}"
    )
    if debug:
        print("Test data summmary:\n")
        dl_test_images.summary()

    return dl_train_images, dl_val_images, dl_test_images

def get_qmnist_data_splits(transform=None,debug:bool=False):
    qmist_train_ds = torchvision.datasets.QMNIST(
        root=Path(r"/home/threadripper/Research/Redd_Lab/Data/QMIST"),
        what="train",
        compat=True,
        download=True,
        transform=transform
    )
    qmist_val_ds = torchvision.datasets.QMNIST(
        root=Path(r"/home/threadripper/Research/Redd_Lab/Data/QMIST"),
        what="test10k",
        compat=True,
        download=True,
        transform=transform
    )
    qmist_test_ds = torchvision.datasets.QMNIST(
        root=Path(r"/home/threadripper/Research/Redd_Lab/Data/QMIST"),
        what="test50k",
        compat=True,
        download=True,
        transform=transform
    )

    return qmist_train_ds,qmist_val_ds,qmist_test_ds
                  
class DeeplakeCornealUlcerImageDataset(torch.utils.data.Dataset):
    def __init__(self, deeplake_ds: deeplake.Dataset, transform: Callable = None):
        self.ds = deeplake_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        image = self.ds[item].to_dict()["image"]
        # print(image.shape)
        # label = self.ds[item].to_dict()["bac_LCA_bin"]
        label = self.ds[item].to_dict()["fung_LCA_bin"]
        meta = {
            key: val for key, val in self.ds[item].to_dict().items() if key != "image"
        }
        # print(label,meta)

        if self.transform is not None:
            # print(image.shape)
            image = self.transform(image)
            # image, *_ = self.transform((image, label, meta))

        return image, label, meta

dataset_options = {
    "DeepLakeCUID": DeeplakeCornealUlcerImageDataset,
}

def get_dataset_model(name: str):
    if name in dataset_options.keys():
        return DeeplakeCornealUlcerImageDataset
    else:
        print(f"The requested dataset {name} is not supported")

to_tensor = torch.nn.Sequential(v2.ToImage(), v2.ToDtype(torch.float32, scale=True))

Deeplake_preproc6 = torch.nn.Sequential(
    v2.ToPILImage(),
    v2.Resize(
        size=(initial_config.img_training_size, initial_config.img_training_size)
    ),
    to_tensor,
    v2_NanControl(8),
)

Deeplake_Trivial_Preproc = torch.nn.Sequential(
    v2.ToPILImage(),
    v2.Resize(
        size=(initial_config.img_training_size, initial_config.img_training_size)
    ),
)

crop_size = initial_config.crop_size
crop_scale = initial_config.crop_scale
hfp = initial_config.hfp
vfp = initial_config.vfp
rot_range = initial_config.rot_range
brightness_range = initial_config.brightness_range
contrast_range = initial_config.contrast_range
hue_range = initial_config.hue_range
random_grayscale_prob = initial_config.random_grayscale_prob

transform_options = {
    "resize_normalize" : Deeplake_preproc6,
    "travis_resize_normalize": torch.nn.Sequential(
        v2.ToPILImage(),
        v2.Resize(
            size=(initial_config.img_training_size, initial_config.img_training_size)
        ),
        to_tensor,
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ),
    "travis_transform" : torch.nn.Sequential(
        v2.ToPILImage(),
        v2.Resize(
            size=(initial_config.img_training_size, initial_config.img_training_size)
        ),
        v2.RandomHorizontalFlip(hfp),
        v2.RandomVerticalFlip(vfp),
        to_tensor,
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        v2.RandomRotation(rot_range),
    ),
    "transform_0": torch.nn.Sequential(
        # v2.RandomResizedCrop((192,192),(0.9,1.1)),
        # v2.RandomCrop((192,192)),
        Deeplake_preproc6,
        v2.RandomResizedCrop(crop_size, crop_scale),
        v2.RandomHorizontalFlip(hfp),
        v2.RandomVerticalFlip(vfp),
        v2.RandomRotation(rot_range),
        v2.ColorJitter(
            brightness=brightness_range, contrast=contrast_range, hue=hue_range
        ),
        v2.RandomGrayscale(p=random_grayscale_prob),
    ),
    "transform_trivial": torch.nn.Sequential(
        Deeplake_Trivial_Preproc, v2.TrivialAugmentWide()
    ),
    "trivial_like": torch.nn.Sequential(
        Deeplake_preproc6,
        v2.RandomHorizontalFlip(hfp),
        v2.RandomVerticalFlip(vfp),
        v2.RandomResizedCrop(crop_size, crop_scale),
        v2.RandomChoice(
            [
                # v2.RandomCrop(192),
                v2.RandomRotation(rot_range),
                # v2.RandomResizedCrop(196, (0.8,1.2)),
                v2.Grayscale(num_output_channels=3),
                v2.ColorJitter(
                    brightness=brightness_range, contrast=(1.0, 1.0), hue=(0.0, 0.0)
                ),
                v2.ColorJitter(
                    brightness=(1.0, 1.0), contrast=contrast_range, hue=(0.0, 0.0)
                ),
                v2.ColorJitter(
                    brightness=(1.0, 1.0), contrast=(0.0, 1.0), hue=hue_range
                ),
            ]
        ),
    ),
}

def get_train_dataset_transform(name: str):
    if name in transform_options.keys():
        return transform_options[name]
    else:
        print(f"The requested transform {name} is not supported")

def get_val_dataset_transform(name: str):
    if name in transform_options.keys():
        return transform_options[name]
    else:
        print(f"The requested transform {name} is not supported")

def get_test_dataset_transform(name: str):
    if name in transform_options.keys():
        return transform_options[name]
    else:
        print(f"The requested transform {name} is not supported")

loss_options = {
    #"crossentropy" : torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5,0.5]))
    "crossentropy" : torch.nn.CrossEntropyLoss()
}

def get_loss(name:str):
    if name in loss_options.keys():
        return loss_options[name]
    else:
        print(f"The requested transform {name} is not supported")

class FreezeUnfreezeCallback(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        #self.unfrozen = False
        self.modules_to_freeze = []

    def freeze_before_training(self, pl_module):

        names_to_freeze = []
        module_to_not_freeze = None
        for name, module in pl_module.model.named_modules():
            #print(f"module name: {name}")
            if name != "classifier":
                names_to_freeze.append(name)
                self.modules_to_freeze.append(module)
            else:
                #print(f"{name} is not included in the list")
                module_to_not_freeze = module

        #print(f"modules to freeze:\n{names_to_freeze}")

        self.freeze(self.modules_to_freeze, train_bn=False) # Train batch norms or no?

        # not sure why I need this this module is excluded from the freeze operation or I thought it should be.
        for param in module_to_not_freeze.parameters():
            param.requires_grad = True

        print("Backbone frozen, final layer unfrozen")
        config = pl_module.config
        summary(pl_module,(config.batch_size,config.input_channels,config.img_training_size,config.img_training_size))

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch: #and not self.unfrozen:
            #for name, param in pl_module.named_parameters():
            #    param.requires_grad = True
            #self.unfrozen = True
            self.unfreeze_and_add_param_group(
                modules=self.modules_to_freeze,
                optimizer=optimizer,
                train_bn=True
            )
            print("Backbone unfrozen")
            config = pl_module.config
            summary(pl_module,(config.batch_size,config.input_channels,config.img_training_size,config.img_training_size))

def generate_callbacks(
        config,
        trial_name,
        #top_k:int = 5,
        #n_epoch:int = 10,
        #n_steps:int = 500,
        #save_path = "../outpl"
):
    top_k = config.top_k
    n_epoch = config.n_epoch
    n_steps = config.n_steps
    save_path = Path(__file__).parents[1] / f"{config.save_chkpt_path}/{trial_name}"
    
    # saves top-K checkpoints based on "val_loss" metric
    tl_checkpoint_callback = ModelCheckpoint(
        save_top_k=top_k,
        monitor="Train/loss",
        mode="min",
        dirpath=save_path,
        filename=f"{trial_name}_top_loss_{top_k}"+"-train_loss={Val/loss:1.4f}-epoch={epoch:04d}-step={step}", #filename=f"{trial_name}"+"-{epoch:03d}-{Val\loss:.3f}",
        save_weights_only=True, #False,
    )
    ta_checkpoint_callback = ModelCheckpoint(
        save_top_k=top_k,
        monitor="Train/acc",
        mode="min",
        dirpath=save_path,
        filename=f"{trial_name}_top_{top_k}_acc"+"-train_acc={Val/acc:1.4f}-epoch={epoch:04d}-step={step}", #filename=f"{trial_name}"+"-{epoch:03d}-{Val\loss:.3f}",
        auto_insert_metric_name=False,
        save_weights_only=True, #False,
    )

    # saves top-K checkpoints based on "val_acc" metric
    va_checkpoint_callback = ModelCheckpoint(
        save_top_k=top_k,
        monitor="Val/acc",
        mode="max",
        dirpath=save_path,
        filename=f"{trial_name}_max_{top_k}_acc"+"-val_acc={Val/acc:1.4f}-epoch={epoch:04d}-step={step}", #filename=f"{trial_name}"+"-{epoch:03d}-{Val\loss:.3f}",
        auto_insert_metric_name=False,
        save_weights_only=True, #False,
    )

    # saves top-K checkpoints based on "val_loss" metric
    vl_checkpoint_callback = ModelCheckpoint(
        save_top_k=top_k,
        monitor="Val/loss",
        mode="min",
        dirpath=save_path,
        filename=f"{trial_name}_min_{top_k}_loss"+"-val_loss={Val/loss:1.4f}-epoch={epoch:04d}-step={step}", #filename=f"{trial_name}"+"-{epoch:03d}-{Val\loss:.3f}",
        auto_insert_metric_name=False,
        save_weights_only=True, #False,
    )
    # saves last checkpoint
    l_checkpoint_callback = ModelCheckpoint(
        #save_last=True,
        save_top_k=1,
        dirpath=save_path,
        filename=f"{trial_name}_latest"+"-val_acc={Val/acc:1.4f}-val_loss={Val/loss:1.4f}-train_acc={Val/acc:1.4f}-train_loss={Val/loss:1.4f}-{epoch:04d}-{step}", #filename=f"{trial_name}"+"-{epoch:03d}-{Val\loss:.3f}",
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    freeze_unfreeze_callback = FreezeUnfreezeCallback(unfreeze_at_epoch=config.unfreeze_epoch)

    #callbacks = [freeze_unfreeze_callback,l_checkpoint_callback,va_checkpoint_callback,vl_checkpoint_callback]
    callbacks = [l_checkpoint_callback,va_checkpoint_callback,vl_checkpoint_callback]
    return callbacks

class MobilenetV4_Redd_DCUC(L.LightningModule):
    def __init__(self, model,loss_metric,config): #lr,metadata:bool=False):
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes
        self.loss_metric = loss_metric
        self.drop_rate = config.dropout_rate
        #self.lr = lr
        #self.metadata = metadata
        self.config = config
        self.lr = config.lr
        self.metadata = config.has_metadata

        self.f1_metric = F1Score(task='multiclass',num_classes=self.num_classes,average=None)
        self.acc_metric = Accuracy(task='multiclass',num_classes=self.num_classes,average='weighted')
        self.recall_metric = Recall(task='multiclass',num_classes=self.num_classes,average='weighted')
        self.precision_metric = Precision(task='multiclass',num_classes=self.num_classes,average='weighted')
        self.confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=self.num_classes)
        self.AUROC = AUROC(task='multiclass',num_classes=self.num_classes,average='weighted')
        self.AP = AveragePrecision(task='multiclass',num_classes=self.num_classes,average='weighted')

    def forward(self, input):     
        return self.model(input)
    
    def training_step(self, batch, batch_idx):

        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch

        pred = self.model(img)

        if isinstance(lbl,torch.Tensor):
            lbl_t = lbl.to(torch.long)
        else:
            lbl_t = torch.tensor(lbl,dtype=torch.long)

        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)

        self.log("Train/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Train/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch

        pred = self.model(img)

        if isinstance(lbl,torch.Tensor):
            lbl_t = lbl.to(torch.long)
        else:
            lbl_t = torch.tensor(lbl,dtype=torch.long)

        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)

        self.log("Val/loss",loss,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        self.log("Val/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)

        return
    
    def test_step(self, batch, batch_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch
        
        pred = self.model(img)
        lbl_t = lbl.clone().detach().to(torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self.model(batch)
        k = self.num_classes
        print(f"pred shape: {pred.shape}\n")
        topk_probabilities, topk_classes = torch.topk(pred.softmax(dim=1),k=k)
        topk_probabilities = topk_probabilities * 100
        pred_out = pred.softmax(dim=1).argmax(dim=1)

        return pred_out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), lr=self.lr)
        return optimizer
    
class MobilenetV2_Redd_DCUC(L.LightningModule):
    def __init__(self, model,loss_metric,config): #lr,metadata:bool=False):
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes
        self.loss_metric = loss_metric
        self.drop_rate = config.dropout_rate
        #self.lr = lr
        #self.metadata = metadata
        self.config = config
        self.lr = config.lr
        self.metadata = config.has_metadata

        self.f1_metric = F1Score(task='multiclass',num_classes=self.num_classes,average=None)
        self.acc_metric = Accuracy(task='multiclass',num_classes=self.num_classes,average='macro')
        self.recall_metric = Recall(task='multiclass',num_classes=self.num_classes,average='macro')
        self.precision_metric = Precision(task='multiclass',num_classes=self.num_classes,average='macro')
        self.confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=self.num_classes)
        self.AUROC = AUROC(task='multiclass',num_classes=self.num_classes)
        self.AP = AveragePrecision(task='multiclass',num_classes=self.num_classes)

    def forward(self, input):     
        return self.model(input)
    
    def training_step(self, batch, batch_idx):

        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch

        pred = self.model(img)

        if isinstance(lbl,torch.Tensor):
            lbl_t = lbl.to(torch.long)
        else:
            lbl_t = torch.tensor(lbl,dtype=torch.long)

        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)

        self.log("Train/loss",loss,prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)
        self.log("Train/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=True,sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch

        pred = self.model(img)

        if isinstance(lbl,torch.Tensor):
            lbl_t = lbl.to(torch.long)
        else:
            lbl_t = torch.tensor(lbl,dtype=torch.long)

        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
        acc = self.f1_metric(pred,lbl_t)

        self.log("Val/loss",loss,prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)
        self.log("Val/acc",acc.mean(),prog_bar=True,on_epoch=True,on_step=False,sync_dist=True)

        return
    
    def test_step(self, batch, batch_idx):
        if self.metadata:
            img,lbl,meta = batch
        else:
            img,lbl = batch
        
        pred = self.model(img)
        lbl_t = lbl.clone().detach().to(torch.long)
        lbl_hot_t = torch.nn.functional.one_hot(lbl_t,num_classes=self.num_classes)

        loss = self.loss_metric(pred, lbl_hot_t.float())
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self.model(batch)
        k = self.num_classes
        print(f"pred shape: {pred.shape}\n")
        topk_probabilities, topk_classes = torch.topk(pred.softmax(dim=1),k=k)
        topk_probabilities = topk_probabilities * 100
        pred_out = pred.softmax(dim=1).argmax(dim=1)

        return pred_out
    
    def configure_optimizers(self):
        # Define the RMSprop optimizer with a learning rate, alpha, and L2 regularization (weight_decay)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, weight_decay=0.3)
        return optimizer
    
architecture_options = {
    "MobileNetV2": MobilenetV2_Redd_DCUC,
    "MobileNetV4": MobilenetV4_Redd_DCUC,
}

def select_model(name:str):
    if name in architecture_options.keys():
        return architecture_options[name]
    else:
        print(f"The requested transform {name} is not supported")

#def generate_Redd_model(config,loss_metric,metadata):
def generate_Redd_model(config,loss_metric,config_yaml=None,save_chkpt_path=None):
    #print(f"mobilenets:\n{timm.list_models('*mobilenet*', pretrained=True)}")
    #print(f"efficientets:\n{timm.list_models('*efficientnet*', pretrained=True)}")

    if config.model_arch == "MobileNetV2":
        model = timm.create_model(
            "mobilenetv2_100", pretrained=True, num_classes=config.num_classes, in_chans=config.input_channels #, num_classes=10
        )
        redd_model = MobilenetV2_Redd_DCUC(model=model,loss_metric=loss_metric,config=config)

    elif config.model_arch == "MobileNetV4":
        # get desired model
        model = timm.create_model(
            "mobilenetv4_hybrid_medium.e500_r224_in1k", pretrained=True, num_classes=config.num_classes, in_chans=config.input_channels #, num_classes=10
        )
        redd_model = MobilenetV4_Redd_DCUC(model=model,loss_metric=loss_metric,config=config)
    elif config.model_arch == "dinov2-jj-pre":
        model = get_pretrained_ssl_model(weights_path=r"/mnt/d/JJ/Dev2/Redd_Lab/Projects/Saved_Models/KDB_Pass_0/Full_SSL/active_weights/")

        #LoRA_args = config.lora_settings
        LoRA_args = config_yaml["model_settings"]["lora_settings"]

        redd_model = LitDINOv2(
            model,
            LoRA_args,
            loss_metric=loss_metric, #criterion,
            num_classes=config.num_classes, #NUM_CLASSES,
            lr=config.lr,#LR,
            save_path=save_chkpt_path,
            batch_save_every=config.batch_save_every, #BATCH_SAVE_EVERY,
            metadata=True,
        )
    else:
        print(f"\nThe requested architecture {config.model_arch} is not supported\n")

    # summary
    #summary(model, input_size=(config.batch_size, config.input_channels, config.img_training_size, config.img_training_size))

    
    # get model specific normalization resizing transforms
    #data_config = timm.data.resolve_data_config({}, model=model)
    #preproc_transform = timm.data.create_transform(data_config)
    #print(preproc_transform)
    return redd_model

def generate_trial_name(config):
    # meta data for trial
    now = datetime.datetime.now()
    time = f"{now:%m-%d-%Y_%Hh%Mm%Ss}"
    trial_name = f"{config.trial_prefix}_{config.model_arch}_Full_Data_{time}"
    return trial_name

@my_timer
def test_dataloaders():
    # get config
    config,_ = get_config()

    loss_metric = get_loss(config.loss_type)
    batch_accum = config.target_batch_size/config.batch_size

    # set random seed to make data deterministic
    pl.set_random_seed(config.data_split_seed)
    set_deterministic(config=config)

    # get data splits
    dl_train_data, dl_val_data, dl_test_data = get_deeplake_data_splits(config=config)
    dl_train_data.summary(), dl_val_data.summary(), dl_test_data.summary()

    # get dataset_model
    dataset_model = get_dataset_model(config.dataset_model)

    # get transform
    train_dataset_transform = get_train_dataset_transform(config.train_dataset_transform)
    val_dataset_transform = get_val_dataset_transform(config.val_dataset_transform)
    test_dataset_transform = get_test_dataset_transform(config.test_dataset_transform)

    # create dataset
    dl_train_ds = dataset_model(dl_train_data, transform=train_dataset_transform)
    dl_val_ds = dataset_model(dl_train_data, transform=val_dataset_transform)
    dl_test_ds = dataset_model(dl_test_data, transform=test_dataset_transform)

    # create dataloader
    train_dl = DataLoader(
        dl_train_ds,
        batch_size=16, # TODO change to config
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    val_dl = DataLoader(
        dl_val_ds,
        batch_size=16, # TODO change to config
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    test_dl = DataLoader(
        dl_test_ds,
        batch_size=16, # TODO change to config
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    
    # generate test batch
    train_batch = next(iter(train_dl))
    val_batch = next(iter(val_dl))
    test_batch = next(iter(test_dl))
    #print(f"batch:\n {batch}\n")

    #if config.visual_sanity_check:
    print("\n\nPerforming Visual Sanity Check\n\n")
    # view test batch
    import napari

    viewer = napari.Viewer(show=False)
    viewer.add_image(
        train_batch[0].detach().cpu().squeeze().permute(-4, -2, -1, -3).numpy(),name="image_batch",
    )
    viewer.add_image(
        val_batch[0].detach().cpu().squeeze().permute(-4, -2, -1, -3).numpy(),name="image_batch",
    )
    viewer.add_image(
        test_batch[0].detach().cpu().squeeze().permute(-4, -2, -1, -3).numpy(),name="image_batch",
    )
    viewer.show()
    napari.run()
    print("\n\nVisual Sanity Check Complete\n\n")

    

@my_timer
def test_model_generation():

    config,_ = get_config()

    torch.set_float32_matmul_precision = config.matmul_precision
    print(f"\n\ntorch matmul precision set to: {config.matmul_precision}\n\n")

    print("\n\nSetting Deterministic Flags\n\n")
    # set random seed to make data deterministic
    set_deterministic(config=config)

    print("\n\nConfiguration loaded\n\n")
    loss_metric = get_loss(config.loss_type)
    
    batch_accum = config.target_batch_size/config.batch_size

    qmnist_train_ds,qmnist_val_ds,qmnist_test_ds = get_qmnist_data_splits(transform=Deeplake_preproc6)
    print("\n\nData Splits acquired\n\n")
    
    qmnist_train_dl = DataLoader(
        qmnist_train_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )

    qmnist_val_dl = DataLoader(
        qmnist_val_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )

    qmnist_test_dl = DataLoader(
        qmnist_test_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )

    print("Dataloaders initialized")

    trial_name = generate_trial_name(config)

    callbacks = generate_callbacks(config, trial_name=trial_name)
    print("Callbacks initialized")

    log_path = Path(__file__).parents[1] / f"{config.log_path}/{trial_name}"
    logger = TensorBoardLogger(save_dir=log_path, name=trial_name)
    print("Logger initialized")

    batch = next(iter(qmnist_train_dl))

    if config.visual_sanity_check:
        print("Performing Visual Sanity Check")
        # view test batch
        import napari

        viewer = napari.Viewer(show=False)
        viewer.add_image(
            batch[0].detach().cpu().squeeze().numpy(),
            name="image_batch",
        )
        viewer.show()
        napari.run()

        print("Visual Sanity Check Complete")

    redd_model = generate_Redd_model(config,loss_metric=loss_metric) #,metadata=False)

    print(f"Model loaded:\n{summary(redd_model, input_size=(config.batch_size, 1, config.img_training_size, config.img_training_size))}")

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        logger=logger,
        callbacks=callbacks,
        precision=config.precision,
        strategy=config.strategy,
        accelerator=config.accelerator,
        devices=config.num_devices,
        num_nodes=config.num_nodes,
        accumulate_grad_batches=batch_accum,
    )

    print(f"\n\nStarting trial for {trial_name} with learning rate: {config.lr}.\n\n")
    trainer.fit(model=redd_model,train_dataloaders=qmnist_train_dl,val_dataloaders=qmnist_val_dl)
    print("Training complete")

    logs= trainer.test(dataloaders=qmnist_test_dl,ckpt_path="best")

@my_timer
def train_model():

    config,config_yaml = get_config()

    torch.set_float32_matmul_precision = config.matmul_precision
    print(f"\n\ntorch matmul precision set to: {config.matmul_precision}\n\n")

    print("\n\nSetting Deterministic Flags\n\n")
    # set random seed to make data deterministic
    set_deterministic(config=config)

    print("\n\nConfiguration loaded\n\n")
    loss_metric = get_loss(config.loss_type)
    print(f"\n\nloss metric: {loss_metric}\n\n")

    if config.target_batch_size > config.batch_size:
        batch_accum = config.target_batch_size/config.batch_size
    else:
        batch_accum = 1

    # get data splits
    dl_train_data, dl_val_data, dl_test_data = get_deeplake_data_splits(config=config)
    dl_train_data.summary(), dl_val_data.summary(), dl_test_data.summary()

    # get dataset_model
    dataset_model = get_dataset_model(config.dataset_model)
    
    # get transform
    train_dataset_transform = get_train_dataset_transform(config.train_dataset_transform)
    val_dataset_transform = get_val_dataset_transform(config.val_dataset_transform)
    test_dataset_transform = get_test_dataset_transform(config.test_dataset_transform)

    # create dataset
    dl_train_ds = dataset_model(dl_train_data, transform=train_dataset_transform)
    dl_val_ds = dataset_model(dl_train_data, transform=val_dataset_transform)
    dl_test_ds = dataset_model(dl_test_data, transform=test_dataset_transform)

    # create dataloader
    train_dl = DataLoader(
        dl_train_ds,
        batch_size=16, # TODO change to config
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    val_dl = DataLoader(
        dl_val_ds,
        batch_size=16, # TODO change to config
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    test_dl = DataLoader(
        dl_test_ds,
        batch_size=16, # TODO change to config
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )

    print("\n\nDataloaders initialized\n\n")

    trial_name = generate_trial_name(config)

    callbacks = generate_callbacks(config, trial_name=trial_name)
    print("\n\nCallbacks initialized\n\n")

    log_path = Path(__file__).parents[1] / f"{config.log_path}/{trial_name}"
    logger = TensorBoardLogger(save_dir=log_path, name=trial_name)
    print("\n\nLogger initialized\n\n")

    # save trial settings with checkpoints
    save_chkpt_path = Path(__file__).parents[1] / f"{config.save_chkpt_path}/{trial_name}"
    save_chkpt_path.mkdir(parents=False,exist_ok=True)
    save_config_path = f"{save_chkpt_path}/{trial_name}_config.yaml"
    config_df = pl.DataFrame(config_yaml)
    config_df.write_ndjson(save_config_path)

    batch = next(iter(train_dl))

    if config.visual_sanity_check:
        print("\n\nPerforming Visual Sanity Check\n\n")
        # view test batch
        import napari

        viewer = napari.Viewer(show=False)
        viewer.add_image(
            batch[0].detach().cpu().squeeze().numpy(),
            name="image_batch",
        )
        viewer.show()
        napari.run()

        print("\n\nVisual Sanity Check Complete\n\n")

    redd_model = generate_Redd_model(config,loss_metric=loss_metric,config_yaml=config_yaml,save_chkpt_path=save_chkpt_path) #,metadata=config.has_metadata)

    print(f"\n\nModel loaded:\n{summary(redd_model, input_size=(config.batch_size, config.input_channels, config.img_training_size, config.img_training_size))}\n\n")

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        logger=logger,
        callbacks=callbacks,
        precision=config.precision,
        strategy=config.strategy,
        accelerator=config.accelerator,
        devices=config.num_devices,
        num_nodes=config.num_nodes,
        accumulate_grad_batches=batch_accum,
    )

    #tuner =Tuner(trainer)
#
    #lr_finder = tuner.lr_find(redd_model)
#
    ## Results can be found in
    #print(lr_finder.results)
#
    ## Plot with
    #fig = lr_finder.plot(suggest=True)
    #fig.show()
#
    ## Pick point based on plot, or get suggestion
    #new_lr = lr_finder.suggestion()
#
    ## update hparams of the model
    #redd_model.hparams.lr = new_lr



    #return
    #break

    print(f"\n\nStarting trial for {trial_name} with learning rate: {config.lr}.\n\n")
    #trainer.fit(model=redd_model,train_dataloaders=train_dl,val_dataloaders=val_dl)
    trainer.fit(model=redd_model,train_dataloaders=test_dl,val_dataloaders=val_dl)
    print("\n\nTraining complete\n\n")

    #logs= trainer.test(dataloaders=test_dl,ckpt_path="best")
    logs= trainer.test(dataloaders=train_dl,ckpt_path="best")

#loss_metric = torch.nn.CrossEntropyLoss()
#redd_model = generate_Redd_model(initial_config,loss_metric=loss_metric,metadata=False)

# run testd
#test_model_generation()
test_dataloaders()
#train_model()
