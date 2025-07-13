""""""
from pathlib import Path
from functools import reduce
from typing import Callable
import random

import deeplake
import polars as pl
import numpy as np
import torch
from torch.utils.data import ConcatDataset,DataLoader,random_split
from torchvision.transforms import v2
from omegaconf import OmegaConf
from ruamel.yaml import YAML
from jj_nn_framework.nn_transforms_v2 import v2_NanControl

# Settings to put into configuration file

def remove_organizational_headings(config):
    """
    """
    config_dict = OmegaConf.to_container(config)
    config_wo_headings = reduce(lambda d1,d2: d1 | d2,config_dict.values()) # strip yaml section headings
    return OmegaConf.create(config_wo_headings)

def load_yaml_settings(config_path):
    with open(config_path) as f:
        yaml = YAML(typ='safe')
        config_yaml = yaml.load(f)
    return config_yaml

def get_config():
    umbrella = Path(__file__).parents[1]
    config_path = umbrella/f"config/KDB_dinoV2_refine_config.yaml"

    current_path = Path(__file__)
    config_path = current_path.parent.parent / f"config/current"
    config_file_path = list(config_path.glob("*_config.yaml"))[0]

    config_dict = load_yaml_settings(config_path=config_file_path)
    full_config = OmegaConf.create(config_dict)
    config = remove_organizational_headings(full_config)
    return config

initial_config = get_config()

def set_deterministic(config):
    """
    """
    #seed = 0
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=True,warn_only=True)
    return

def get_deeplake_data_splits(config, debug:bool=False):

    # load metadata
    metadata_path = Path("/home/threadripper/Research/Redd_Lab/Data/DeepLake/Polars_Metadata/corneal_ulcer_study_data.csv")
    corneal_ulcer_meta = pl.read_csv(metadata_path)
    unique_cu_meta = corneal_ulcer_meta.unique(subset=["pID"],maintain_order=True)

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
    combined_train_meta = pl.concat([mad_train_bac,mad_train_fung])

    # get study site data for each site not in MADURAI study
    not_mad_study = unique_cu_meta.filter(~pl.col("study").str.contains("MADURAI"))
    # study site counts
    site_counts = not_mad_study.select(pl.col("study_site").value_counts(sort=True)).unnest("study_site")
    site_list = not_mad_study.select(pl.col("study_site").value_counts(sort=True)).unnest("study_site").select(pl.col("study_site")).to_series().to_list()
    # metadata for all hold out data not used for training
    hold_out_meta = {}
    for site in site_list:
        hold_out_meta[site] = not_mad_study.filter(
            pl.col("study_site") == site
        )
    # focus on 4 main sites as with original papers Lubini (42), Dartmouth (8), Proctor (7), Bharaptur (3) are ommitted
    madurai_meta = hold_out_meta["Madurai"] # 50% split stratified random sample 50 each to match Tirunelveli
    madurai_bac_meta = madurai_meta.filter(pl.col("bac_LCA_bin")==1)
    madurai_fung_meta = madurai_meta.filter(pl.col("fung_LCA_bin")==1)
    coimbatore_meta = hold_out_meta["Coimbatore"] # 50% split stratified random sample 50 each to match Tirunelveli
    coimbatore_bac_meta = coimbatore_meta.filter(pl.col("bac_LCA_bin")==1)
    coimbatore_fung_meta = coimbatore_meta.filter(pl.col("fung_LCA_bin")==1)
    pondicherry_meta = hold_out_meta["Pondicherry"] # all fungal so only requires a random sample get 100 to match Tirunelveli
    tirunelveli_meta = hold_out_meta["Tirunelveli"] # all bacterial so only requires a random sample 118 total smallest amount so 100 will be used
    # get test data
    pondicherry_test_meta = pondicherry_meta.sample(100)
    tirunelveli_test_meta = tirunelveli_meta.sample(100)
    madurai_test_bac_meta = madurai_bac_meta.sample(50)
    madurai_test_fung_meta = madurai_fung_meta.sample(50)
    coimbatore_test_bac_meta = coimbatore_bac_meta.sample(50)
    coimbatore_test_fung_meta = coimbatore_fung_meta.sample(50)
    combined_test_meta = pl.concat([pondicherry_test_meta,tirunelveli_test_meta,madurai_test_bac_meta,madurai_test_fung_meta,coimbatore_test_bac_meta,coimbatore_test_fung_meta])
    combined_test_counts = combined_test_meta.select(pl.col("study_site").value_counts(sort=True)).unnest("study_site")
    # get validation data
    coimbatore_val_meta = hold_out_meta["Coimbatore"].join(combined_test_meta,on="file_path",how="anti")
    coimbatore_val_counts = coimbatore_val_meta.select(pl.col("bac_LCA_bin").value_counts(sort=True)).unnest("bac_LCA_bin")
    # 43 bacterial vs 90 fungal so can do 43+43 = 86 for validation
    coimbatore_val_bac_meta = coimbatore_val_meta.filter(pl.col("bac_LCA_bin")==1)
    coimbatore_val_fung_meta = coimbatore_val_meta.filter(pl.col("fung_LCA_bin")==1).sample(len(coimbatore_val_bac_meta))
    combined_val_meta = pl.concat([coimbatore_val_bac_meta,coimbatore_val_fung_meta])
    # Generate text lists of file paths to index data from DeepLake dataset
    train_list = combined_train_meta.select(pl.col("file_path")).to_series().to_list()
    train_str = f"{str(train_list).replace("[","(").replace("]",")")}"
    val_list = combined_val_meta.select(pl.col("file_path")).to_series().to_list()
    val_str = f"{str(val_list).replace("[","(").replace("]",")")}"
    test_list = combined_test_meta.select(pl.col("file_path")).to_series().to_list()
    test_str = f"{str(test_list).replace("[","(").replace("]",")")}"

    # load DeepLake data
    corneal_ulcer_ds_path = str(Path("/home/threadripper/Research/Redd_Lab/Data/DeepLake/corneal_dataset_uniform_size_v2"))
    dl_corneal_ulcer_ds = deeplake.open_read_only(corneal_ulcer_ds_path)

    if debug:
        print(f"Dataset @ {corneal_ulcer_ds_path}\nsuccesfully loaded.\nSummary:")
        dl_corneal_ulcer_ds.summary()

    # filter training images 
    dl_train_images = dl_corneal_ulcer_ds.query(f"SELECT * WHERE file_path IN {train_str}")
    if debug:
        print("Train data summmary:\n")
        dl_train_images.summary()
    # filter validation images
    dl_val_images = dl_corneal_ulcer_ds.query(f"SELECT * WHERE file_path IN {val_str}")
    if debug:
        print("Validation data summmary:\n")
        dl_val_images.summary()
    # filter test images
    dl_test_images = dl_corneal_ulcer_ds.query(f"SELECT * WHERE file_path IN {test_str}")
    if debug:
        print("Test data summmary:\n")
        dl_test_images.summary()

    return dl_train_images,dl_val_images,dl_test_images

class DeeplakeCornealUlcerImageDataset(torch.utils.data.Dataset):
    def __init__(self, deeplake_ds: deeplake.Dataset, transform: Callable = None):
        self.ds = deeplake_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        image =  self.ds[item].to_dict()["image"]
        #print(image.shape)
        #label = self.ds[item].to_dict()["bac_LCA_bin"]
        label = self.ds[item].to_dict()["fung_LCA_bin"]
        meta = {key:val for key,val in self.ds[item].to_dict().items() if key != "image"}
        #print(label,meta)

        if self.transform is not None:
            #print(image.shape)
            image = self.transform(image)
            #image, *_ = self.transform((image, label, meta))

        return image, label, meta

dataset_options = {
    "DeepLakeCUID": DeeplakeCornealUlcerImageDataset,
}
    
def get_dataset_model(name:str):

    if name in dataset_options.keys():
        return DeeplakeCornealUlcerImageDataset
    else:
        print(f"The requested dataset {name} is not supported")

to_tensor = torch.nn.Sequential(
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True)
)

Deeplake_preproc6 = torch.nn.Sequential(
    v2.ToPILImage(),
    v2.Resize(size=(initial_config.img_training_size,initial_config.img_training_size)),
    to_tensor,
    v2_NanControl(8),
)

Deeplake_Trivial_Preproc = torch.nn.Sequential(
    v2.ToPILImage(),
    v2.Resize(size=(initial_config.img_training_size,initial_config.img_training_size)),
)

crop_size = initial_config.crop_size
crop_scale = initial_config.crop_scale
rot_range = initial_config.rot_range
brightness_range = initial_config.brightness_range
contrast_range = initial_config.contrast_range
hue_range = initial_config.hue_range
random_grayscale_prob = initial_config.random_grayscale_prob

transform_options = {
    "transform_0" : torch.nn.Sequential(
        #v2.RandomResizedCrop((192,192),(0.9,1.1)),
        #v2.RandomCrop((192,192)),
        Deeplake_preproc6,
        v2.RandomResizedCrop(crop_size,crop_scale),    
        v2.RandomHorizontalFlip(),
        #v2.RandomVerticalFlip(),
        v2.RandomRotation(rot_range),
        v2.ColorJitter(
            brightness=brightness_range,
            contrast=contrast_range,
            hue=hue_range
        ),
        v2.RandomGrayscale(p = random_grayscale_prob),
    ),

    "transform_trivial" : torch.nn.Sequential(
        Deeplake_Trivial_Preproc,
        v2.TrivialAugmentWide()
    ),

    "trivial_like" : torch.nn.Sequential(
        Deeplake_preproc6,
        v2.RandomHorizontalFlip(),
        #v2.RandomVerticalFlip(),
        v2.RandomResizedCrop(crop_size, crop_scale),
        v2.RandomChoice([
            #v2.RandomCrop(192),
            v2.RandomRotation(rot_range),
            #v2.RandomResizedCrop(196, (0.8,1.2)),
            v2.Grayscale(num_output_channels=3),
            v2.ColorJitter(
                brightness=brightness_range,
                contrast=(1.0,1.0),
                hue=(0.0,0.0)
            ),
            v2.ColorJitter(
                brightness=(1.0,1.0),
                contrast=contrast_range,
                hue=(0.0,0.0)
            ),
            v2.ColorJitter(
                brightness=(1.0,1.0),
                contrast=(0.0,1.0),
                hue=hue_range
            )
        ])
    )
}

def get_dataset_transform(name:str):

    if name in transform_options.keys():
        return transform_options[name]
    else:
        print(f"The requested transform {name} is not supported")


def test_pipeline():

    # get config
    config = get_config()


    # set random seed to make data deterministic
    pl.set_random_seed(config.data_split_seed)
    set_deterministic(config=config)

    # get data splits
    dl_train_data,dl_val_data,dl_test_data = get_deeplake_data_splits(config=config)
    dl_train_data.summary(),dl_val_data.summary(),dl_test_data.summary()
    # get dataset_model
    dataset_model = get_dataset_model(config.dataset_model)
    # get transform
    dataset_transform = get_dataset_transform(config.dataset_transform)
    # create dataset
    dl_train_ds = dataset_model(dl_train_data,transform=dataset_transform)
    # create dataloader
    ulcer_train_dl = DataLoader(dl_train_ds,batch_size=16,shuffle=config.shuffle,num_workers=config.num_workers,drop_last=config.drop_last)
    # generate test batch
    batch = next(iter(ulcer_train_dl))

    # view test batch
    import napari
    viewer = napari.Viewer(show=False)
    viewer.add_image(batch[0].detach().cpu().squeeze().permute(-4,-2,-1,-3).numpy(),name="image_batch")
    viewer.show()
    napari.run()


# run pipeline test
test_pipeline()
