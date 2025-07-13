""" """

from pathlib import Path
from functools import reduce

from omegaconf import OmegaConf
from torchinfo import summary
import torch
from ruamel.yaml import YAML

from jj_dinov2.models import build_model


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
    config_path = umbrella / f"config/trained_ssl_jj/KDB_dinoV2_refine_config.yaml"
    config_dict = load_yaml_settings(config_path)
    full_config = OmegaConf.create(config_dict)
    config = remove_organizational_headings(full_config)
    return config


def get_ssl_model(config):
    model_args = {
        "arch": config.student.arch,  # "vit_small",
        "patch_size": config.student.patch_size,  # 16,
        "layerscale": config.student.layerscale,  # 1.0e-05,
        "ffn_layer": config.student.ffn_layer,  # "mlp",
        "block_chunks": config.student.block_chunks,  # 0,
        "qkv_bias": config.student.qkv_bias,  # True,
        "proj_bias": config.student.proj_bias,  # True,
        "ffn_bias": config.student.ffn_bias,  # True,
        "num_register_tokens": config.student.num_register_tokens,  # 0,
        "interpolate_offset": config.student.interpolate_offset,  # 0.1,
        "interpolate_antialias": config.student.interpolate_antialias,  # False,
        "drop_path_rate": config.student.drop_path_rate,  # 0.0, #0.3,
        "drop_path_uniform": config.student.drop_path_uniform,  # False, #True
    }

    model_args = OmegaConf.create(model_args)

    # student_backbone,teacher_backbone,embed_dim = build_model(
    ssl_model, *_ = build_model(
        model_args,
        only_teacher=True,
        img_size=config.student.max_img_size,  # global_crop_size
    )
    return ssl_model


def load_pretrained_ssl_model(ssl_model, weights_path):
    # path_to_ssl_data_weights = r"/mnt/d/JJ/Dev2/Redd_Lab/Projects/Saved_Models/KDB_Pass_0/Full_SSL/active_weights/"

    # specific_ssl_ckpt = r"KDB-DinoV2_JJ-VIT14-gc2-lc8_01-31-2025_23h21m_top_acc_10-epoch=0075-step=122082-kNN_accuracy=64.77.ckpt"

    # ssl_data_ckpt_path = list(Path(path_to_ssl_data_weights).glob("*.ckpt"))[0]
    ssl_data_ckpt_path = list(Path(weights_path).glob("*.ckpt"))[0]

    ssl_data_ckpt = torch.load(ssl_data_ckpt_path)

    def get_relevant_backbone_weights(dinov2_ckpt):
        # get relevant weights
        ckpt = dinov2_ckpt
        state_dict = ckpt["state_dict"]

        model_ckpt = {
            key.split(".", 1)[1]: val
            for key, val in state_dict.items()
            if "teacher_backbone" in key
        }
        # get teacher backbone weights
        return model_ckpt

    ssl_teacher_ckpt = get_relevant_backbone_weights(ssl_data_ckpt)

    ssl_model.load_state_dict(ssl_teacher_ckpt)

    return ssl_model


def get_pretrained_ssl_model(weights_path):
    """"""
    config = get_config()
    ssl_model = get_ssl_model(config)
    pretrained_ssl_model = load_pretrained_ssl_model(
        ssl_model=ssl_model, weights_path=weights_path
    )
    return pretrained_ssl_model

# test
def test_get_pretrained_ssl_model():
    pretrained_ssl = get_pretrained_ssl_model(
        weights_path=r"/mnt/d/JJ/Dev2/Redd_Lab/Projects/Saved_Models/KDB_Pass_0/Full_SSL/active_weights/"
    )
    summary(pretrained_ssl,(1,3,224,224))

#test_get_pretrained_ssl_model()
