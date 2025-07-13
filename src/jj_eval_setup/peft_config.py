"""
"""

from peft import LoraConfig

'''
# ExPLoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["qkv"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"],
    init_lora_weights="pissa_niter_4"
)
'''

#'''
# fine tune
lora_config = LoraConfig(
    r=512,                   #256, #4096, #512, #32,            #64, #512, #1024, #256, #64
    lora_alpha=1024,         #8192, #1024, #64,            #128, #1024, #2048, #512, #128
    target_modules=["qkv"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"],
    init_lora_weights="pissa_niter_4"
)
#'''

'''
seed = 0
last layer unfrozen
norms unfrozen

loss =  0.253251850605011
mean_loss = 0.27829108512151747
'''

'''
seed = 0
last layer unfrozen
norms unfrozen

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["qkv"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"],
    init_lora_weights="pissa_niter_4"
)

loss =  0.18679755926132202
mean_loss =  0.24392759451966087
'''

'''
seed = 0
last layer unfrozen
norms unfrozen

lora_config = LoraConfig(
    r=256,
    lora_alpha=512,
    target_modules=["qkv"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"],
    init_lora_weights="pissa_niter_4"
)

loss = 
mean_loss = 
'''
