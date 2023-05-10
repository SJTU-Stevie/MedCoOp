import pdb
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Medclip.Medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from Medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from Medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from Medclip.losses import ImageTextContrastiveLoss
from Medclip.trainer import Trainer
from Medclip.evaluator import Evaluator
from Medclip import constants
from Medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_config = {
    'batch_size': 100,
    'num_epochs': 10,
    'warmup': 0.1,  # the first 10% of training steps are used for warm-up
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'eval_batch_size': 256,
    'eval_steps': 1000,
    'save_steps': 1000,
}

# build medclip model
model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()
model.cuda()

# build evaluator
cls_prompts = generate_covid_class_prompts(n=12)
val_data = ZeroShotImageDataset(['COVID','Normal'],
                                class_names=constants.COVID_TASKS)
val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts,
                                       mode='binary')
eval_dataloader = DataLoader(val_data,
                             batch_size=train_config['eval_batch_size'],
                             collate_fn=val_collate_fn,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=4,
                             )


medclip_clf = PromptClassifier(model)
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
    mode='binary',
)

output=evaluator.evaluate()
print(output)