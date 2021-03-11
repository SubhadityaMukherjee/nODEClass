# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + run_control={"marked": false}
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import cv2
import json
from collections import Counter
import pickle
import numpy as np

from efficientnet_pytorch import EfficientNet

import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torchdiffeq import odeint

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from sklearn import metrics, model_selection,preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


import optuna
from optuna.integration import PyTorchLightningPruningCallback
os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# -

import torchsnooper as sn

# # Look at data 
# - Create a csv for easy loading

main_path = "/media/hdd/Datasets/asl/"

all_ims = glob.glob(main_path+"/*/*/*/*.jpg");all_ims[0]

len(all_ims)


def create_label(x):
    return x.split("/")[-2]


df = pd.DataFrame.from_dict({x:create_label(x) for x in all_ims} ,orient='index').reset_index()
df.columns = ["image_id","label"]

df.head()

df_b = df

df.label.unique()

subs = list('VSM');subs

df = df[df["label"].isin(subs)]

temp = preprocessing.LabelEncoder()
df['label'] = temp.fit_transform(df.label.values)

label_map=  {i: l for i, l in enumerate(temp.classes_)}

df.label.nunique()

df.label.value_counts()

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop=True)
stratify = StratifiedKFold(n_splits=5)
for i, (t_idx, v_idx) in enumerate(
        stratify.split(X=df.image_id.values, y=df.label.values)):
    df.loc[v_idx, "kfold"] = i
    df.to_csv("train_folds.csv", index=False)

pd.read_csv("train_folds.csv").head(1)


# # Architecture

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


downsampling_layers = [
    nn.Conv2d(3, 64, 3, 1),
    norm(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 4, 2, 1),
    norm(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 4, 2, 1),
]

feature_layers = [ODEBlock(ODEfunc(64))]


# # Create model

# +
# Efficient net b5
# @sn.snoop()
class LitModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, weight_decay=0.0001):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

#         self.enet = EfficientNet.from_pretrained('efficientnet-b5',
#                                                  num_classes=self.num_classes)
        self.fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, self.num_classes)]
        self.enet = nn.Sequential(*downsampling_layers, *feature_layers, *self.fc_layers)
#         in_features = self.enet._fc.in_features
#         self.enet._fc = nn.Linear(in_features, num_classes)

#     @sn.snoop()

    def forward(self, x):
        out = self.enet(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=2,
                                                    gamma=0.1)

        return ([optimizer], [scheduler])



    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["x"], train_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        #         loss.requires_grad = True
        acc = accuracy(preds, y)
        self.log('train_acc_step', acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["x"], val_batch["y"]
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        #         loss.requires_grad = True
        acc = accuracy(preds, y)
        self.log('val_acc_step', acc)
        self.log('val_loss', loss)
#         print(preds.detach().cpu(), y.detach().cpu())
        y2 = torch.argmax(preds, dim = 1)
        conf = confusion_matrix(y2.detach().cpu(), y.detach().cpu())
        fig = plt.figure()
        plt.imshow(conf)
#         plt.show()
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)


# -

class ImageClassDs(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 imfolder: str,
                 train: bool = True,
                 transforms=None):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = self.df.iloc[index]['image_id']
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if (self.transforms):
            x = self.transforms(image=x)['image']

        y = self.df.iloc[index]['label']
        return {
            "x": x,
            "y": y,
        }

    def __len__(self):
        return len(self.df)


# # Load data

class ImDataModule(pl.LightningDataModule):
    def __init__(
            self,
            df,
            batch_size,
            num_classes,
            data_dir: str = "/media/hdd/Datasets/asl/",
            img_size=(256, 256)):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = A.Compose([
            A.RandomResizedCrop(img_size, img_size, p=1.0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2,
                                 sat_shift_limit=0.2,
                                 val_shift_limit=0.2,
                                 p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                       contrast_limit=(-0.1, 0.1),
                                       p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ],
            p=1.)

        self.valid_transform = A.Compose([
            A.CenterCrop(img_size, img_size, p=1.),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            ToTensorV2(p=1.0),
        ],
            p=1.)

    def setup(self, stage=None):
        dfx = pd.read_csv("./train_folds.csv")
        train = dfx.loc[dfx["kfold"] != 1]
        val = dfx.loc[dfx["kfold"] == 1]

        self.train_dataset = ImageClassDs(train,
                                          self.data_dir,
                                          train=True,
                                          transforms=self.train_transform)

        self.valid_dataset = ImageClassDs(val,
                                          self.data_dir,
                                          train=False,
                                          transforms=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=12,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=12)


batch_size = 128
# num_classes = 29
num_classes = 3
img_size = 64

dm = ImDataModule(df,
                  batch_size=batch_size,
                  num_classes=num_classes,
                  img_size=img_size)
class_ids = dm.setup()

# # Logs

model = LitModel(num_classes)

logger = TensorBoardLogger("logs/")

trainer = pl.Trainer(auto_select_gpus=True,
                     gpus=1,
                     precision=16,
                     profiler=False,
                     max_epochs=50,
                     callbacks=[pl.callbacks.ProgressBar()],
                     enable_pl_optimizer=True,
                     logger=logger,
                     accelerator='ddp',
                     plugins='ddp_sharded')

trainer.fit(model, dm)

# +
trainer.test()

trainer.save_checkpoint('model1.ckpt')
# -

# # Inference

best_checkpoints = trainer.checkpoint_callback.best_model_path

pre_model = LitModel.load_from_checkpoint(checkpoint_path= best_checkpoints).to("cuda")

pre_model.eval()
pre_model.freeze()

transforms = A.Compose([
            A.CenterCrop(img_size, img_size, p=1.),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            ToTensorV2(p=1.0),
        ],
            p=1.)


test_img = transforms(image=cv2.imread("/media/hdd/Datasets/asl/asl_alphabet_test/asl_alphabet_test/C_test.jpg"))

y_hat = pre_model(test_img["image"].unsqueeze(0).to("cuda"))

label_map

label_map[int(torch.argmax(y_hat, dim = 1))]
















