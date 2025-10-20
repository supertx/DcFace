
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
from collections import namedtuple
from skimage import transform as trans
from model import get_model
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
from torch.nn import DataParallel

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class IJBDataset(Dataset):

    def __init__(self, img_path, img_list_path):
        super().__init__()
        with open(img_list_path, "r") as f:
            self.table = pd.read_table(f, sep=' ', header=None)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            src = np.array(
                [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                 [33.5493, 92.3655], [62.7299, 92.2041]],
                dtype=np.float32)
            src[:, 0] += 8.0
            self.src = src
            self.img_path = img_path
            self.image_size = [112, 112]

    def __getitem__(self, index):
        img_path = self.table.iloc[index, 0]
        img = cv2.imread(os.path.join(self.img_path, img_path))
        landmark = self.table.iloc[index,
                                   1:-1].values.astype(np.float32).reshape(
                                       -1, 2)
        tform = trans.SimilarityTransform()
        tform.estimate(landmark, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        sample = self.transform(img)
        sample_flip = self.transform(img_flip)
        return sample, sample_flip

    def __len__(self):
        return len(self.table)


class Embedding:

    def __init__(self, prefix, ctx_id=0):
        if isinstance(prefix, str):
            self.device = torch.device("cuda:{}".format(ctx_id))
            model = get_model("MobileNetV2").to(self.device)
            # model = get_model("ResNet100").to(self.device)
            self.model = DataParallel(model)
            state_dict = torch.load(prefix, map_location=self.device)
            self.model.module.load_state_dict(state_dict)
        else:
            self.model = prefix
        self.model.eval()

    def get(self, img_path, img_list_path):
        ijb_dataset = IJBDataset(img_path, img_list_path)
        dataloader = DataLoader(ijb_dataset,
                                batch_size=512,
                                shuffle=False,
                                num_workers=4)
        feats = []
        t = tqdm(total=len(dataloader),
                 leave=False,
                 desc="Extracting features...",
                 ncols=100)
        with torch.no_grad():
            for i, (img, img_flip) in enumerate(dataloader):
                feat = self.model(img.cuda()).cpu().numpy()
                feat_flip = self.model(img_flip.cuda()).cpu().numpy()
                feat = feat + feat_flip
                t.update(1)
                feats.extend(feat)
        t.close()
        return feats
