import os.path as osp
import glob
import os
import numpy as np
import torch
from PIL import Image

import transforms as T

class make_datapath_list():
  def __init__(self, rootpath):
    """
    rootpath: Absolute path
    """
    img_file_path = sorted(glob.glob(rootpath+ '/kaggle_3m/TCGA*'))
    self.train_file_path = img_file_path[:75]
    self.val_file_path = img_file_path[75:95]
    self.test_file_path = img_file_path[95:]
  
  def get_list(self, path_type):
    """
    path_type: select a path type from "train", "val" and "test"
    """
    if path_type=="train":
      file_path = self.train_file_path

    elif path_type=="val":
      file_path = self.val_file_path

    else:
      file_path = self.test_file_path

    img_list = []
    anno_list = []
    for path in file_path:
      path = glob.glob(path+"/*.tif")
      img_path = sorted([p for p in path if "mask" not in p])
      anno_path = [p[:-4]+"_mask.tif" for p in img_path]
      img_list += img_path
      anno_list += anno_path

    return img_list, anno_list

class BrainDataset(object):
  def __init__(self, img_path_list, anno_path_list, transforms=None):
    self.img_path_list = img_path_list
    self.anno_path_list = anno_path_list
    self.transforms = transforms

  def __getitem__(self, idx):

    img_path = self.img_path_list[idx]
    anno_path = self.anno_path_list[idx]

    img = Image.open(img_path).convert("RGB")

    mask = Image.open(anno_path)
    mask = np.array(mask)

    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]

    masks = mask == obj_ids[:, None, None]

    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    image_id = torch.tensor([idx])
    if len(boxes)!=0:
      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
      area = torch.as_tensor([], dtype=torch.int64)
    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    if self.transforms is not None:
        img, target = self.transforms(img, target)

    return img, target

  def __len__(self):
    return len(self.img_path_list)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)