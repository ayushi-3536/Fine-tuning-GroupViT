# -------------------------------------------------------------------------
# Written by Jilan Xu
# -------------------------------------------------------------------------

from re import L
import torch
import json
import os.path as osp
import requests
import numpy as np
import time
from typing import List
from .base_dataset import BaseDataset
# from prototype.data.image_reader import build_image_reader
from .image_reader import build_image_reader
# import linklink as link
import random
import os
import omegaconf
import clip
from .imagenet_template import full_imagenet_templates
from nltk.stem import WordNetLemmatizer
from PIL import Image

from .tokenizer import SimpleTokenizer
# lemmatizer = WordNetLemmatizer()

# ### frequently appeared 100 entities ###
# TOP_CLASSES_1=[
#     'people', 'man', 'men', 'woman', 'women', 'girl', 'boy', 'lady', 'kid', 'child', 'children', 'baby', 'student', 'bride', 'groom', 'couple', 'prince', 'princess', \
#     'car', 'bus', 'truck', 'motorcycle', 'train', 'bicycle', 'boat', 'aeroplane', 'airplane', 'motorbike', 'bike',\
#     'cup', 'bottle', 'bowl', 'knife', 'spoon',  'glass', 'fork',\
#     'chair', 'table', 'bench', 'clock', 'laptop', 'light', 'vase', 'plant', 'remote', 'microwave', 'toaster', 'oven','mouse', 'keyboard','sofa', 'monitor','desk', 'tv','TV', 'couch', 'flower','refrigerator', \
#     'house', 'building', 'hotel',\
#     'handbag', 'umbrella','book', 'backpack', 'phone', 'shirt', 'tie', 'suitcase','T-shirt', 'bag',  'box', \
#     'sink','bed','toilet',\
#     'cat','dog',  'horse', 'bird','cow', 'sheep' ,'elephant', 'bear', 'zebra', 'giraffe', \
#     'ball', 'racket', 'skateboard', 'skis', 'snowboard', 'surfboard', 'kite', \
#     'pizza', 'cake', 'apple', 'banana', 'sandwich', 'orange', 'carrot', 'donut' ,\
# ]

# ### some of the entities are similar, map them to a single one ###
# syn_dict = {
#     'people':'people', 'man':'people', 'men':'people', 'woman':'people', 'women':'people', 'girl':'people', 'boy':'people', 'lady':'people', 'kid':'people', 'child':'people', 'children':'people', 'baby':'people', 'student':'people', 'bride':'people', 'groom':'people', 'couple':'people', 'prince':'people', 'princess':'people',\
#     'airplane': 'aeroplane','motorbike': 'motorcycle','bike': 'bicycle',\
#     'TV':'tv', 'desk': 'table', 'couch':'sofa',\
#     'building': 'house', 'hotel': 'house', \
#     'T-shirt': 'shirt','T-Shirt': 'shirt', 'handbag': 'bag', \
# }

# ### unique entities ###
# TOP_UNIQUE_CLASSES = [
#     'people', 'car', 'bus', 'truck', 'motorcycle', \
#     'train', 'bicycle', 'boat', 'aeroplane', 'cup', \
#     'bottle', 'bowl', 'knife', 'spoon',  'glass', \
#     'fork', 'chair', 'table', 'bench', 'clock', \
#     'laptop', 'light', 'vase', 'plant', 'remote',\
#     'microwave', 'toaster', 'oven','mouse', 'keyboard',\
#     'sofa', 'monitor', 'tv', 'flower','refrigerator', \
#     'house', 'bag', 'umbrella','book', 'backpack', \
#     'phone', 'shirt', 'tie', 'suitcase', 'box',\
#     'sink','bed','toilet', 'cat','dog', \
#     'horse', 'bird','cow', 'sheep' ,'elephant', \
#     'bear', 'zebra', 'giraffe',  'ball', 'racket', \
#     'skateboard', 'skis', 'snowboard', 'surfboard', 'kite',\
#     'pizza', 'cake', 'apple', 'banana', 'sandwich',\
#     'orange', 'carrot', 'donut' ,\
# ]

# TOP_UNIQUE_CLASSES_IDX = {}
# for i, x in enumerate(TOP_UNIQUE_CLASSES):
#     TOP_UNIQUE_CLASSES_IDX[x] = i

class CLIPDataset(BaseDataset):
    """
    Clip Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'
        - topnoun: 'none' / 'coco_top50' / 'cc3m_top50' / ...
    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG", "label": 0, "label_name": "dog"}\n"
    """

    def __init__(self, root_dir, meta_file, img_transform=None, text_transform=None,
                evaluator=None, image_reader_type='pil',
                 fseek=False, split='train', multi_label=5):
        if not isinstance(meta_file, List) and not isinstance(meta_file, omegaconf.listconfig.ListConfig):
            meta_file = [meta_file]
        if not isinstance(root_dir, List) and not isinstance(meta_file, omegaconf.listconfig.ListConfig):
            root_dir = [root_dir]

        self.meta_file = meta_file
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)

        self.fseek = fseek
        self.initialized = False
        self.num = 0
        self.split=split
        self.tokenizer = SimpleTokenizer()    

        self.metas = []
        self.multi_label = multi_label

        ### fseek uses file seek to load each line with pointer online ###
        ### this saves the memory while adding the loading time ###
        if self.fseek:
            self.line_offsets = []
            for each_meta_file in meta_file:
                line_offset = []
                offset = 0
                with open(each_meta_file) as f:
                    for line in f:
                        line_offset.append(offset)
                        offset += len(line.encode('UTF-8'))
                    f.close()
                self.num += len(line_offset)
                self.line_offsets.append(line_offset)
        else:
            ### read from local file and load all metafile info ###
            for rd, each_meta_file in zip(root_dir, meta_file):
                with open(each_meta_file) as f:
                    lines = f.readlines()
                self.num += len(lines)

                for line in lines:
                    info = json.loads(line)
                    filename = osp.join(rd, info['filename'])
                    ### add root_dir to filename ###
                    info['filename'] = filename
                    self.metas.append(info)

        super(CLIPDataset, self).__init__(root_dir=root_dir,
                                          meta_file=meta_file,
                                          transform=img_transform,
                                          evaluator=evaluator)


    def __len__(self):        
        return self.num

    def _str2list(self, x):
        if type(x) is list:
            return x
        elif type(x) is str:
            return [x]
        else:
            raise RuntimeError(
                "unknown value for _str2list: {}".format(type(x)))

    def _load_meta(self, idx):
        if self.fseek:
            source_id = 0
            while idx >= len(self.line_offsets[source_id]):
                idx -= len(self.line_offsets[source_id])
                source_id += 1 #fixed
            with open(self.meta_file[source_id]) as f:
                f.seek(self.line_offsets[source_id][idx])
                line = f.readline()
                meta = json.loads(line)
                filename = osp.join(self.root_dir[source_id], meta['filename'])
                meta['filename'] = filename
                f.close()
            return meta
        else:
            return self.metas[idx]
         

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = curr_meta['filename']
        caption = curr_meta['caption'] if 'caption' in curr_meta else ''
        ret_info = {}

        #############

        #try:
            #assert self.is_contains_chinese(caption) == False
            #if self.read_from == 'dir':
        image = Image.open(filename).convert('RGB')
            # else:
            #     ### load from bytes ###
            # img_bytes = self.read_file(curr_meta)
            # img = self.image_reader(img_bytes, filename)
            
        if self.img_transform is not None:
            image = self.img_transform(image)
                    
            #if self.text_transform is not None:
        if self.multi_label >0:
            texts, nouns = self.text_transform(caption)
        else:
            texts = self.text_transform(caption)



        ret_info['image'] = image
        ret_info['text'] = texts
        #ret_info['filename'] = filename
        if self.multi_label >0:
            ret_info['text_meta'] = nouns
        return ret_info
                        
        # except Exception as e:          
        #     print(e)
            # return self.__getitem__(0)
    

  
    
