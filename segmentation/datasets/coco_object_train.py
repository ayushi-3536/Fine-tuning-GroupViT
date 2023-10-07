# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from mmseg.datasets import DATASETS, CustomDataset
from collections import OrderedDict
from mmcv.utils import print_log
from prettytable import PrettyTable
import mmcv
import numpy as np
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

@DATASETS.register_module()
class COCOObjectDataset(CustomDataset):
    """COCO-Object dataset.

    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    SYNONYMS = {
                "background": ["background"],
                "aeroplane": ["aeroplane","airplane","plane","aircraft","jet","airliner","jetliner","jetplane","airbus"],
                "bicycle": ["bicycle","bike","cycle","pedal bike"],
                "bird": ["bird","avian","fowl"],
                "boat": ["boat","vessel", "watercraft", "ship", "yacht", "canoe","sailboat"],
                "bottle": ["bottle","bottleful","flask","flagon","jar","jug"],
                "bus": ["bus"],
                "car": ["car","auto","automobile","motorcar","vehicle"],
                "cat": ["cat","feline","felid","kitten"],
                "chair": ["chair","seat"],
                "cow": ["cow","bovine","cattle","kine","ox","oxen","bison","buffalo"],
                "diningtable": ["diningtable","dining table","dining-room table","kitchen table","table"],
                "dog": ["dog","canine","canid","puppy","hound"],
                "horse": ["horse","equine","pony"],
                "motorbike": ["motorbike","motorcycle","motor cycle","moto","moped","mop"],
                "person": ["person","man","woman","adult","people","someone","kid","guy","boy","girl","baby","child","human"],
                "pottedplant": ["pottedplant","houseplant","plant","potted plant"],
                "sheep": ["sheep","lamb","ewe","ram","wether","hogget","mutton"],
                "sofa": ["sofa","couch","lounge","lounge chair","lounge suite","sofa bed","sofabed"],
                "train": ["train","tram","subway","metro"],
                "tvmonitor": ["tvmonitor","tv","tv screen","television"]
                }
    
    PALETTE = [[0, 0, 0], [0, 192, 64], [6, 230, 230], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
               [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
               [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
               [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
               [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
               [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
               [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
               [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
               [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0],
               [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]]

    def __init__(self, **kwargs):
        super(COCOObjectDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)

    # def evaluate(self,
    #              results,
    #              metric='mIoU',
    #              logger=None,
    #              gt_seg_maps=None,
    #              **kwargs):
    #     """Evaluate the dataset.

    #     Args:
    #         results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
    #              results or predict segmentation map for computing evaluation
    #              metric.
    #         metric (str | list[str]): Metrics to be evaluated. 'mIoU',
    #             'mDice' and 'mFscore' are supported.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.
    #         gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
    #             used in ConcatDataset

    #     Returns:
    #         dict[str, float]: Default metrics.
    #     """
    #     classnames = self.CLASSES

    #     #if add_synonyms and dataset.synonyms is not None then add for every unique value in dataset.synonyms dict a new class
    #     for key, value in self.SYNONYMS.items():
    #         if value not in classnames:
    #             #change list to tuple
    #             value = tuple(value)
    #             classnames = classnames + value
    #         #keep only unique values
    #         classnames = list(set(classnames))
        
    #     self.CLASSES = classnames
    #     print("evaluating on :", self.CLASSES)
        
    #     if isinstance(metric, str):
    #         metric = [metric]
    #     allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    #     if not set(metric).issubset(set(allowed_metrics)):
    #         raise KeyError('metric {} is not supported'.format(metric))

    #     eval_results = {}
    #     # test a list of files
    #     if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
    #             results, str):
    #         if gt_seg_maps is None:
    #             gt_seg_maps = self.get_gt_seg_maps()
    #         num_classes = len(self.CLASSES)
    #         ret_metrics = eval_metrics(
    #             results,
    #             gt_seg_maps,
    #             num_classes,
    #             self.ignore_index,
    #             metric,
    #             label_map=self.label_map,
    #             reduce_zero_label=self.reduce_zero_label)
    #     # test a list of pre_eval_results
    #     else:
    #         ret_metrics = pre_eval_to_metrics(results, metric)

    #     # Because dataset.CLASSES is required for per-eval.
    #     if self.CLASSES is None:
    #         class_names = tuple(range(num_classes))
    #     else:
    #         class_names = self.CLASSES

    #     # summary table
    #     ret_metrics_summary = OrderedDict({
    #         ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
    #         for ret_metric, ret_metric_value in ret_metrics.items()
    #     })

    #     # each class table
    #     ret_metrics.pop('aAcc', None)
    #     ret_metrics_class = OrderedDict({
    #         ret_metric: np.round(ret_metric_value * 100, 2)
    #         for ret_metric, ret_metric_value in ret_metrics.items()
    #     })
    #     ret_metrics_class.update({'Class': class_names})
    #     ret_metrics_class.move_to_end('Class', last=False)

    #     # for logger
    #     class_table_data = PrettyTable()
    #     for key, val in ret_metrics_class.items():
    #         class_table_data.add_column(key, val)

    #     summary_table_data = PrettyTable()
    #     for key, val in ret_metrics_summary.items():
    #         if key == 'aAcc':
    #             summary_table_data.add_column(key, [val])
    #         else:
    #             summary_table_data.add_column('m' + key, [val])

    #     print_log('per class results:', logger)
    #     print_log('\n' + class_table_data.get_string(), logger=logger)
    #     print_log('Summary:', logger)
    #     print_log('\n' + summary_table_data.get_string(), logger=logger)

    #     # each metric dict
    #     for key, value in ret_metrics_summary.items():
    #         if key == 'aAcc':
    #             eval_results[key] = value / 100.0
    #         else:
    #             eval_results['m' + key] = value / 100.0

    #     ret_metrics_class.pop('Class', None)
    #     for key, value in ret_metrics_class.items():
    #         eval_results.update({
    #             key + '.' + str(name): value[idx] / 100.0
    #             for idx, name in enumerate(class_names)
    #         })

    #     return eval_results