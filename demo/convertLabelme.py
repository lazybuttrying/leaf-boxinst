from copy import deepcopy
import os
import numpy as np
import torch
import json

from detectron2.utils.visualizer import GenericMask
from detectron2.structures.masks import PolygonMasks
# https://detectron2.readthedocs.io/en/latest/_modules/detectron2/utils/visualizer.html


# https://www.kaggle.com/code/corochann/vinbigdata-detectron2-prediction/notebook

# https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/instances.html

lableme = {"version":"5.0.1", "flags":{}, "shapes":[],
    "imagePath":"", "imageData":None, "imageHeight":0, "imageWidth":0}
shape = {"label":"", "points":[], 
    "group_id":None, "shape_type": "polygon", "flags":{}}

classLabel = ["big", "small"]

def convertInstanceToLabelme(path, fname, instances):
    if len(instances) == 0:
        # No finding
        return {}
    
    # # Find some bbox...
    # # print(f"index={index}, find {len(instances)} bbox.")
    # fields: Dict[str, Any] = instances.get_fields()
    # pred_classes = fields["pred_classes"]  # (n_boxes,)
    # pred_scores = fields["scores"]
    # # shape (n_boxes, 4). (xmin, ymin, xmax, ymax)
    # pred_boxes = fields["pred_boxes"].tensor
    # locations = fields["locations"]


    # # h_ratio = dim0 / instances.image_height
    # # w_ratio = dim1 / instances.image_width
    # # pred_boxes[:, [0, 2]] *= w_ratio
    # # pred_boxes[:, [1, 3]] *= h_ratio

    # pred_classes_array = pred_classes.cpu().numpy()
    # pred_boxes_array = pred_boxes.cpu().numpy()
    # pred_scores_array = pred_scores.cpu().numpy()

    # return {
    #     "image": {"width": instances.image_size[1], "height": instances.image_size[0]}
    #     "score": pred_scores,
    #     "classes" : pred_classes,
    #     "boxes": pred_boxes,

    # }
    labelmeTemp = deepcopy(lableme)
    img_height, img_width = instances.image_size

    labelmeTemp["imagePath"] = fname
    labelmeTemp["imageHeight"] = img_height
    labelmeTemp["imageWidth"] = img_width

    # print(instances.pred_classes.shape, instances.pred_masks.shape)
    # torch.Size([14]) torch.Size([14, 1080, 1318])

    mask = np.asarray(instances.pred_masks.cpu())
    mask = [GenericMask(x, img_height, img_width) for x in mask]
    polygons = [mask[i].polygons for i in range(len(mask))]

    #  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/masks.html

    for a, b in zip(instances.pred_classes, polygons):
        shapeTemp = deepcopy(shape)
        shapeTemp["label"] = classLabel[a.item()]
        shapeTemp["points"] = list(max(b, key = lambda x:len(x)))
        shapeTemp["points"] = [[x,y] 
            for x,y in zip(shapeTemp["points"][::2],shapeTemp["points"][1::2])]
        labelmeTemp["shapes"].append(shapeTemp)

    path = os.path.join(path, fname[:-4])+ ".json"
    with open(path, 'w') as outfile:
        json.dump(labelmeTemp, outfile, indent=4)
