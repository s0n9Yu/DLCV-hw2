import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes=11):
        super(MyModel, self).__init__()

        self.num_classes = num_classes
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained = True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
    @staticmethod
    def convert_targets(boxes, labels, image_ids=None):
        targets = []
        for i, (box, label) in enumerate(zip(boxes, labels)):
            target = {}
            target["boxes"] = box
            target["labels"] = label
            if image_ids is not None:
                target["image_id"] = image_ids[i]
            targets.append(target)
        return targets