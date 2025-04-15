import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN

import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes=11):
        super(MyModel, self).__init__()

        self.num_classes = num_classes
        # Load pretrained ResNet-152
        backbone = resnet_fpn_backbone(
            backbone_name="resnet152",
            weights=torchvision.models.ResNet152_Weights.DEFAULT
        )

        # Output channels from resnet101's last conv layer

        # Create Faster R-CNN model with this backbone
        self.model = FasterRCNN(backbone=backbone,
                                num_classes=self.num_classes)

        # Load pretrained weights from ResNet-50 model
        pretrained_model = torchvision.models. \
            detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

        # Copy RPN and ROI head weights (optional, partial weight transfer)
        self.model.rpn.load_state_dict(pretrained_model.rpn.state_dict())
        self.model.roi_heads.box_head.\
            load_state_dict(pretrained_model.roi_heads.box_head.state_dict())

        # Replace the final predictor with one that matches your num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            FastRCNNPredictor(in_features, num_classes)

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
