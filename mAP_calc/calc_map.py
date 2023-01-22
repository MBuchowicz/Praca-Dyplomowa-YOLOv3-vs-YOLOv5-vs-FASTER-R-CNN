import json
import argparse
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# todo frcnn ma inny numer dla person
# todo yolov5 ma inna nazwe dla probability
# todo dodac wyswietlanie obrazkow
class CalcMap:
    class_variable = 0  # class variable moze byc nadpisana
    num_of_instances = 0
    metric = MeanAveragePrecision()

    def __init__(self, gt, preds):
        self.gt = gt  # moze byc lista jsonow
        self.preds = preds
        CalcMap.num_of_instances += 1
        self.metric = MeanAveragePrecision() #iou_thresholds=list([0.05, 0.2, 0.45, 0.5])

    # def json_dump_gt(self):
    #     return json.dumps(self.gt, indent=2)
    #
    # def json_dump_preds(self):
    #     return json.dumps(self.preds, indent=2)

    def calculate_mAP(self):
        if type(self.gt) is list:
            for j in range(len(self.gt)):
                for img_filename in self.gt[j]['images']:
                    img_id = img_filename['id']
                    gt_boxes = torch.zeros(1, 4)
                    gt_labels = torch.zeros(1)
                    for item in self.gt[j]['annotations']:
                        if item['image_id'] == img_id:  # szuka w jsonie annotations dla image
                            gt_bbox = item['bbox']
                            category_id = item['category_id'] - 1
                            box0 = torch.tensor(
                                [[int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[0] + gt_bbox[2]),
                                  int(gt_bbox[1] + gt_bbox[3])]])
                            label0 = torch.tensor([category_id])
                            if gt_boxes[0, 1] == 0:
                                gt_boxes = torch.add(gt_boxes, box0)
                                gt_labels = torch.add(gt_labels, label0)
                            else:
                                gt_boxes = torch.cat((gt_boxes, box0), 0)
                                gt_labels = torch.cat((gt_labels, label0), 0)
                    for i in self.preds[j]:
                        if img_filename['file_name'] == i['filename']:
                            test = i['filename']
                            # print("TAAAAK!")
                            preds_bboxes = i['bb']
                            preds_boxes = torch.zeros(1, 4)
                            preds_scores = torch.zeros(1)
                            preds_labels = torch.zeros(1)
                            for preds_bbox in preds_bboxes:
                                if preds_bbox['id'] == 0:
                                    box1 = torch.tensor(
                                        [[preds_bbox['xmin'], preds_bbox['ymin'], preds_bbox['xmax'],
                                          preds_bbox['ymax']]])
                                    score1 = torch.tensor([preds_bbox['probability']])
                                    label1 = torch.tensor([preds_bbox['id']], )
                                    if preds_boxes[0, 1] == 0:
                                        preds_boxes = torch.add(preds_boxes, box1)
                                        preds_scores = torch.add(preds_scores, score1)
                                        preds_labels = torch.add(preds_labels, label1)
                                    else:
                                        preds_boxes = torch.cat((preds_boxes, box1), 0)
                                        preds_scores = torch.cat((preds_scores, score1), 0)
                                        preds_labels = torch.cat((preds_labels, label1), 0)

                    if test == img_filename['file_name']:
                        # print(img_filename['file_name'], ' ')
                        target = [
                            dict(
                                boxes=gt_boxes,
                                labels=gt_labels,
                            )]
                        # print('target: ', target, ' ')
                        preds = [
                            dict(
                                boxes=preds_boxes,
                                scores=preds_scores,
                                labels=preds_labels,
                            )
                        ]
                        # print('preds: ', preds, ' ')
                        # print(' ')

                        self.metric.update(preds, target)
        else:
            self._calculate_mAP_single()

    def _calculate_mAP_single(self):
        for img_filename in self.gt['images']:
            img_id = img_filename['id']
            gt_boxes = torch.zeros(1, 4)
            gt_labels = torch.zeros(1)
            for item in self.gt['annotations']:
                if item['image_id'] == img_id:  # szuka w jsonie annotations dla image
                    gt_bbox = item['bbox']
                    category_id = item['category_id'] - 1
                    box0 = torch.tensor(
                        [[int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[0] + gt_bbox[2]),
                          int(gt_bbox[1] + gt_bbox[3])]])
                    label0 = torch.tensor([category_id])
                    if gt_boxes[0, 1] == 0:
                        gt_boxes = torch.add(gt_boxes, box0)
                        gt_labels = torch.add(gt_labels, label0)
                    else:
                        gt_boxes = torch.cat((gt_boxes, box0), 0)
                        gt_labels = torch.cat((gt_labels, label0), 0)
            for i in self.preds:
                if img_filename['file_name'] == i['filename']:
                    test = i['filename']
                    # print("TAAAAK!")
                    preds_bboxes = i['bb']
                    preds_boxes = torch.zeros(1, 4)
                    preds_scores = torch.zeros(1)
                    preds_labels = torch.zeros(1)
                    for preds_bbox in preds_bboxes:
                        if preds_bbox['id'] == 0:
                            box1 = torch.tensor(
                                [[preds_bbox['xmin'], preds_bbox['ymin'], preds_bbox['xmax'], preds_bbox['ymax']]])
                            score1 = torch.tensor([preds_bbox['probability']])
                            label1 = torch.tensor([preds_bbox['id']], )
                            if preds_boxes[0, 1] == 0:
                                preds_boxes = torch.add(preds_boxes, box1)
                                preds_scores = torch.add(preds_scores, score1)
                                preds_labels = torch.add(preds_labels, label1)
                            else:
                                preds_boxes = torch.cat((preds_boxes, box1), 0)
                                preds_scores = torch.cat((preds_scores, score1), 0)
                                preds_labels = torch.cat((preds_labels, label1), 0)

            if test == img_filename['file_name']:
                # print(img_filename['file_name'], ' ')
                target = [
                    dict(
                        boxes=gt_boxes,
                        labels=gt_labels,
                    )]
                # print('target: ', target, ' ')
                preds = [
                    dict(
                        boxes=preds_boxes,
                        scores=preds_scores,
                        labels=preds_labels,
                    )
                ]
                # print('preds: ', preds, ' ')
                # print(' ')

                self.metric.update(preds, target)

    def compute_mAP(self):
        return self.metric.compute()


# parser = argparse.ArgumentParser()
# parser.add_argument('--gt', '-g', required=True)
# parser.add_argument('--preds', '-p', required=True)
# args = parser.parse_args()

with open("data/gt_speed_test.json") as file:
    gt_data = json.load(file)

with open('data/refactored_speed_test.json') as file:
    yolov3_preds = json.load(file)

with open("data/frcnn_rf_speed_test.json") as file:
    frcnn_preds = json.load(file)

with open("data/yolov5_rf_speed_test.json") as file:
    yolov5_preds = json.load(file)

gt_data_big = [gt_data, gt_data, gt_data]
preds_data_big = [yolov3_preds, frcnn_preds, yolov5_preds]

combined = CalcMap(gt_data_big, preds_data_big)

yolov3 = CalcMap(gt_data, yolov3_preds)

faster_rcnn = CalcMap(gt_data, frcnn_preds)

yolov5 = CalcMap(gt_data, yolov5_preds)
# print(yolov3.json_dump_gt())

# print(CalcMap.num_of_instances)
print("yolov3: ")
yolov3.calculate_mAP()
yolov3_results = yolov3.compute_mAP()
pprint(yolov3_results)
print("")

print("frcnn: ")
faster_rcnn.calculate_mAP()
frcnn_results=faster_rcnn.compute_mAP()
pprint(frcnn_results)
print("")
#
print("yolov5: ")
yolov5.calculate_mAP()
yolov5_results = yolov5.compute_mAP()
pprint(yolov5_results)
print("")


# combined.calculate_mAP()
# pprint(combined.compute_mAP())
# print(CalcMap.num_of_instances)

# set width of bar
barWidth = 0.2
fig, ax = plt.subplots(figsize=(12, 8))

# set height of barA
YoloV3 = [float(yolov3_results['map']), float(yolov3_results['map_50']), float(yolov3_results['mar_10']), float(yolov3_results['mar_small']), float(yolov3_results['map_small'])]
FasterRCNN = [float(frcnn_results['map']), float(frcnn_results['map_50']), float(frcnn_results['mar_10']), float(frcnn_results['mar_small']), float(frcnn_results['map_small'])]
YoloV5 = [float(yolov5_results['map']), float(yolov5_results['map_50']), float(yolov5_results['mar_10']), float(yolov5_results['mar_small']), float(yolov5_results['map_small'])]

# Set position of bar on X axis
br1 = np.arange(len(YoloV3))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, YoloV3, color='tab:orange', width=barWidth,
        edgecolor='grey', label='YoloV3')
plt.bar(br2, FasterRCNN, color='tab:green', width=barWidth,
        edgecolor='grey', label='FasterRCNN')
plt.bar(br3, YoloV5, color='tab:purple', width=barWidth,
        edgecolor='grey', label='YoloV5')

ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

# Adding Xticks
plt.xlabel('Metryka', fontweight='bold', fontsize=15)
plt.ylabel('Wartość', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(YoloV3))],
           ['mAP', 'mAP_50', 'mAR_10', 'mAR_small', 'mAP_small'])

plt.legend()

plt.savefig("mAP_speed_test.png")
plt.show()