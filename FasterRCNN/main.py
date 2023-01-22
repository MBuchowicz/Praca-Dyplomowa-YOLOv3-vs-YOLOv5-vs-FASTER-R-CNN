import glob
import json

from PIL import Image
import os
import argparse
import cv2

import torch
import torchvision
from tqdm import tqdm

import detect_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', '-i', default='img')
    parser.add_argument('--datadir', '-d', default='data')
    parser.add_argument('--output', '-o', default='faster_rcnn_adjusted.json')
    args = parser.parse_args()

    # download or load the model from disk
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 min_size=800)  # args['min_size'])
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    model.eval().to(device)
    filenames = sorted(glob.glob(os.path.join(args.imgdir, "*.png")))
    annotations = []
    for filename in tqdm(filenames):
        image = Image.open(filename)
        # img = np.array(pic)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, classes, labels, scores = detect_utils.predict(image, model, device, 0.8)
        annot_bboxes = []
        for index, box in enumerate(boxes):
            #labels[index].item() = labels[index].item() - 1
            bbox = {
                'xmin': int(box[0]),
                'ymin': int(box[1]),
                'xmax': int(box[2]),
                'ymax': int(box[3]),
                'probability': float(scores[index]),
                'Class': classes[index],
                'id': labels[index].item() - 1 # FIXME konwencja person: id =0 więc odejmuje 1
            }
            annot_bboxes.append(bbox)
        image = detect_utils.draw_boxes(boxes, classes, labels, image)
        annotations.append({
            "filename": os.path.basename(filename),
            "bb": annot_bboxes
        })

        cv2.imshow('img', image)

        # ww_filename = os.path.split(filename)
        # w_filename = os.path.join(ww_filename[0], "rcnn__" + ww_filename[1])
        # print('head = ',ww_filename[0])
        # print("tail = ",ww_filename[1])
        # print(w_filename)
        # cv2.imwrite(w_filename, image)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

    with open(args.output, 'w') as out_file:  # zapisywanie do jsona
        json.dump(annotations, out_file, indent=2)

    print(annotations)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
