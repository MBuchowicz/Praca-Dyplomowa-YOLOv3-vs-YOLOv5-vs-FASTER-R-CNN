import cv2
import torch
from PIL import Image
import json
from tqdm import tqdm
import argparse
import os
import glob
import pandas
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', '-i', default='img')
    parser.add_argument('--datadir', '-d', default='data')
    parser.add_argument('--output', '-o', default='data_yolov5_adjusted.json')
    args = parser.parse_args()

    filenames = sorted(glob.glob(os.path.join(args.imgdir, "*.png")))
    annotations = []

    for filename in tqdm(filenames):
        image = cv2.imread(filename)
        results = model(image, size=640)
        #results.print()
        results_pd=results.pandas()
        results_xyxy = results.pandas().xyxy[0]
        annot_bboxes = []

        for row in range(results_xyxy.shape[0]):
            bbox = results_xyxy.iloc[row]
            if bbox['confidence'] >= 0.5:
                # print('row: ', row, '   ', results_xyxy.iloc[row])
                # print(bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
                # print(' ')
                xmin = bbox['xmin'].astype('int32')
                ymin = bbox['ymin'].astype('int32')
                xmax = bbox['xmax'].astype('int32')
                ymax = bbox['ymax'].astype('int32')

                cv2.rectangle(image,(xmin, ymin) ,(xmax, ymax) , (255,0,0), 1)
                cv2.putText(image, bbox['name'], (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2,
                            lineType=cv2.LINE_AA)
                bbox_save= {
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'probability': bbox['confidence'], #FIXME moja konwencja to probability jako score, confidence
                'Class': bbox['name'],
                'id': int(bbox['class'])
                }
                annot_bboxes.append(bbox_save)
        annotations.append({
            "filename": os.path.basename(filename),
            "bb": annot_bboxes
        })




        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('img', image_rgb)




        if cv2.waitKey(100) == 27:
            break
    cv2.destroyAllWindows()

    with open(args.output, 'w') as out_file:# zapisywanie do jsona
        json.dump(annotations, out_file, indent=2)

if __name__ == '__main__':
    main()
