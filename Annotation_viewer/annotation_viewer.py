import os
import argparse
import glob
import json
import logging

import cv2

logging.basicConfig()
logger = logging.getLogger(os.path.basename(__file__))


def draw_img_annot_data(data, filename, img, color):
    for item in data:
        if item['filename'] == os.path.basename(filename):
            bboxes = item['bb']
            print(len(bboxes))
            for bbox in bboxes:
                logger.debug(bbox)
                cv2.rectangle(img,
                              (bbox['xmin'], bbox['ymin']),
                              (bbox['xmax'], bbox['ymax']),
                              color,
                              1)
                cv2.putText(img, bbox['Class'], (bbox['xmin'], bbox['ymin'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1,
                            lineType=cv2.LINE_AA)
            break


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--datadir', '-d', default='data')
    parser.add_argument('--imgdir', '-i', default='img')
    parser.add_argument('--filenames', '-f', default='*.png')
    parser.add_argument('--delay', '-t', default=500)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('datafile', nargs='+')

    args = parser.parse_args()

    if args.verbose == 0:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    with open(args.datafile[0]) as in_file:
        annot_green_data = json.load(in_file)

    if len(args.datafile) > 1:
        with open(args.datafile[1]) as in_file:
            annot_red_data = json.load(in_file)
    else:
        annot_red_data = None

    filenames = sorted(glob.glob(os.path.join(args.imgdir, args.filenames)))
    for filename in filenames:
        img = cv2.imread(filename)
        draw_img_annot_data(annot_green_data, filename, img, (0, 255, 0))
        logger.info(filename)

        if annot_red_data:
            draw_img_annot_data(annot_red_data, filename, img, (0, 0, 255))
        cv2.imshow('img', img)
        if cv2.waitKey(int(args.delay)
                       ) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
