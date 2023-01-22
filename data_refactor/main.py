import argparse
import os
import json
import glob
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', default='')  # input json data directory
    parser.add_argument('--imgdir', '-i')  # input image directory
    parser.add_argument('--delay', '-t', default=1)
    parser.add_argument('--filenames', '-f', default='*.png')  # image extension
    parser.add_argument('--outdir', '-o', default="")  # optional output data directory
    parser.add_argument('--outimgdir', '-oi')  # optional image output data directory
    parser.add_argument('--datafile', '-df')  # input json filename
    parser.add_argument('--skipfrequency', '-sf', default=3)
    args = parser.parse_args()
    out_json = "refactored_" + args.datafile

    if args.imgdir is None:
        out_img_path = "refactored_" + args.imgdir
    else:
        out_img_path = "refactored_" + os.path.splitext(args.datafile)[0]
    with open(os.path.join(args.datadir, args.datafile)) as in_file:
        annot_data = json.load(in_file)

    if os.path.exists(out_img_path):
        print("Image path exists")
    else:
        os.mkdir(out_img_path)

    filenames = sorted(glob.glob(os.path.join(args.imgdir, args.filenames)))
    counter = 0
    skip = 0
    annotations = []
    for filename in filenames:
        counter += 1
        if counter >= int(args.skipfrequency):
            img = cv2.imread(filename)
            cv2.imshow('img', img)
            if cv2.waitKey(int(args.delay)
                           ) == 27:
                break
            img_filename = "frame%06i.png" % skip
            cv2.imwrite(os.path.join(out_img_path, img_filename), img)
            for item in annot_data:
                if item['filename'] == os.path.basename(filename):
                    annotations.append({
                        "filename": os.path.basename(img_filename),
                        "bb": item['bb']
                    })
            counter = 0
            skip += 1
    print(annotations)
    print(skip)
    with open(os.path.join(args.outdir, out_json), 'w') as out_file:
        json.dump(annotations, out_file, indent=2)

    print("Remember to check frequency of skipping!!!!")
    print(int(args.skipfrequency))


if __name__ == "__main__":
    main()
