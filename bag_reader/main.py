import cv2
import sys
import json
import argparse
import os
from os.path import exists as file_exists
import rosbag
from cv_bridge import CvBridge

parser = argparse.ArgumentParser()
parser.add_argument('--bagname', '-bag', required=True, help='podaj bagname:')
parser.add_argument('--bagdir', '-bgdir', default='bags')
parser.add_argument('--imgdir', '-img', required=False)
parser.add_argument('--save_imgs', '-svimg', action='store_true')  # if -svimg present in parameters then save imgs
parser.add_argument('--save_json', '-svjson', action='store_true')  # if -svjson present in parameters then save json
args = parser.parse_args()

# check if specified bag file exists
if not file_exists(args.bagname):
    pass
else:
    raise Exception("Bag file not found.")
    sys.exit(0)

# if imgdir was passed create custom path
if args.imgdir is None:
    img_path = os.path.join(os.path.dirname(__file__), os.path.splitext(args.bagname)[0])
else:
    img_path = os.path.join(os.path.dirname(__file__), args.imgdir)

print "Saved images path: ", img_path

# check if img path already exists
if os.path.exists(img_path):
    print "Image path exists"
else:
    os.mkdir(img_path)

# read the bag
bag = rosbag.Bag(os.path.join(args.bagdir, args.bagname), "r")
bag1 = rosbag.Bag(os.path.join(args.bagdir, args.bagname), "r")

# get information about topics in bags
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for i in range(0, len(bag.get_type_and_topic_info()[1].values())):
    types.append(bag.get_type_and_topic_info()[1].values()[i][0])
print "Topics: ", types

# initialize CV bridge, counters and times buffer
bridge = CvBridge()
data = []
frames_w_boxes = 0
frames_wo_boxes = 0
count = 0
times = []
for topic, msg, t in bag.read_messages():
    if 'image' in topic:
        t_img_header = msg.header.seq  # looking for detection with this frame's timestamp

        for topic1, msg1, t1 in bag1.read_messages():
            if 'bounding' in topic1:
                t_bb = msg1.image_header.seq
                if t_bb == t_img_header:  # if detection timestamp matches frame timestamp save bboxes
                    print "Found matching detection and image for frame: ", count
                    times.append(t_bb)
                    boxes = []
                    img_filename = "frame%06i.png" % count
                    for box in msg1.bounding_boxes:
                        boxes.append({
                            "id": box.id,
                            "Class": box.Class,
                            "probability": box.probability,
                            "xmin": box.xmin,
                            "xmax": box.xmax,
                            "ymin": box.ymin,
                            "ymax": box.ymax
                        })
                    data.append({"filename": img_filename,
                                 "bb": boxes})
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    if args.save_imgs:
                        cv2.imwrite(os.path.join(img_path, img_filename), image_rgb)
                    count += 1
                    frames_w_boxes += 1
                    break
    # if there was no detection found matching the frame timestamp then save the image and empty bbox
    if "image" in topic and t_img_header not in times:
        t_img_header = msg.header.seq
        print "There was no detection for frame: ", count
        times.append(t_img_header)
        boxes = []
        img_filename = "frame%06i.png" % count
        data.append({"filename": img_filename,
                     "bb": boxes})
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        if args.save_imgs:
            cv2.imwrite(os.path.join(img_path, img_filename), image_rgb)
        count += 1
        frames_wo_boxes += 1

if args.save_json:
    f = open(os.path.splitext(args.bagname)[0] + ".json", "w")
    json.dump(data, f, indent=2)
    f.close()

# check if every frame was extracted

print "Frames with boxes: ", frames_w_boxes
print "Frames witout boxes: ", frames_wo_boxes
print "Sum of frames: ", (frames_w_boxes + frames_wo_boxes)
print "Counter: ", count

# double check if every frame was extracted

frames = 0
for topic, msg, t in bag.read_messages():
    if 'image' in topic:
        frames += 1

print "True number of frames:", frames

# calculate FPS in bag and bag duration
print "Bag duration: ", (bag.get_end_time() - bag.get_start_time()), "s"
print "Average FPS: ", frames / (bag.get_end_time() - bag.get_start_time())
