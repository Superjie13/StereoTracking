# This script converts kitti tracking labels into COCO style.
# Official website of the Kitti tracking dataset: https://www.cvlibs.net/datasets/kitti/eval_tracking.php
# Kitti tracking FORMAT
'''
 Values    Name      Description
----------------------------------------------------------------------------
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   1    conf         Confidence
   3    location     3D object location x,y,z in camera coordinates (in meters)
----------------------------------------------------------------------------

'''

import argparse
import os
from collections import defaultdict

import mmengine

CLASSES = [dict(id=1, name='drone')]
cat2id = {'drone': 1}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Airsim MOT label to COCO-VID format.')
    parser.add_argument('-i', '--input',
                        default='/home/sijie/Documents/datasets/MOT/AirSim_drone/',
                        help='path of MOT Kitti data')
    parser.add_argument(
        '-o', '--output',
        default='/home/sijie/Documents/datasets/MOT/AirSim_drone/annotations',
        help='path to save coco formatted label file')
    parser.add_argument(
        '--split-train',
        default=False,
        help='split the train set into half-train and half-validate.')
    parser.add_argument('--distance_thr', default=80.0, type=float,
                        help='distance threshold to filter out distant objects')
    parser.add_argument('--area_thr', default=30.0, type=float,
                        help='area threshold to filter out small objects')
    return parser.parse_args()


def parse_gts(gts, args=None):
    outputs = defaultdict(list)
    for gt in gts:
        params = gt.split(',')
        frame_id = int(params[0])
        track_id = int(params[1])
        cat_id = cat2id['drone']
        bbox = [float(params[2]), float(params[3]), float(params[4]), float(params[5])]
        conf = 1.  # float(params[6])
        visibility = 1.
        location = [float(params[7]), float(params[8]), float(params[9])]
        area = bbox[2] * bbox[3]

        # if depth is NaN, skip
        if location[2] != location[2]:  # ！= itself means NaN
            continue

        if args is not None:
            if area < args.area_thr or location[2] > args.distance_thr:
                # print(location)
                continue

        anns = dict(
            category_id=cat_id,
            bbox=bbox,
            area=area,
            depth=location[2],
            location=location,
            iscrowd = False,
            visibility=visibility,
            mot_instance_id = track_id,
            mot_conf = conf)
        outputs[frame_id].append(anns)
    return outputs


def main():
    args = parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    sets = ['train', 'val']  # ['train', 'test']
    if args.split_train:
        sets += ['half-train', 'half-val']
    vid_id, img_id, ann_id = 1, 1, 1

    for subset in sets:
        ins_id = 0
        print(f'Converting {subset} set to COCO format')
        if 'half' in subset:
            in_folder = os.path.join(args.input, 'train')
        else:
            in_folder = os.path.join(args.input, subset)

        out_file = os.path.join(args.output, f'{subset}_cocoformat_{str(int(args.distance_thr))}.json')
        outputs = defaultdict(list)
        outputs['categories'] = CLASSES

        video_names = os.listdir(in_folder)
        for video_name in video_names:
            # basic params
            parse_gt = 'test' not in subset  # test file do not have gt
            ins_maps = dict()  # instance id map

            # load video infos
            video_folder = os.path.join(in_folder, video_name)
            # video-level infos
            img_folder = 'left'
            img_names = [name for name in os.listdir(f'{video_folder}/{img_folder}') if
                         name.endswith('.png')]
            img_names = sorted(img_names)
            fps = 60
            num_imgs = len(img_names)
            width = 1280
            height = 720
            video = dict(
                id=vid_id,
                name=video_name,
                fps=fps,
                width=width,
                height=height)

            # parse annotations
            if parse_gt:
                gts = mmengine.list_from_file(f'{video_folder}/gt/gt.txt')
                img2gts = parse_gts(gts, args)
            # make half sets
            if 'half' in subset:
                split_frame = num_imgs // 2 + 1
                if 'train' in subset:
                    img_names = img_names[: split_frame]
                elif 'val' in subset:
                    img_names = img_names[split_frame:]
                else:
                    raise ValueError(
                        'subset must be named with `train` or `val`')
                mot_frame_ids = [str(int(fname.split('.')[0])) for fname in img_names]
                with open(f'{video_folder}/gt/gt_{subset}.txt', 'wt') as f:
                    for gt in gts:
                        if gt.split(' ')[0] in mot_frame_ids:
                            f.writelines(f'{gt}\n')

            # image and box level infos
            for frame_id, name in enumerate(img_names):
                img_name = os.path.join(video_name, img_folder, name)
                mot_frame_ids = int(name.split('.')[0])
                image = dict(
                    id=img_id,
                    video_id=vid_id,
                    file_name=img_name,
                    height=height,
                    width=width,
                    frame_id=frame_id,
                    mot_frame_ids=mot_frame_ids)

                if parse_gt:
                    gts = img2gts[mot_frame_ids]
                    if len(gts) < 1 and 'train' in subset:
                        continue
                    for gt in gts:
                        gt.update(id=ann_id, image_id=img_id)
                        mot_ins_id = gt['mot_instance_id']
                        if mot_ins_id in ins_maps:
                            gt['instance_id'] = ins_maps[mot_ins_id]
                        else:
                            gt['instance_id'] = ins_id
                            ins_maps[mot_ins_id] = ins_id
                            ins_id += 1
                        outputs['annotations'].append(gt)
                        ann_id += 1

                outputs['images'].append(image)
                img_id += 1

            outputs['videos'].append(video)
            vid_id += 1
            outputs['num_instances'] = ins_id

        print(f'{subset} has {ins_id} instances.')
        mmengine.dump(outputs, out_file)
        print(f'Done! Saved as {out_file}')


if __name__ == '__main__':
    main()
