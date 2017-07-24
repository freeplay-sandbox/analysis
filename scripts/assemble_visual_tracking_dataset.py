#! /usr/bin/env python

import argparse
import sys
import os
import json

import yaml
import rosbag

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Freeplay sandbox Dataset Statistics')
    parser.add_argument("path", help="root path of the dataset -- recordings are recursively looked for from this path")
    parser.add_argument("--bags", action='store_true', help="check the duration of the bags and resulting frequency")
    parser.add_argument("--multisets", action='store_true', help="prepare several sets (70% training, 10% validations, 20% test)")

    args = parser.parse_args()
    
    fulldataset = []

    totalduration = 0
    idx = 0
    for dirpath, dirs, files in os.walk(args.path, topdown=False):
        for name in files:
            fullpath = os.path.join(dirpath, name)
            if name == "visual_tracking_dataset.json":
                idx += 1
                print("Processing %d - %s" % (idx, fullpath))
                with open(fullpath, 'r') as f:
                    fulldataset += json.load(f)
            if args.bags and name == "visual_tracking.bag":
                baginfo = yaml.load(rosbag.Bag(fullpath, 'r')._get_yaml_info())
                duration = baginfo["end"] - baginfo["start"]
                print("Bag duration: %ssec" % duration)
                totalduration += duration




    print("In total, " + str(len(fulldataset)) + " data points")
    if args.bags:
        print("Total duration: %ss -> %sHz" % (totalduration, len(fulldataset) / totalduration))
    with open('visual_tracking_full_dataset.json', 'w') as outfile:
            json.dump(fulldataset, outfile)

    if args.multisets:
        from random import shuffle
        shuffle(fulldataset)
        train = fulldataset[:int(len(fulldataset) * 0.7)]
        print("Train dataset: %d datapoints" % len(train))
        validation = fulldataset[int(len(fulldataset) * 0.7 + 1): int(len(fulldataset) * 0.8)]
        print("Validation dataset: %d datapoints" % len(validation))
        test = fulldataset[int(len(fulldataset) * 0.8 + 1): ]
        print("Test dataset: %d datapoints" % len(test))

        with open('visual_tracking_full_dataset.train.json', 'w') as f:
            json.dump(train, f)
        with open('visual_tracking_full_dataset.validation.json', 'w') as f:
            json.dump(validation, f)
        with open('visual_tracking_full_dataset.test.json', 'w') as f:
            json.dump(test, f)

