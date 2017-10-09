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
    with open('visual_tracking_full_dataset.json', 'r') as outfile:
            fulldataset = json.load(outfile)

    if args.multisets:
        from random import shuffle
        shuffle(fulldataset)
        train = fulldataset[:int(len(fulldataset) * 0.8)]
        print("Train dataset: %d datapoints" % len(train))
        test = fulldataset[int(len(fulldataset) * 0.8 + 1): ]
        print("Test dataset: %d datapoints" % len(test))

        with open('visual_tracking_full_dataset.train.json', 'w') as f:
            json.dump(train, f)
        with open('visual_tracking_full_dataset.test.json', 'w') as f:
            json.dump(test, f)

