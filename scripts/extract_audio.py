#!/usr/bin/env python
"""
Based on code taken from https://github.com/ElenaDiamantidou/rosbagAnnotator/blob/master/rosbagAudio.py
Copyright Eleni Diamantidou
"""


import roslib
import cv2
#audio packages
#import audio_common_msgs
#from cv_bridge import CvBridge, CvBridgeError
#import rospy
from std_msgs.msg import String
import signal
import os
import os.path
import sys
import time
import threading
import rosbag
import yaml
import numpy as np
import argparse
import struct
import wave
import subprocess
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

programmName = os.path.basename(sys.argv[0])

#input parameters
def parse_arguments():
    inputFile = sys.argv[-1]
    #print inputFile 
    return inputFile
    

def audio_bag_file(bag, topic):
    """ Convert ROS audio messages into a stand-alone MP3 file
    """

    info_dict = yaml.load(bag._get_yaml_info())

    topic_name = topic['topic']
    messages =  topic['messages']
    duration = info_dict['duration']
    frequency = topic['frequency']

    audio = []

    nb_msg = 0
    last_percent = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        audio += msg.data
        nb_msg += 1
        percent = int(nb_msg * 100 / messages)
        if percent != last_percent:
            last_percent = percent
            print("\x1b[F%d%% done" % percent)

    #nBytes = len(audio)
    #nSamples = nBytes / 2
    #print "total number of compressed bytes {0:d}".format(nBytes)
    #print "total number of compressed bytes {0:d}".format(nSamples)
    #print "total duration {0:.2f}".format(duration)
    #print "average bit rate {0:.2f}".format(float(nBytes) * 8.0 / float(duration))

    frequency = frequency * 1000
    #bag.close()    

    return audio, frequency

#save mp3 file
def write_mp3_file(audioData, mp3FileName):
    mp3_file = open(mp3FileName, 'w')
    mp3_file.write(''.join(audioData))
    mp3_file.close()
    return mp3FileName

#convert mp3 to wav
def mp3_to_wav(mp3Path, frequency):
    #call arg not file...
    wavFileName = mp3Path.replace(".mp3",".wav")
    #write 1.6 kHz
    subprocess.call(['ffmpeg', '-i', mp3Path, '-y', '-ar', '16000', '-ac', '1', wavFileName])
    return wavFileName

#play wav file
def play_wav(wavFileName):
    #-nodisp   : display not in new window
    #-autoexit : stop automatically
    #-ss       : start time
    #-t        : duration
    subprocess.call(['ffplay', '-nodisp', '-autoexit','-ss','0', '-t', '3.5', wavFileName])


#Plot waveform in GUI
#create waveform of wav file
def createWaveform(wavFileName):
    spf = wave.open(wavFileName,'r')
    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')

    #If Stereo
    if spf.getnchannels() == 2:
        print 'Just mono files'
        sys.exit(0)

    fig = plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(signal)
    play_wav(wavFileName)
    
    plt.grid()
    plt.show()


def process_bag(bag_path, force=False):

    dest = os.path.join(os.path.abspath(os.path.dirname(bag_path)), "audio")
    basename = os.path.basename(bag_path)[:-4] # remove .bag extension
    print("Extracted audio files will be saved to %s" % dest)
    if not os.path.exists(dest):
        os.makedirs(dest)

    print("Opening %s (this may take up to several min depending on the size of the bag file)..." % bag_path)

    bag = rosbag.Bag(bag_path, 'r')

    topics = yaml.load(bag._get_yaml_info())["topics"]

    for topic in topics:
        if "audio" in topic["topic"]:
            filename = "%s_%s.mp3" % (basename, topic["topic"].replace("/", "_"))
            if os.path.exists(os.path.join(dest, filename)) and not force:
                continue
            print("Extracting %s to %s\n" % (topic["topic"], filename))
            audioData, frequency = audio_bag_file(bag, topic) 

            mp3FileName = write_mp3_file(audioData, os.path.join(dest, filename))
            print("Converting %s to WAV..." % filename)
            mp3_to_wav(mp3FileName, frequency)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='PInSoRo Dataset -- ROS bags audio extractor')
    parser.add_argument("path", help="path to a directory or a bag file. Directories are recursively traversed for bag files")
    parser.add_argument("-f", "--force", action='store_true', help="if true, overwrite existing files. Otherwise, skip them.")

    args = parser.parse_args()

    if args.path.endswith(".bag"):
        process_bag(args.path, args.force)
    else: # directory

        for dirpath, dirs, files in os.walk(args.path, topdown=False):
            print("Processing %s" % dirpath)
            for name in files:
                if name.endswith(".bag"):
                    sourcepath = os.path.abspath(os.path.join(args.path, dirpath, name))
                    process_bag(sourcepath, args.force)

