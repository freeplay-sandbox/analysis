import numpy as np
import cv2

import sys

import yaml

FPS=30

SOURCE="env_camera_skel"

bag = yaml.load(open(sys.argv[1] + "/freeplay.bag.yaml", 'r'))
annotations = yaml.load(open(sys.argv[1] + "/freeplay.annotations.yaml", 'r'))

cap = cv2.VideoCapture(sys.argv[1] + "/"+ SOURCE + ".mkv")
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)


fourcc = cv2.VideoWriter_fourcc(*'H264')
videowriter = cv2.VideoWriter(sys.argv[1] + "/" + SOURCE + "_annotations.mkv", fourcc, 30.0, (960, 540) );

timestamp = bag["start"]

TASK={"goaloriented": "goal oriented", "aimless": "aimless", "adultseeking": "adult seeking", "noplay": "no play"}
SOCIAL={"solitary":"solitary", "onlooker":"onlooker", "parallel":"parallel play", "associative":"associative play", "cooperative":"cooperative play"}
ATTITUDE={"prosocial":"pro-social", "adversarial":"adversarial", "assertive":"assertive", "frustrated":"frustrated", "passive":"passive"}

def getAnnotations(annots, t):
    res = []
    for a in annots:
        for k,v in a.items():
            if v[0] < t and v[1] > t:
                if k in TASK: res.append("Task engag.: " + TASK[k])
                elif k in SOCIAL: res.append("Social engag.: " + SOCIAL[k])
                elif k in ATTITUDE: res.append("Attitude: " + ATTITUDE[k])
                else:
                    print(k)
                    sys.exit()

    return sorted(res)


frame_idx = 0
last_percent = 0
while(frame_idx < length):

    frame_idx += 1
    percentage = int(frame_idx * 100 / length)
    if percentage != last_percent:
        print(str(percentage) + "% done")
        last_percent = percentage

    timestamp += 1./FPS

    # Capture frame-by-frame
    ret, frame = cap.read()


    for i, a in enumerate(getAnnotations(annotations["purple"], timestamp)):
        cv2.putText(frame, a, (630, 50 * i + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (155,0,78), 2)

    for i, a in enumerate(getAnnotations(annotations["yellow"], timestamp)):
        cv2.putText(frame, a, (20, 50 * i + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,120), 2)

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1000/FPS) & 0xFF == ord('q'):
    #    break
    videowriter.write(frame)

# When everything done, release the capture
cap.release()
videowriter.release()
cv2.destroyAllWindows()
