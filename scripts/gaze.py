#! /usr/bin/env python

import sys
import numpy

import rospy
import tf

import tf.transformations as transformations
import geometry_msgs.msg


YELLOW_CAM_TO_CENTRE_QUATERNION=[0.530, -0.220, 0.314, 0.757]
YELLOW_CAM_TO_CENTRE_TRANSLATION=[0.141, 0.182, 0.397]

PURPLE_CAM_TO_CENTRE_QUATERNION=[0.220, -0.530, 0.757, 0.314]
PURPLE_CAM_TO_CENTRE_TRANSLATION=[-0.154, 0.178, 0.385]

def normalize(vec):
    norm=numpy.linalg.norm(vec)
    if norm==0: 
        return vec
    return vec/norm


def project_gaze_on_plane(gaze, plane):
    """ Calculates the 2D coordinate of the intersection between a ray cast
    along the X axis of the 'gaze' transform and the XY plane defined by 'plane'
    pose.

    Both 'gaze' and 'plane' are 4x4 transformation matrices, expressed in the
    same (arbitrary) reference frame.

    :returns: the [x,y,z] coordinates of the ray intersection, expressed in the
    'plane' reference frame (as such, the z coordinate should always be 0).

    """
    
    gaze_origin = numpy.array(gaze, copy=False)[:3, 3].copy()

    # casting a ray along the X axis
    gaze_vector = numpy.dot(gaze,[1,0,0,1])
    gaze_vector = (gaze_vector / gaze_vector[3])[:3] - gaze_origin
    gaze_vector = normalize(gaze_vector)

    plane_normal = normalize(numpy.dot(plane, [0,0,1,1])[:3])
    distance_plane_to_origin = 0

    t = - (numpy.dot(gaze_origin, plane_normal) + distance_plane_to_origin) / (numpy.dot(gaze_vector, plane_normal))

    gaze_projection = gaze_origin + gaze_vector * t

    return gaze_projection


def make_transform_matrix(quaternion, translation):
    M = numpy.identity(4)
    T = transformations.translation_matrix(YELLOW_CAM_TO_CENTRE_TRANSLATION)
    M = numpy.dot(M, T)
    R = transformations.quaternion_matrix(YELLOW_CAM_TO_CENTRE_QUATERNION)
    M = numpy.dot(M, R)

    M /= M[3, 3]

    return M
    
def transform(r11,r12,r13,r21,r22,r23,r31,r32,r33,tx,ty,tz, transform):


    M = numpy.identity(4)
    M[0, 0] = r11 
    M[0, 1] = r12 
    M[0, 2] = r13 
    M[1, 0] = r21 
    M[1, 1] = r22 
    M[1, 2] = r23 
    M[2, 0] = r31 
    M[2, 1] = r32 
    M[2, 2] = r33 
    M[0,3] = tx
    M[1,3] = ty
    M[2,3] = tz

    return numpy.dot(transform, M)

YELLOW_CAM_TO_CENTRE =  make_transform_matrix(YELLOW_CAM_TO_CENTRE_QUATERNION, YELLOW_CAM_TO_CENTRE_TRANSLATION)
PURPLE_CAM_TO_CENTRE =  make_transform_matrix(PURPLE_CAM_TO_CENTRE_QUATERNION, PURPLE_CAM_TO_CENTRE_TRANSLATION)

EPSILON = 1./30 # camera record at 30fps, so the time interval between 2 frames should be 1/30s

def find_next_matching_ts(l1, l2, idx1, idx2):

    while idx1 < len(l1) and idx2 < len(l2):


        ts1 = float(l1[idx1]["timestamp"])
        ts2 = float(l2[idx2]["timestamp"])

        if abs(ts1 - ts2) < EPSILON:
            return idx1 + 1, idx2 + 1

        if ts1 < ts2:
            idx1 += 1
        else:
            idx2 += 1

    return -1, -1




if __name__ == '__main__':
    rospy.init_node('test_broadcaster')

    br = tf.TransformBroadcaster()

    purpleposepub = rospy.Publisher("/purple_gaze", geometry_msgs.msg.PointStamped, queue_size=1)
    yellowposepub = rospy.Publisher("/yellow_gaze", geometry_msgs.msg.PointStamped, queue_size=1)

    import csv

    rate = rospy.Rate(60)

    purple = []
    yellow = []
    with open(sys.argv[1], 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['child'] == 'p':
                purple.append(row)
            else:
                yellow.append(row)
    print("%d frames for purple child, %d frames for yellow child" % (len(purple), len(yellow)))

    idx1 = 0
    idx2 = 0

    matching_ts = []
    while True:
        idx1, idx2 = find_next_matching_ts(purple, yellow, idx1, idx2)
        if idx1 < 0:
            break

        matching_ts.append((purple[idx1], yellow[idx2]))

    print("Found %d matching frames" % len(matching_ts))

    for pair in matching_ts:

        for row in pair:

            t = transform(row['r11'],row['r12'],row['r13'],
                            row['r21'],row['r22'],row['r23'],
                            row['r31'],row['r32'],row['r33'],
                            row['tx'],row['ty'],row['tz'],
                            PURPLE_CAM_TO_CENTRE if row["child"] == "p" else YELLOW_CAM_TO_CENTRE)

            br.sendTransform(transformations.translation_from_matrix(t),
                                transformations.quaternion_from_matrix(t),
                                rospy.Time.now(),
                                row["child"] + "_child",
                                "sandtray_centre")


            proj = project_gaze_on_plane(t, numpy.identity(4))
            pose = geometry_msgs.msg.PointStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "sandtray_centre"
            pose.point.x = proj[0]
            pose.point.y = proj[1]
            pose.point.z = proj[2]
            if row["child"] == "p":
                purpleposepub.publish(pose)
            else:
                yellowposepub.publish(pose)


        #rospy.spinOnce()
        rate.sleep()
