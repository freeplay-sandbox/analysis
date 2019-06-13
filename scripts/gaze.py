#! /usr/bin/env python

import sys
import numpy

import rospy
import tf

import tf.transformations as transformations
import geometry_msgs.msg
from visualization_msgs.msg import Marker

# Obtained with the following steps:
# $ rosparam set /use_sim_time True
# $ rosbag play --clock freeplay.bag
# $ rosrun tf static_transform_publisher -0.3 0.169 0 0 0 0 sandtray_centre sandtray 20
# $ rosrun tf tf_echo sandtray_centre camera_{purple|yellow}_rgb_optical_frame
YELLOW_CAM_TO_CENTRE_QUATERNION=[-0.530, 0.220, -0.314, 0.757]
YELLOW_CAM_TO_CENTRE_TRANSLATION=[-0.408, -0.208, 0.035]

PURPLE_CAM_TO_CENTRE_QUATERNION=[0.220, -0.530, 0.757, -0.314]
PURPLE_CAM_TO_CENTRE_TRANSLATION=[-0.408, 0.190, 0.035]

SANDTRAY_WIDTH=0.338
SANDTRAY_LENGHT=0.6
OUT_OF_BOUND_MARGIN = 1.20 # 20% of the table size

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
    T = transformations.translation_matrix(translation)
    M = numpy.dot(M, T)
    R = transformations.quaternion_matrix(quaternion)
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
            return ts1, idx1 + 1, idx2 + 1

        if ts1 < ts2:
            idx1 += 1
        else:
            idx2 += 1

    return 0, -1, -1


def make_marker(proj, child, outofbound=False):


    color = [1., 1., 1.]
    if child=="p":
        color = [1.,0,1]
    if child=="y":
        color = [1.,1,0]
    if outofbound:
        color = [1.,0,0]

    marker = Marker()
    marker.ns = child + "_child"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "sandtray_centre"
    marker.pose.position.x = proj[0]
    marker.pose.position.y = proj[1]
    marker.pose.position.z = proj[2]
    marker.scale.x = .05
    marker.scale.y = .05
    marker.scale.z = .05
    marker.color.a = 1.0
    r,g,b = color
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    return marker


if __name__ == '__main__':
    rospy.init_node('test_broadcaster')

    br = tf.TransformBroadcaster()

    gazeposepub = rospy.Publisher("visualization_marker", Marker, queue_size=1)

    import csv

    rate = rospy.Rate(30)

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
        ts, idx1, idx2 = find_next_matching_ts(purple, yellow, idx1, idx2)
        if idx1 < 0:
            break

        matching_ts.append((ts, purple[idx1], yellow[idx2]))

    print("Found %d matching frames" % len(matching_ts))

    idx=0
    for ts, purple, yellow in matching_ts:

        for row in [purple, yellow]:

            t = transform(float(row['r11']),float(row['r12']),float(row['r13']),
                          float(row['r21']),float(row['r22']),float(row['r23']),
                          float(row['r31']),float(row['r32']),float(row['r33']),
                          float(row['tx'] ),float(row['ty'] ),float(row['tz']),
                          PURPLE_CAM_TO_CENTRE if row["child"] == 'p' else YELLOW_CAM_TO_CENTRE)

            br.sendTransform(transformations.translation_from_matrix(t),
                             transformations.quaternion_from_matrix(t),
                             #rospy.Time.now(),
                             rospy.Time(ts),
                             row["child"] + "_child",
                             "sandtray_centre")


            proj = project_gaze_on_plane(t, numpy.identity(4))


            outofbound = False
            px,py,pz=proj
            if abs(px) > SANDTRAY_LENGHT * OUT_OF_BOUND_MARGIN or abs(py) > SANDTRAY_WIDTH * OUT_OF_BOUND_MARGIN:
                outofbound = True

            marker = make_marker(proj, row["child"], outofbound)
            gazeposepub.publish(marker)


        #idx += 1
        #print("Done %d%%" % (idx*100./len(matching_ts)))
        rate.sleep()

