import sys
import numpy
import transformations


YELLOW_CAM_TO_CENTRE_QUATERNION=[0.530, -0.220, 0.314, 0.757]
YELLOW_CAM_TO_CENTRE_TRANSLATION=[0.141, 0.182, 0.397]

PURPLE_CAM_TO_CENTRE_QUATERNION=[0.220, -0.530, 0.757, 0.314]
PURPLE_CAM_TO_CENTRE_TRANSLATION=[-0.154, 0.178, 0.385]

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


import rospy

import tf

if __name__ == '__main__':
    rospy.init_node('test_broadcaster')

    br = tf.TransformBroadcaster()

    import csv

    rate = rospy.Rate(60)

    with open(sys.argv[1], 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            t = transform(row['r11'],row['r12'],row['r13'],
                          row['r21'],row['r22'],row['r23'],
                          row['r31'],row['r32'],row['r33'],
                          row['tx'],row['ty'],row['tz'],
                          PURPLE_CAM_TO_CENTRE)
            br.sendTransform(transformations.translation_from_matrix(t),
                             transformations.quaternion_from_matrix(t),
                             rospy.Time.now(),
                             "purple_child",
                             "sandtray_centre")

            #rospy.spinOnce()
            rate.sleep()
