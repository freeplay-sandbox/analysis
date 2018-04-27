#ifndef __HEAD_POSE_ESTIMATION
#define __HEAD_POSE_ESTIMATION

#include <opencv2/core/core.hpp>

#include <vector>
#include <tuple>
#include <array>
#include <string>

#include "gaze_features.hpp" // for the 'feature' typedef

// Anthropometric for male adult
// Relative position of various facial feature relative to sellion
// Values taken from https://en.wikipedia.org/wiki/Human_head
// X points forward
const static cv::Point3f P3D_SELLION(0., 0.,0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -65.5,-5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 65.5,-5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -77.5,-6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 77.5,-6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -48.0);
const static cv::Point3f P3D_STOMMION(10.0, 0., -75.0);
const static cv::Point3f P3D_MENTON(0., 0.,-133.0);


// Interesting facial features with their OpenPose/dlib landmark index
enum FACIAL_FEATURE {
    NOSE=30,
    RIGHT_EYE=36,
    LEFT_EYE=45,
    RIGHT_SIDE=0,
    LEFT_SIDE=16,
    EYEBROW_RIGHT=21,
    EYEBROW_LEFT=22,
    MOUTH_UP=51,
    MOUTH_DOWN=57,
    MOUTH_RIGHT=48,
    MOUTH_LEFT=54,
    SELLION=27,
    MOUTH_CENTER_TOP=62,
    MOUTH_CENTER_BOTTOM=66,
    MENTON=8
};


typedef cv::Matx44d head_pose;

class HeadPoseEstimation {

public:

    HeadPoseEstimation(float focalLength, float opticalCenterX, float opticalCenterY);

    head_pose pose(const std::array<feature,70> facial_features) const;
    std::vector<head_pose> poses(const std::vector<std::array<feature,70>> faces) const;

    float focalLength;
    float opticalCenterX;
    float opticalCenterY;

#ifdef HEAD_POSE_ESTIMATION_DEBUG
    mutable cv::Mat _debug;
#endif

private:

    /** Return the point corresponding to the dictionary marker.
    */
    cv::Point2f coordsOf(const std::array<feature,70> facial_features, FACIAL_FEATURE feature) const;

    /** Returns true if the lines intersect (and set r to the intersection
     *  coordinates), false otherwise.
     */
    bool intersection(cv::Point2f o1, cv::Point2f p1,
                      cv::Point2f o2, cv::Point2f p2,
                      cv::Point2f &r) const;

};

#endif // __HEAD_POSE_ESTIMATION
