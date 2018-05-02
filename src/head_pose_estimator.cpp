#include <cmath>
#include <ctime>

#include <opencv2/calib3d/calib3d.hpp>

#ifdef HEAD_POSE_ESTIMATION_DEBUG
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#endif

#include "head_pose_estimator.hpp"

using namespace std;
using namespace cv;


HeadPoseEstimation::HeadPoseEstimation(float focalLength, float opticalCenterX, float opticalCenterY) :
        focalLength(focalLength),
        opticalCenterX(opticalCenterX),
        opticalCenterY(opticalCenterY)
{
}


pair<head_pose, float> HeadPoseEstimation::pose(const std::array<feature,70> facial_features) const
{

    cv::Mat projectionMat = cv::Mat::zeros(3,3,CV_32F);
    cv::Matx33f projection = projectionMat;
    projection(0,0) = focalLength;
    projection(1,1) = focalLength;
    projection(0,2) = opticalCenterX;
    projection(1,2) = opticalCenterY;
    projection(2,2) = 1;

    std::vector<Point3f> head_points;

    head_points.push_back(P3D_SELLION);
    head_points.push_back(P3D_RIGHT_EYE);
    head_points.push_back(P3D_LEFT_EYE);
    head_points.push_back(P3D_RIGHT_EAR);
    head_points.push_back(P3D_LEFT_EAR);
    head_points.push_back(P3D_MENTON);
    head_points.push_back(P3D_NOSE);
    head_points.push_back(P3D_STOMMION);

    std::vector<Point2f> detected_points;

    detected_points.push_back(coordsOf(facial_features, SELLION));
    detected_points.push_back(coordsOf(facial_features, RIGHT_EYE));
    detected_points.push_back(coordsOf(facial_features, LEFT_EYE));
    detected_points.push_back(coordsOf(facial_features, RIGHT_SIDE));
    detected_points.push_back(coordsOf(facial_features, LEFT_SIDE));
    detected_points.push_back(coordsOf(facial_features, MENTON));
    detected_points.push_back(coordsOf(facial_features, NOSE));

    auto stomion = (coordsOf(facial_features, MOUTH_CENTER_TOP) + coordsOf(facial_features, MOUTH_CENTER_BOTTOM)) * 0.5;
    detected_points.push_back(stomion);
        
    std::vector<FACIAL_FEATURE> FACIAL_FEATURES = {
        SELLION,
        RIGHT_EYE,
        LEFT_EYE,
        RIGHT_SIDE,
        LEFT_SIDE,
        MOUTH_CENTER_TOP,
        MOUTH_CENTER_BOTTOM,
        MENTON,
        NOSE
    };

    float confidence = 1;
    for (auto feature : FACIAL_FEATURES) {
        confidence = min(confidence, confidenceOf(facial_features, feature));
    }

    // Initializing the head pose 1m away, roughly facing the robot
    // This initialization is important as it prevents solvePnP to find the
    // mirror solution (head *behind* the camera)
    Mat tvec = (Mat_<double>(3,1) << 0., 0., 1.);
    Mat rvec = (Mat_<double>(3,1) << 1.2, 1.2, -1.2);

    // Find the 3D pose of our head
    solvePnP(head_points, detected_points,
            projection, noArray(),
            rvec, tvec, true,
            cv::SOLVEPNP_ITERATIVE);

    Matx33d rotation;
    Rodrigues(rvec, rotation);

    head_pose pose = {
        rotation(0,0),    rotation(0,1),    rotation(0,2),    tvec.at<double>(0)/1000,
        rotation(1,0),    rotation(1,1),    rotation(1,2),    tvec.at<double>(1)/1000,
        rotation(2,0),    rotation(2,1),    rotation(2,2),    tvec.at<double>(2)/1000,
                    0,                0,                0,                     1
    };

#ifdef HEAD_POSE_ESTIMATION_DEBUG


    std::vector<Point2f> reprojected_points;

    projectPoints(head_points, rvec, tvec, projection, noArray(), reprojected_points);

    for (auto point : reprojected_points) {
        circle(_debug, point,2, Scalar(0,255,255),2);
    }

    std::vector<Point3f> axes;
    axes.push_back(Point3f(0,0,0));
    axes.push_back(Point3f(50,0,0));
    axes.push_back(Point3f(0,50,0));
    axes.push_back(Point3f(0,0,50));
    std::vector<Point2f> projected_axes;

    projectPoints(axes, rvec, tvec, projection, noArray(), projected_axes);

    line(_debug, projected_axes[0], projected_axes[3], Scalar(255,0,0),2,CV_AA);
    line(_debug, projected_axes[0], projected_axes[2], Scalar(0,255,0),2,CV_AA);
    line(_debug, projected_axes[0], projected_axes[1], Scalar(0,0,255),2,CV_AA);

    putText(_debug, "(" + to_string(int(pose(0,3) * 100)) + "cm, " + to_string(int(pose(1,3) * 100)) + "cm, " + to_string(int(pose(2,3) * 100)) + "cm)", coordsOf(face_idx, SELLION), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),2);


#endif

    return make_pair(pose, confidence);
}

std::vector<pair<head_pose,float>> HeadPoseEstimation::poses(const vector<array<feature,70>> faces) const {

    std::vector<pair<head_pose,float>> res;

    for (const auto& face : faces){
        res.push_back(pose(face));
    }

    return res;

}

Point2f HeadPoseEstimation::coordsOf(const std::array<feature,70> facial_features, FACIAL_FEATURE feature) const
{
    return Point2f(get<0>(facial_features[feature]),
                   get<1>(facial_features[feature]));
}

float HeadPoseEstimation::confidenceOf(const std::array<feature,70> facial_features, FACIAL_FEATURE feature) const
{
    return get<2>(facial_features[feature]);
}


// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// taken from: http://stackoverflow.com/a/7448287/828379
bool HeadPoseEstimation::intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
                                      Point2f &r) const
{
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

