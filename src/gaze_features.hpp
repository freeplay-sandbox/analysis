#ifndef _GAZE_FEATURES_HPP
#define _GAZE_FEATURES_HPP

#include <vector>
#include <utility>
#include <memory>

#ifdef WITH_CAFFE
#include <opencv2/core.hpp>
#include <caffe/caffe.hpp>
#endif

#include "json.hpp"

const std::vector<size_t> FACIAL_POI    {36, 37, 38, 39, 40, 41, // right eye
                                    17, 18, 19, 20, 21, // right eyebrow
                                    42, 43, 44, 45, 46, 47, // left eye
                                    22, 23, 24, 25, 26, // right eyebrow
                                    68, 69 // pupils
                                    };

const std::vector<size_t> SKELETON_POI {0,1, 2, 5, 14,15,16,17}; // nose, neck, shoulders, left/right eyes & ears

std::vector<float> getfeatures(const nlohmann::json& frame, bool mirror);

#ifdef WITH_CAFFE
class GazeEstimator {

public:
    GazeEstimator();
    void initialize();
    cv::Point2f estimate(const nlohmann::json& frame, bool mirror);

private:

    std::shared_ptr<caffe::Net<float>> net;
};
#endif

#endif
