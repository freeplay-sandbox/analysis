#ifndef _GAZE_FEATURES_HPP
#define _GAZE_FEATURES_HPP

#include <vector>
#include <deque>
#include <utility>
#include <memory>

#ifdef WITH_CAFFE
#include <opencv2/core.hpp>
#include <caffe/caffe.hpp>
#endif

#include "json.hpp"

typedef std::tuple<uint, uint, float> feature; // a point in pixels + confidence

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


const std::vector<size_t> FACIAL_POI    {36, 37, 38, 39, 40, 41, // right eye
                                    17, 18, 19, 20, 21, // right eyebrow
                                    42, 43, 44, 45, 46, 47, // left eye
                                    22, 23, 24, 25, 26, // right eyebrow
                                    68, 69 // pupils
                                    };

const std::vector<size_t> SKELETON_POI {0,1, 2, 5, 14,15,16,17}; // nose, neck, shoulders, left/right eyes & ears

/*
 * Returns a vector of [x0,y0, x1, y1, ...] for (1) the facial POI 
 * defined in FACIAL_POI, followed by (2) the skeleton POI defined 
 * in SKELETON_POI
 */
std::vector<float> getfeatures(const nlohmann::json& frame, bool mirror=false);

/*
 * Returns an array of all the 70 facial landmarks with their x, y 
 * and confidence value.
 * Coordinates are in image pixels
 */
std::array<feature, 70> getfaciallandmarks(const nlohmann::json& frame, bool mirror=false, size_t src_width=960, size_t source_height=480);

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

template<class T>
class valuefilter {

#define VALUE_FILTER_CAPACITY 10

public:
    valuefilter(size_t maxlen=VALUE_FILTER_CAPACITY) : maxlen(maxlen) {}

    void append(T val) {
        _dirty = true;
        _vals.push_back(val);
        if (_vals.size() > maxlen) _vals.pop_front();
    }

    T get() {

        if (_dirty) {
            T sum;
            for (auto& v : _vals) sum += v;
            _lastval = sum / (int) _vals.size();
            _dirty = false;
        }

        return _lastval;
    }

private:
        std::deque<T> _vals;
        size_t maxlen;
        T _lastval;
        bool _dirty;
};

#endif
