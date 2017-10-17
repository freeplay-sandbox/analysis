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
