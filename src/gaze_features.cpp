#include "gaze_features.hpp"

using namespace std;
using namespace nlohmann; // json

vector<double> getfeatures(const json& frame, bool mirror) {

    vector<double> f;

    for (size_t i : FACIAL_POI) {
        auto x = max(0., min(1., frame["faces"]["1"][i][0].get<double>()));
        if(mirror) x = (1 - x);
        f.push_back(x);
        auto y = max(0., min(1., frame["faces"]["1"][i][1].get<double>())); // Y coordinates of facial features
        f.push_back(y);
    }

    for (size_t i : SKELETON_POI) {
        auto x = max(0., min(1., frame["poses"]["1"][i][0].get<double>()));
        if(mirror) x = (1 - x);
        f.push_back(x);
        auto y = max(0., min(1., frame["poses"]["1"][i][1].get<double>())); // Y coordinates of facial features
        f.push_back(y);
    }

    return f;

}


