
#include "gaze_features.hpp"

using namespace std;
#ifdef WITH_CAFFE
using namespace cv;
#endif
using namespace nlohmann; // json

vector<float> getfeatures(const json& frame, bool mirror) {

    vector<float> f;

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

array<feature, 70> getfaciallandmarks(const json& frame, bool mirror, size_t src_width, size_t src_height) {

    array<feature, 70> features;

    json face;


    if(frame["faces"].size() > 0) // one face or more detected. Assume the correct one is the one closest to the image centre
    {
        face = frame["faces"]["1"];
        auto min_dist = pow(face[NOSE][0].get<double>()-0.5,2) +
                        pow(face[NOSE][0].get<double>() - 0.5, 2);

        for (auto& f : frame["faces"]) {
            auto dist = pow(f[NOSE][0].get<double>()-0.5,2) + 
                        pow(f[NOSE][0].get<double>() - 0.5, 2);

            if(dist < min_dist) {
                face = f;
                min_dist = dist;
            }
        }
    }
    else {
        return features;
    }



    for (size_t i = 0; i < 70; i++) {
        auto x = max(0., min(1., face[i][0].get<double>()));
        if(mirror) x = (1 - x);
        auto y = max(0., min(1., face[i][1].get<double>())); // Y coordinates of facial features
        auto c = max(0., min(1., face[i][2].get<double>())); // confidence
        features[i] = {src_width * x, src_height * y, c};
    }

    return features;

}

#ifdef WITH_CAFFE

GazeEstimator::GazeEstimator() {}

void GazeEstimator::initialize()
{
    net = make_shared<caffe::Net<float>>("share/models/gaze_estimate_ANN.prototxt", caffe::TEST);
    net->CopyTrainedLayersFrom("share/models/gaze.caffemodel");

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly 1 inputs.";
    CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly 1 outputs.";

    auto input_layer = net->input_blobs()[0];

    auto num_channels_ = input_layer->channels();
    CHECK_EQ(num_channels_, 64) << "Input layer should have 64 channels.";


}

Point2f GazeEstimator::estimate(const nlohmann::json& frame, bool mirror) {

    auto features = getfeatures(frame, mirror);

    auto input_layer = net->input_blobs()[0];

    float* input_data = input_layer->mutable_cpu_data();

    for(size_t i = 0; i < features.size(); i++) {
        input_data[i] = features[i];
    }


    net->Forward();

    /* Copy the output layer to a std::vector */
    auto output_layer = net->output_blobs()[0];
    auto output_data = output_layer->cpu_data();
    if(!mirror)
        return Point2f(output_data[0], output_data[1]);
    else
        return Point2f(output_data[0], 1.-output_data[1]);
}
#endif
