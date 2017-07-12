#include <iostream>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <thread> // std::this_thread
#include <vector>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/program_options.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <yaml-cpp/yaml.h>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;


const string POSES_FILE ("poses.yaml");

const float SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD = 0.05;
const float SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.2;
const size_t NB_SKEL_FEATURES = 18;
const vector<size_t> SKEL_SEGMENTS {  0,1, // neck
                                      1,2,    2,3,   3,4, // right arm
                                      1,5,    5,6,   6,7, // left arm
                                      1,8,   8,9,  9,10, // right leg
                                      1,11,  11,12,  12,13, // left leg
                                      0,14,  14,16, // right eye/ear
                                      0,15,  15,17 // left eye/ear
                                    };

const float FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.4;
const float FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD = 0.2;
const float PUPILS_CONFIDENCE_THRESHOLD = 0.8;
const size_t NB_FACE_FEATURES = 70;
const vector<size_t> FACE_SEGMENTS {  0,1,  1,2,    2,3,   3,4,   4,5,   5,6,   6,7,   7,8,   8,9,  9,10,  10,11,  11,12,  12,13,  13,14,  14,15,  15,16, // face contour
                                    17,18,  18,19,  19,20, 20,21, // right eyebrow
                                    22,23,  23,24, 24,25, 25,26, // left eyebrow
                                    27,28, 28,29, 29,30, // nose line
                                    31,32,  32,33,  33,34,  34,35, // nosetrils
                                    36,37,  37,38,  38,39,  39,40,  40,41, 41,36, // right eye
                                    42,43,  43,44, 44,45, 45,46, 46,47, 47,42, // left eye
                                    48,49, 49,50,  50,51,  51,52,  52,53,  53,54,  54,55,  55,56,  56,57,  57,58, 58,59,  59,48, // outer lips
                                    60,61, 61,62, 62,63, 63,64, 64,65, 65,66, 66,67, 67,60 // inner lips
                                   };

bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
}

cv::Mat drawSkeleton(cv::Mat image, YAML::Node skel) {

    for(size_t skel_idx = 1; skel_idx <= skel.size(); skel_idx++) {
        for(size_t i = 0; i < SKEL_SEGMENTS.size(); i+=2) {

            float confidence1 = skel[skel_idx][SKEL_SEGMENTS[i]+1][2].as<float>();
            if (confidence1 < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p1(skel[skel_idx][SKEL_SEGMENTS[i]+1][0].as<float>(), skel[skel_idx][SKEL_SEGMENTS[i]+1][1].as<float>());

            float confidence2 = skel[skel_idx][SKEL_SEGMENTS[i+1]+1][2].as<float>();
            if (confidence2 < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            bool highconfidence = (confidence1 > SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

            int width = highconfidence ? 3 : 1;

            Point2f p2(skel[skel_idx][SKEL_SEGMENTS[i+1]+1][0].as<float>(), skel[skel_idx][SKEL_SEGMENTS[i+1]+1][1].as<float>());

            cv::line(image, p1, p2, Scalar(200,100,20), width, cv::LINE_AA);
        }

        for(size_t i = 0; i < NB_SKEL_FEATURES; i++) {
            float confidence = skel[skel_idx][i+1][2].as<float>();
            if (confidence < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p(skel[skel_idx][i+1][0].as<float>(), skel[skel_idx][i+1][1].as<float>());
            cv::circle(image, p, 5, Scalar(200,100,20), -1, cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawFace(cv::Mat image, YAML::Node face) {

    for(size_t face_idx = 1; face_idx <= face.size(); face_idx++) {
        for(size_t i = 0; i < FACE_SEGMENTS.size(); i+=2) {

            float confidence1 = face[face_idx][FACE_SEGMENTS[i]+1][2].as<float>();
            if (confidence1 < FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p1(face[face_idx][FACE_SEGMENTS[i]+1][0].as<float>(), face[face_idx][FACE_SEGMENTS[i]+1][1].as<float>());

            float confidence2 = face[face_idx][FACE_SEGMENTS[i+1]+1][2].as<float>();
            if (confidence2 < FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            bool highconfidence = (confidence1 > FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

            int width = highconfidence ? 2 : 1;

            Point2f p2(face[face_idx][FACE_SEGMENTS[i+1]+1][0].as<float>(), face[face_idx][FACE_SEGMENTS[i+1]+1][1].as<float>());

            cv::line(image, p1, p2, Scalar(20,100,200), width, cv::LINE_AA);
        }

        //for(size_t i = 0; i < NB_FACE_FEATURES - 2; i++) {
        //    float confidence = face[face_idx][i+1][2].as<float>();
        //    if (confidence < FACE_FEATURE_CONFIDENCE_THRESHOLD) continue;

        //    Point2f p(face[face_idx][i+1][0].as<float>(), face[face_idx][i+1][1].as<float>());
        //    cv::circle(image, p, 2, Scalar(50,100,200), -1, cv::LINE_AA);
        //}

        // pupils
        for(size_t i = 68; i < NB_FACE_FEATURES; i++) {
            float confidence = face[face_idx][i+1][2].as<float>();
            if (confidence < PUPILS_CONFIDENCE_THRESHOLD) continue;

            Point2f p(face[face_idx][i+1][0].as<float>(), face[face_idx][i+1][1].as<float>());
            cv::circle(image, p, 3, Scalar(50,200,100), 1, cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawPose(cv::Mat image, YAML::Node frame) {

    auto img1 = drawSkeleton(image, frame["poses"]);
    return drawFace(img1, frame["faces"]);

}

int main(int argc, char **argv) {


    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);

    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("version,v", "shows version and exits")
        ("topic", po::value<string>(), "topic to process (must be of type CompressedImage)")
        ("path", po::value<string>(), "record path (must contain experiment.yaml and freeplay.bag)")
        ("face", po::value<bool>()->default_value(true), "detect faces as well")
        ("hand", po::value<bool>()->default_value(true), "detect hands as well")
        ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
            .options(desc)
            .positional(p)
            .run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << argv[0] << " " << STR(FREEPLAY_ANALYSIS_VERSION) << "\n\n" << desc << "\n";
        return 1;
    }

    if (vm.count("version")) {
        cout << argv[0] << " " << STR(FREEPLAY_ANALYSIS_VERSION) << "\n";
        return 0;
    }

    if (vm.count("topic") == 0) {
        cout << "You must specify a topic to process" << endl;
        return 1;
    }

    if (!vm.count("path")) {
        cerr << "You must provide a record path.\n";
        return 1;
    }

    cout << "Opening " << vm["path"].as<string>() << "/freeplay.bag..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/freeplay.bag", rosbag::bagmode::Read);

    string topic(vm["topic"].as<string>());

    std::vector<std::string> topics;
    topics.push_back(topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    cout << view.size() << " messages to process" << endl << endl;

    cout << "Opening " << POSES_FILE << "..." << flush;

    YAML::Node posesYaml = YAML::LoadFile(vm["path"].as<string>() + "/" + POSES_FILE);

    cout << "done" << endl;


    int idx = 0;
    int last_percent = 0;

    for(rosbag::MessageInstance const m : view)
    {
        idx++;

        auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();
        if (compressed_rgb != NULL) {
            auto cvimg = imdecode(compressed_rgb->data,1);


            cvimg = drawPose(cvimg, posesYaml[topic]["frames"][idx]);


            imshow(topic, cvimg);
            waitKey(30);
        }

        int percent = idx * 100 / view.size();
        if (percent != last_percent) {
            cout << "\x1b[FDone " << percent << "% (" << idx << " images)" << endl;
            last_percent = percent;
        }

        if(interrupted) {
            cout << "Interrupted." << endl;
            break;
        }
    }

    bag.close();
}
