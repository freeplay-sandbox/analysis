#include <fstream>
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

#include "json/json.h"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;

// color palette generated from http://paletton.com
const Scalar A (110, 37,110); // purple
const Scalar A1(166,111,166);
const Scalar A2(138, 69,138);
const Scalar A3( 83, 14, 83);
const Scalar A4( 55,  0, 55);
                           
const Scalar B ( 57, 80,170); // saumon
const Scalar B1(170,188,255);
const Scalar B2(106,128,212);
const Scalar B3( 21, 43,128);
const Scalar B4(  0, 18, 85);
                           
const Scalar C ( 76,121, 40); // teal
const Scalar C1(148,182,121);
const Scalar C2(109,151, 76);
const Scalar C3( 48, 91, 15);
const Scalar C4( 26, 61,  0);
                           
const Scalar D ( 55,164,145); // lime
const Scalar D1(164,246,232);
const Scalar D2(103,205,188);
const Scalar D3( 21,123,106);
const Scalar D4(  0, 82, 68);


//const string BAG_FILE ("rectified_streams.bag");
const string BAG_FILE ("freeplay.bag");
const string POSES_FILE ("poses.json");

const float SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD = 0.05;
const float SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.2;
const uint NB_SKEL_FEATURES = 18;
const vector<uint>  SKEL_SEGMENTS {   0,1, // neck
                                      1,2,   2,3, 3,4, // right arm
                                      1,5,   5,6, 6,7, // left arm
                                      1,8,   8,9, 9,10, // right leg
                                     1,11, 11,12, 12,13, // left leg
                                     0,14, 14,16, // right eye/ear
                                     0,15, 15,17 // left eye/ear
                                    };
const vector<Scalar> SKEL_COLORS {      C1,
                                        A3,    A,     A,
                                        A3,    A,     A,
                                        A3,    A,     A,
                                        A3,    A,     A,
                                         C,   C4,
                                         C,   C4
                                 };

const float FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.4;
const float FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD = 0.2;
const float PUPILS_CONFIDENCE_THRESHOLD = 0.8;
const uint NB_FACE_FEATURES = 70;
const vector<uint> FACE_SEGMENTS {    0,1,   1,2,   2,3,   3,4,   4,5,   5,6,   6,7,   7,8,   8,9,  9,10, 10,11, 11,12, 12,13, 13,14, 14,15, 15,16, // face contour
                                    17,18, 18,19, 19,20, 20,21, // right eyebrow
                                    22,23, 23,24, 24,25, 25,26, // left eyebrow
                                    27,28, 28,29, 29,30, // nose line
                                    31,32, 32,33, 33,34, 34,35, // nosetrils
                                    36,37, 37,38, 38,39, 39,40, 40,41, 41,36, // right eye
                                    42,43, 43,44, 44,45, 45,46, 46,47, 47,42, // left eye
                                    48,49, 49,50, 50,51, 51,52, 52,53, 53,54, 54,55, 55,56, 56,57, 57,58, 58,59, 59,48, // outer lips
                                    60,61, 61,62, 62,63, 63,64, 64,65, 65,66, 66,67, 67,60 // inner lips
                                   };
const vector<Scalar> FACE_COLORS {      D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,      D,    D,
                                       D4,    D4,    D4,    D4,
                                       D4,    D4,    D4,    D4,
                                        D,     D,     D,
                                        D,     D,     D,     D,
                                       D4,    D4,    D4,    D4,    D4,    D4,
                                       D4,    D4,    D4,    D4,    D4,    D4,
                                       D3,    D3,    D3,    D3,    D3,    D3,    D3,    D3,    D3,    D3,    D3,    D3,
                                       D3,    D3,    D3,    D3,    D3,    D3,    D3,    D3
                                };

const float HAND_FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.4;
const float HAND_FEATURE_LOW_CONFIDENCE_THRESHOLD = 0.2;
const uint NB_HAND_FEATURES = 21;
const vector<uint> HAND_SEGMENTS {   0,1,   1,2,   2,3,  3,4,  // thumb
                                     0,5,   5,6,   6,7,  7,8, // index
                                     0,9,  9,10, 10,11, 11,12, // middle finger
                                    0,13, 13,14, 14,15, 15,16, // ring finger
                                    0,17, 17,18, 18,19, 19,20 // little finger
                                   };
const vector<Scalar> HAND_COLORS {      B3,   B,    B2,    B1,
                                        B3,   B,    B2,    B1,
                                        B3,   B,    B2,    B1,
                                        B3,   B,    B2,    B1
                                };

bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
}

cv::Mat drawSkeleton(cv::Mat image, Json::Value skel) {

    auto w = image.size().width;
    auto h = image.size().height;

    for(uint skel_idx = 1; skel_idx <= skel.size(); skel_idx++) {
        for(uint i = 0; i < SKEL_SEGMENTS.size(); i+=2) {

            float confidence1 = skel[to_string(skel_idx)][SKEL_SEGMENTS[i]][2].asFloat();
            if (confidence1 < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p1(w*skel[to_string(skel_idx)][SKEL_SEGMENTS[i]][0].asFloat(), h*skel[to_string(skel_idx)][SKEL_SEGMENTS[i]][1].asFloat());

            float confidence2 = skel[to_string(skel_idx)][SKEL_SEGMENTS[i+1]][2].asFloat();
            if (confidence2 < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            bool highconfidence = (confidence1 > SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

            int width = highconfidence ? 3 : 1;

            Point2f p2(w*skel[to_string(skel_idx)][SKEL_SEGMENTS[i+1]][0].asFloat(), h*skel[to_string(skel_idx)][SKEL_SEGMENTS[i+1]][1].asFloat());

            cv::line(image, p1, p2, SKEL_COLORS[i/2], width, cv::LINE_AA);
        }

        for(uint i = 0; i < NB_SKEL_FEATURES; i++) {
            float confidence = skel[to_string(skel_idx)][i][2].asFloat();
            if (confidence < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p(w*skel[to_string(skel_idx)][i][0].asFloat(), h*skel[to_string(skel_idx)][i][1].asFloat());
            cv::circle(image, p, 5, A1, -1, cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawFace(cv::Mat image, Json::Value face) {

    auto w = image.size().width;
    auto h = image.size().height;

    for(uint face_idx = 1; face_idx <= face.size(); face_idx++) {
        for(uint i = 0; i < FACE_SEGMENTS.size(); i+=2) {

            float confidence1 = face[to_string(face_idx)][FACE_SEGMENTS[i]][2].asFloat();
            if (confidence1 < FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p1(w*face[to_string(face_idx)][FACE_SEGMENTS[i]][0].asFloat(), h*face[to_string(face_idx)][FACE_SEGMENTS[i]][1].asFloat());

            float confidence2 = face[to_string(face_idx)][FACE_SEGMENTS[i+1]][2].asFloat();
            if (confidence2 < FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            bool highconfidence = (confidence1 > FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

            int width = highconfidence ? 2 : 1;

            Point2f p2(w*face[to_string(face_idx)][FACE_SEGMENTS[i+1]][0].asFloat(), h*face[to_string(face_idx)][FACE_SEGMENTS[i+1]][1].asFloat());

            cv::line(image, p1, p2, FACE_COLORS[i/2], width, cv::LINE_AA);
        }

        // pupils
        for(uint i = 68; i < NB_FACE_FEATURES; i++) {
            float confidence = face[to_string(face_idx)][i][2].asFloat();
            if (confidence < PUPILS_CONFIDENCE_THRESHOLD) continue;

            Point2f p(w*face[to_string(face_idx)][i][0].asFloat(), h*face[to_string(face_idx)][i][1].asFloat());
            cv::circle(image, p, 3, B, 1, cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawHands(cv::Mat image, Json::Value hand) {

    auto w = image.size().width;
    auto h = image.size().height;

    for(uint hand_idx = 1; hand_idx <= hand.size(); hand_idx++) {
        for(auto handeness : {"left", "right"}) {
            for(uint i = 0; i < HAND_SEGMENTS.size(); i+=2) {

                float confidence1 = hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i]][2].asFloat();
                if (confidence1 < HAND_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

                Point2f p1(w*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i]][0].asFloat(), h*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i]][1].asFloat());

                float confidence2 = hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i+1]][2].asFloat();
                if (confidence2 < HAND_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

                bool highconfidence = (confidence1 > HAND_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > HAND_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

                int width = highconfidence ? 2 : 1;

                Point2f p2(w*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i+1]][0].asFloat(), h*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i+1]][1].asFloat());

                cv::line(image, p1, p2, HAND_COLORS[i/2], width, cv::LINE_AA);
            }
        }
    }

    return image;
}


cv::Mat drawPose(cv::Mat image, Json::Value frame, bool skeletons, bool faces, bool hands) {

    if(skeletons) image = drawSkeleton(image, frame["poses"]);
    if(hands) image = drawHands(image, frame["hands"]);
    if(faces) image = drawFace(image, frame["faces"]);
    return image;


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
        ("skeleton", po::value<bool>()->default_value(true), "display skeletons")
        ("face", po::value<bool>()->default_value(true), "display faces")
        ("hand", po::value<bool>()->default_value(true), "display hands")
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

    cout << "Opening " << vm["path"].as<string>() << "/" << BAG_FILE << "..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/" + BAG_FILE, rosbag::bagmode::Read);

    string topic(vm["topic"].as<string>());

    std::vector<std::string> topics;
    topics.push_back(topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    cout << view.size() << " messages to process" << endl << endl;

    Json::Value root;

    cout << "Opening " << POSES_FILE << "..." << flush;
    auto start = std::chrono::system_clock::now();
    std::ifstream file(vm["path"].as<string>() + "/" + POSES_FILE);
    file >> root;

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    cout << "done (took " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)" << endl;


    int idx = 0;
    int last_percent = 0;

    for(rosbag::MessageInstance const m : view)
    {
        idx++;

        auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();
        if (compressed_rgb != NULL) {
            auto cvimg = imdecode(compressed_rgb->data,1);


            cvimg = drawPose(cvimg, root[topic]["frames"][idx],vm["skeleton"].as<bool>(), vm["face"].as<bool>(), vm["hand"].as<bool>() );


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
