#include <fstream>
#include <iostream>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <thread> // std::this_thread
#include <vector>
#include <algorithm>

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

#include "json.hpp"
#include "gaze_features.hpp"
#include "histogram.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
using namespace nlohmann; // json
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

const Scalar E ( 57, 68,174); // light red
const Scalar E1(170,178,255);
const Scalar E2(106,116,212);
const Scalar E3( 21, 31,128);
const Scalar E4(  0,  8, 85);

const Scalar F ( 57,108,174); // light brown
const Scalar F1(170,207,255);
const Scalar F2(106,152,212);
const Scalar F3( 21, 67,128);
const Scalar F4(  0, 37, 85);

const Scalar G (102,107, 35); // blue
const Scalar G1(153,157,105);
const Scalar G2(126,131, 65);
const Scalar G3( 74, 78, 13);
const Scalar G4( 48, 52,  0);

const Scalar H ( 62,131, 43); // green
const Scalar H1(143,193,129);
const Scalar H2( 98,161, 80);
const Scalar H3( 34, 96, 16);
const Scalar H4( 14, 64,  0);

const Scalar WHITE(255,255,255);



//const string BAG_FILE ("freeplay.bag");
//const string POSES_FILE ("freeplay.poses.json");
const string BAG_FILE ("visual_tracking.bag");
const string POSES_FILE ("visual_tracking.poses.json");
const string VISUAL_TARGET_POSES_FILE ("visual_target.poses.json");

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
    H3,   H3,    H3,
    F3,   F3,    F3,
    D2,   D2,    D2,
    E,     E,     E,
    C1,   C1,
    C1,   C1
};

const float FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD = 0.4;
const float FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD = 0.2;
const float PUPILS_CONFIDENCE_THRESHOLD = 0.8;
const uint NB_FACE_FEATURES = 70;
const vector<uint> FACE_SEGMENTS {
    0,1,   1,2,   2,3,   3,4,   4,5,   5,6,   6,7,   7,8,   8,9,  9,10, 10,11, 11,12, 12,13, 13,14, 14,15, 15,16, // face contour
    17,18, 18,19, 19,20, 20,21, // right eyebrow
    22,23, 23,24, 24,25, 25,26, // left eyebrow
    27,28, 28,29, 29,30, // nose line
    31,32, 32,33, 33,34, 34,35, // nosetrils
    36,37, 37,38, 38,39, 39,40, 40,41, 41,36, // right eye
    42,43, 43,44, 44,45, 45,46, 46,47, 47,42, // left eye
    48,49, 49,50, 50,51, 51,52, 52,53, 53,54, 54,55, 55,56, 56,57, 57,58, 58,59, 59,48, // outer lips
    60,61, 61,62, 62,63, 63,64, 64,65, 65,66, 66,67, 67,60 // inner lips
};
const vector<Scalar> FACE_COLORS {
    D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,     D,      D,    D,
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
const vector<Scalar> HAND_COLORS {      G1,   G1,    G1,    G1,
    G2,   G2,    G2,    G2,
    G,   G,    G,    G,
    G3,   G3,    G3,    G3,
    G4,   G4,    G4,    G4
};

const float SANDTRAY_LENGTH=600.; //mm
const float SANDTRAY_WIDTH=340.; //mm

#ifdef WITH_CAFFE
GazeEstimator gazeEstimator;
valuefilter<Point2f> purpleGaze;
valuefilter<Point2f> yellowGaze;

Mat mapBg = imread("share/map.png");
#endif
Histogram<float> gazeAccuracyDistribution(10); // histogram bins = 10mm

bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
}

cv::Mat drawSkeleton(cv::Mat image, const json& skel, bool bg=false) {

    auto w = image.size().width;
    auto h = image.size().height;


    for(uint skel_idx = 1; skel_idx <= skel.size(); skel_idx++) {
        for(uint i = 0; i < SKEL_SEGMENTS.size(); i+=2) {

            float confidence1 = skel[to_string(skel_idx)][SKEL_SEGMENTS[i]][2].get<float>();
            if (confidence1 < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p1(w*skel[to_string(skel_idx)][SKEL_SEGMENTS[i]][0].get<float>(), h*skel[to_string(skel_idx)][SKEL_SEGMENTS[i]][1].get<float>());

            float confidence2 = skel[to_string(skel_idx)][SKEL_SEGMENTS[i+1]][2].get<float>();
            if (confidence2 < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            bool highconfidence = (confidence1 > SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > SKEL_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

            int width = highconfidence ? (bg ? 4 : 2) : (bg? 2 : 1);

            Point2f p2(w*skel[to_string(skel_idx)][SKEL_SEGMENTS[i+1]][0].get<float>(), h*skel[to_string(skel_idx)][SKEL_SEGMENTS[i+1]][1].get<float>());

            cv::line(image, p1, p2, bg ? WHITE : SKEL_COLORS[i/2], width, cv::LINE_AA);
        }

        for(uint i = 1; i < NB_SKEL_FEATURES; i++) { // [1-14] -> all features except on the face
            float confidence = skel[to_string(skel_idx)][i][2].get<float>();
            if (confidence < SKEL_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p(w*skel[to_string(skel_idx)][i][0].get<float>(), h*skel[to_string(skel_idx)][i][1].get<float>());
            cv::circle(image, p, bg ? 5 : 3, bg ? WHITE : SKEL_COLORS[i-1], -1, cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawFace(cv::Mat image, const json& face, bool bg=false) {

    auto w = image.size().width;
    auto h = image.size().height;

    for(uint face_idx = 1; face_idx <= face.size(); face_idx++) {
        for(uint i = 0; i < FACE_SEGMENTS.size(); i+=2) {

            float confidence1 = face[to_string(face_idx)][FACE_SEGMENTS[i]][2].get<float>();
            if (confidence1 < FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            Point2f p1(w*face[to_string(face_idx)][FACE_SEGMENTS[i]][0].get<float>(), h*face[to_string(face_idx)][FACE_SEGMENTS[i]][1].get<float>());

            float confidence2 = face[to_string(face_idx)][FACE_SEGMENTS[i+1]][2].get<float>();
            if (confidence2 < FACE_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

            bool highconfidence = (confidence1 > FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > FACE_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

            int width = highconfidence ? (bg ? 4 : 2) : (bg? 2 : 1);

            Point2f p2(w*face[to_string(face_idx)][FACE_SEGMENTS[i+1]][0].get<float>(), h*face[to_string(face_idx)][FACE_SEGMENTS[i+1]][1].get<float>());

            cv::line(image, p1, p2, bg ? WHITE : FACE_COLORS[i/2], width, cv::LINE_AA);
        }

        // pupils
        for(uint i = 68; i < NB_FACE_FEATURES; i++) {
            float confidence = face[to_string(face_idx)][i][2].get<float>();
            if (confidence < PUPILS_CONFIDENCE_THRESHOLD) continue;

            Point2f p(w*face[to_string(face_idx)][i][0].get<float>(), h*face[to_string(face_idx)][i][1].get<float>());
            cv::circle(image, p, 3, B, 1, cv::LINE_AA);
        }
    }

    return image;
}

cv::Mat drawHands(cv::Mat image, const json& hand, bool bg=false) {

    auto w = image.size().width;
    auto h = image.size().height;

    for(uint hand_idx = 1; hand_idx <= hand.size(); hand_idx++) {
        for(auto handeness : {"left", "right"}) {
            for(uint i = 0; i < HAND_SEGMENTS.size(); i+=2) {

                float confidence1 = hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i]][2].get<float>();
                if (confidence1 < HAND_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

                Point2f p1(w*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i]][0].get<float>(), h*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i]][1].get<float>());

                float confidence2 = hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i+1]][2].get<float>();
                if (confidence2 < HAND_FEATURE_LOW_CONFIDENCE_THRESHOLD) continue;

                bool highconfidence = (confidence1 > HAND_FEATURE_HIGH_CONFIDENCE_THRESHOLD && confidence2 > HAND_FEATURE_HIGH_CONFIDENCE_THRESHOLD); 

                int width = highconfidence ? (bg ? 4 : 2) : (bg? 2 : 1);

                Point2f p2(w*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i+1]][0].get<float>(), h*hand[to_string(hand_idx)][handeness][HAND_SEGMENTS[i+1]][1].get<float>());

                cv::line(image, p1, p2, bg ? WHITE : HAND_COLORS[i/2], width, cv::LINE_AA);
            }
        }
    }

    return image;
}


cv::Mat drawPose(cv::Mat image, const json& frame, bool skeletons, bool faces, bool hands) {

    Mat bg = image.clone();

    // first, background
    if(skeletons) bg = drawSkeleton(bg, frame["poses"], true); // only background
    if(hands) bg = drawHands(bg, frame["hands"], true);
    if(faces) bg = drawFace(bg, frame["faces"], true);

    cv::addWeighted(image, 0.5, bg, 0.5, 0.0, image);

    // then foreground
    if(skeletons) image = drawSkeleton(image, frame["poses"], false); // only foreground
    if(hands) image = drawHands(image, frame["hands"], false);
    if(faces) image = drawFace(image, frame["faces"], false);


    return image;


}

void printGazeEstimate(const json& frame, bool mirror, const string& topic) {

#ifdef WITH_CAFFE
    auto gaze = gazeEstimator.estimate(frame, mirror);
    cout << topic << ": " << gaze << endl;
#else
    cerr << "Caffe is required to estimate gaze." << endl;
#endif
}


Point2f plotGazeEstimate(cv::Mat& image, const json& frame, bool mirror, const string& topic) {

#ifdef WITH_CAFFE

    auto gaze = gazeEstimator.estimate(frame, mirror);

    Point2f res;

    if(topic.find("yellow") != std::string::npos) {
        yellowGaze.append(gaze);
        res = Point2f(yellowGaze.get().x * SANDTRAY_LENGTH, yellowGaze.get().y * SANDTRAY_WIDTH);
    }
    else {
        purpleGaze.append(gaze);
        res = Point2f(yellowGaze.get().x * SANDTRAY_LENGTH, yellowGaze.get().y * SANDTRAY_WIDTH);
    }

    cv::circle(image, Point2f(purpleGaze.get().x * SANDTRAY_LENGTH, purpleGaze.get().y * SANDTRAY_WIDTH), 2, A, -1, cv::LINE_AA);
    cv::circle(image, Point2f(yellowGaze.get().x * SANDTRAY_LENGTH, yellowGaze.get().y * SANDTRAY_WIDTH), 2, D, -1, cv::LINE_AA);

    return res;
#else
    return Point2f();
#endif
}

cv::Mat plotGazeAccuracyDistribution() {

    Mat image(Size(700,400), CV_8UC3, Scalar(0,0,0));

    auto height = image.size().height;
    auto caption_margin = 30; // pixels at the bottom to display axis caption

    auto bar_width = 10;

    auto bin_size = gazeAccuracyDistribution.bin_size;
    auto nb_bins = gazeAccuracyDistribution.nb_bins;
    auto hist = gazeAccuracyDistribution.get();

    auto valmax = std::max(100u, gazeAccuracyDistribution.max);

    for (size_t idx=0; idx < nb_bins; idx++) {
        rectangle(image,
                  Point(idx * bar_width, height - caption_margin), 
                  Point(((idx+1) * bar_width) - 2, height - caption_margin - (hist[idx] * (height - 5 - caption_margin)/(float) valmax)),
                  B, -1);
    }

    // average
    auto avg = gazeAccuracyDistribution.avg();
    rectangle(image,
              Point(avg * bar_width / bin_size - 2, height - caption_margin), 
              Point(avg * bar_width / bin_size + 2, caption_margin),
              A, -1);
    putText(image, to_string((int)avg)+"mm", Point(avg * bar_width / bin_size + 10, caption_margin + 50), FONT_HERSHEY_PLAIN, 1, WHITE, 1, cv::LINE_AA);

    // stddev
    auto stddev = gazeAccuracyDistribution.stddev();
    rectangle(image,
              Point(avg * bar_width / bin_size - stddev, height - caption_margin), 
              Point(avg * bar_width / bin_size + stddev, height - caption_margin - 5),
              C, -1);


    for (int i = 0; i * bar_width < image.size().width; i += 10) {
        putText(image, to_string((int)(i * bin_size)) + "mm", Point(i * bar_width, height - 4),  FONT_HERSHEY_PLAIN, 1, WHITE,1, cv::LINE_AA);
    }

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
        ("topics", po::value<vector<string>>()->multitoken(), "topic(s) to process (must be of type CompressedImage)")
        ("path", po::value<string>(), "record path (must contain experiment.yaml and freeplay.bag)")
        ("video", po::value<string>()->default_value(""), "if set to a path, save result as video (eg '/path/to/video.mkv')")
        ("camera", po::value<bool>()->default_value(true), "show camera stream (if disabled, draws skeletons on black background)")
        ("skeleton", po::value<bool>()->default_value(true), "display skeletons")
        ("face", po::value<bool>()->default_value(true), "display faces")
        ("hand", po::value<bool>()->default_value(true), "display hands")
        ("gutter", po::value<int>()->default_value(0), "gutter (in pixels) between the two faces")
        ("estimategaze", po::value<bool>()->default_value(false), "print the gaze pose estimate to stdout")
        ("gaze", po::value<bool>()->default_value(false), "show gaze estimate")
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

    if (vm.count("topics") == 0) {
        cerr << "You must specify topic(s) to process with --topics" << endl;
        return 1;
    }

    if (!vm.count("path")) {
        cerr << "You must provide a record path.\n";
        return 1;
    }

    bool continous_gaze = vm["continuousgaze"].as<bool>();
    bool estimate_gaze = vm["estimategaze"].as<bool>();

#ifdef WITH_CAFFE
    if(estimate_gaze) gazeEstimator.initialize();
#endif

    int gutter = vm["gutter"].as<int>();

    bool with_video_bg = vm["camera"].as<bool>();

    bool show_skel = vm["skeleton"].as<bool>();
    bool show_face = vm["face"].as<bool>();
    bool show_hand = vm["hand"].as<bool>();
    bool no_draw = !show_skel && !show_face && !show_hand;

    int total_nb_frames = 0;

    auto video_path = vm["video"].as<string>();
    bool save_as_video = !video_path.empty();

    std::vector<std::string> topics(vm["topics"].as<vector<string>>());

    json root;

    if(!no_draw) {
        auto start = std::chrono::system_clock::now();
        cerr << "Opening " << POSES_FILE << "..." << flush;
        std::ifstream file(vm["path"].as<string>() + "/" + POSES_FILE);
        file >> root;

        auto end = std::chrono::system_clock::now();
        auto elapsed = end - start;

        cerr << "done (took " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)" << endl << endl;

        for (const auto& topic : topics) {
            auto nb_frames = root[topic]["frames"].size();
            if (nb_frames == 0) {
                cerr << "Found no frames for topic " << topic << " in " << POSES_FILE << ". Aborting." << endl;
                exit(1);
            }
            cerr << "Found " << nb_frames << " poses to render for topic " << topic << endl << endl;
            total_nb_frames += nb_frames;
        }
    }

    rosbag::Bag bag;
    rosbag::View view;

    if (with_video_bg) {
        cerr << "Opening " << vm["path"].as<string>() << "/" << BAG_FILE << "..." << endl;
        bag.open(vm["path"].as<string>() + "/" + BAG_FILE, rosbag::bagmode::Read);
        view.addQuery(bag, rosbag::TopicQuery(topics));
        total_nb_frames = view.size();
        if (total_nb_frames == 0) {
            cerr << "Found no image messages for given topic in " << BAG_FILE << ". Aborting." << endl;
            exit(1);
        }
    }

    map<string, size_t> topicsIndices;

    // feature mirroring, for gaze estimation
    bool mirror = false;

    Size windowSize(960 * topics.size() + gutter, 540);

    Mat gazePlot(Size(SANDTRAY_LENGTH, SANDTRAY_WIDTH), CV_8UC3, Scalar(0,0,0));
    std::ifstream file(vm["path"].as<string>() + "/" + VISUAL_TARGET_POSES_FILE);
    json visual_target_poses;
    file >> visual_target_poses;





    int idx = 0;
    int last_percent = 0;

    {
        VideoWriter videowriter;

        if (save_as_video) {
            videowriter.open(video_path, VideoWriter::fourcc('H', '2', '6', '4'), 30.0, windowSize );
        }


        if(with_video_bg) {
            Mat image(windowSize, CV_8UC3, Scalar(0,0,0));

            for(rosbag::MessageInstance const m : view)
            {
                idx++;

                ///////////////////////
                // Gaze estimation plot
                ///////////////////////

                // fade to black
                //gazePlot = mapBg.clone();
                add(gazePlot, -1, gazePlot);
                float target_idx = idx * visual_target_poses.size() / (float) total_nb_frames;
                Point2f prev_target, next_target;

                prev_target.x = -(float) visual_target_poses[target_idx][0] * 1000; // in mm, 0 > x > 600
                prev_target.y = (float) visual_target_poses[target_idx][1] * 1000; // in mm, 0 > y > 340

                next_target.x = -(float) visual_target_poses[target_idx+1][0] * 1000; // in mm
                next_target.y = (float) visual_target_poses[target_idx+1][1] * 1000; // in mm
                auto target = prev_target + (next_target - prev_target) * (target_idx - (int) target_idx);

                cv::circle(gazePlot, target, 3, E3, -1, cv::LINE_AA);

                auto gazeHist = plotGazeAccuracyDistribution();
                ///////////////////////


                for (size_t t_idx = 0; t_idx < topics.size(); t_idx++) {
                    auto topic = topics[t_idx];

                    if(topic.find("yellow") != string::npos) {
                        mirror = true;
                    }
                    if(topic.find("purple") != string::npos) {
                        mirror = false;
                    }

                    if (m.getTopic() == topic || ("/" + m.getTopic() == topic)) {


                        auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();
                        if (compressed_rgb != NULL) {
                            topicsIndices[topic] += 1;
                            auto camimage = imdecode(compressed_rgb->data,1);
                            Rect roi( Point( 960 * t_idx + (gutter * t_idx), 0 ), camimage.size() );


                            if(!no_draw) {
                                camimage = drawPose(camimage, root[topic]["frames"][topicsIndices[topic]], show_skel, show_face, show_hand);
                            }

                            //if(!no_draw && estimate_gaze) printGazeEstimate(root[topic]["frames"][topicsIndices[topic]], mirror, topic);
                            if(!no_draw && estimate_gaze) {

                                auto gazePose = plotGazeEstimate(gazePlot, root[topic]["frames"][topicsIndices[topic]], mirror, topic);

#ifdef WITH_CAFFE
                                gazeAccuracyDistribution.add(norm(gazePose - target));
#endif

                            }

                            camimage.copyTo( image( roi ) );
                        }
                    }
                }

                if (idx % topics.size() == 0) {
                    if(save_as_video) {
                        videowriter.write(image);
                    }
                    else
                    {
                        imshow("Pose replay", image);
                        //addWeighted(gazePlot, 1, gazePlotWithBg, 1, 0.0, gazePlotWithBg);
                        imshow("Gaze estimate", gazePlot);
                        imshow("Gaze distribution", gazeHist);
                        auto k = waitKey(30) & 0xFF;
                        if (k == 27) interrupted = true;
                        if (k == 32) { // space
                            // pause
                            while (true) {
                                if ((waitKey(30) & 0xFF) == 32) break;
                            }
                        }
                        if (!no_draw && k == 115) show_skel = !show_skel; // s
                        if (!no_draw && k == 102) show_face = !show_face; // f
                        if (!no_draw && k == 104) show_hand = !show_hand; // h
                    }
                }

                int percent = idx * 100 / total_nb_frames;
                if (percent != last_percent) {
                    cerr << "\x1b[FDone " << percent << "% (" << idx << " images)" << endl;
                    last_percent = percent;
                }

                if(interrupted) {
                    cerr << "Interrupted." << endl;
                    break;
                }
            }
        }
        else { // NO VIDEO BACKGROUND
            for(idx = 1; idx <= total_nb_frames; idx++)
            {

                Mat image(windowSize, CV_8UC3, Scalar(0,0,0));
                for (size_t t_idx = 0; t_idx < topics.size(); t_idx++) {
                    auto topic = topics[t_idx];

                    if(topic.find("yellow") != string::npos) {
                        mirror = true;
                    }
                    if(topic.find("purple") != string::npos) {
                        mirror = false;
                    }


                    Mat camimage(960, 540, CV_8UC3, Scalar(0,0,0));
                    Rect roi( Point( 960 * t_idx + (gutter * t_idx), 0 ), camimage.size() );

                    if(!no_draw) {
                        camimage = drawPose(camimage, root[topic]["frames"][idx], show_skel, show_face, show_hand);
                    }

                    if(!no_draw && estimate_gaze) printGazeEstimate(root[topic]["frames"][topicsIndices[topic]], mirror, topic);

                    camimage.copyTo( image( roi ) );
                }

                if(save_as_video) {
                    videowriter.write(image);
                }
                else {
                    imshow("Pose replay", image);
                    auto k = waitKey(30) & 0xFF;
                    if (k == 27) interrupted = true;
                    if (k == 32) { // space
                        // pause
                        while (true) {
                            if ((waitKey(30) & 0xFF) == 32) break;
                        }
                    }

                    if (!no_draw && k == 115) show_skel = !show_skel; // s
                    if (!no_draw && k == 102) show_face = !show_face; // f
                    if (!no_draw && k == 104) show_hand = !show_hand; // h
                }

                int percent = idx * 100 / total_nb_frames;
                if (percent != last_percent) {
                    cerr << "\x1b[FDone " << percent << "% (" << idx << " images)" << endl;
                    last_percent = percent;
                }

                if(interrupted) {
                    cerr << "Interrupted." << endl;
                    break;
                }
            }
        }
    }

    if(save_as_video) {
        cerr << "Video " << video_path << " written" << endl;
    }

    bag.close();
}
