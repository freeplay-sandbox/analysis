#include <iostream>
#include <fstream>
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

#include <openpose/headers.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;


const string POSES_FILE ("poses.json");

/**
 * Format of poses.json:
 * details of parts idx is here: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

 {<topic_name>:
    {"frames" : [{
        "ts": <timestamp in floating sec>,
        "poses": {
            "1": [ # pose index
                # x,y in image coordinate, c is confidence in [0.0,1.0]
                [x, y, c], # Nose
                [x, y, c], # Neck
                [x, y, c], # RShoulder
                [x, y, c], # RElbow
                [x, y, c], # RWrist
                [x, y, c], # LShoulder
                [x, y, c], # LElbow
                [x, y, c], # LWrist
                [x, y, c], # RHip
                [x, y, c], # RKnee
                [x, y, c], # RAnkle
                [x, y, c], # LHip
                [x, y, c], # LKnee
                [x, y, c], # LAnkle
                [x, y, c], # REye
                [x, y, c], # LEye
                [x, y, c], # REar
                [x, y, c] # LEar
            ],
            "2": [ # if present, second skeleton
              ...
            ]
      },
      "faces": {
            "1": [ # face index
              # x,y in image coordinate, c is confidence in [0.0,1.0]
              [x, y, c],
              ... # 70 points in total
            ],
            "2": [
               ...
            ]
      }
      "hands": {
            "1": { # hand index
                "left": [
                    # x,y in image coordinate, c is confidence in [0.0,1.0]
                    [x, y, c],
                    ... # 20 points in total
                ],
                "right": [
                    # x,y in image coordinate, c is confidence in [0.0,1.0]
                    [x, y, c],
                    ... # 20 points in total
                ]
            },
            "2":
              ...
        }
    },
    { # 2nd frame
        "ts": ...
        "poses":
        ...
    }
    ]
  }
}


*/


bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
}

typedef op::Array<float> opPosesKeypoints;
typedef op::Array<float> opFacesKeypoints;
typedef std::array<op::Array<float>, 2> opHandsKeypoints;

const opFacesKeypoints NOFACES;
const opHandsKeypoints NOHANDS{};

/**
 * 'compress' string representation of double to make them as small as possible
 */
std::string str(double n, int precision = 2) {

    if (n == 0) return "0";
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << n;

    auto s = out.str();

    // remove trailing zeros
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    // remove trailing point
    if(s.back() == '.') {
        s.pop_back();
    }

    return s;
}

std::string str(std::array<double,3> p) {

    std::ostringstream out;
    out << "[" << str(p[0], 4) << "," << str(p[1], 4) << "," << str(p[2]) << "]";
    return out.str();
}

std::string makePoseFrame(const opPosesKeypoints poses, const opFacesKeypoints faces, const opHandsKeypoints hands) {

    std::string node;

    const int NB_POSE_KEYPOINTS = 18; // COCO_18
    const int NB_FACE_KEYPOINTS = 70;
    const int NB_HAND_KEYPOINTS = 21;

    // pose
    node += "\"poses\":{";
    for (int i = 0; i < poses.getSize(0); i++) {
        node += "\"" + to_string(i+1) + "\":[";
        if(poses.getSize(1) != NB_POSE_KEYPOINTS) throw range_error("Unexpected number of pose keypoints!");

        for (int idx = 0; idx < NB_POSE_KEYPOINTS; idx++) {
            node += str({{poses.at({i,idx,0}), poses.at({i,idx,1}), poses.at({i,idx,2})}});
            if (idx != NB_POSE_KEYPOINTS-1) node +=",";
        }
        node += "]";
        if (i != poses.getSize(0) - 1) node +=",";
    }
    node += "}";

    // face
    if (!faces.empty())
    {
        node += ",\"faces\":{";
        for (int i = 0; i < faces.getSize(0); i++) {
            node += "\"" + to_string(i+1) + "\":[";
            if(faces.getSize(1) != NB_FACE_KEYPOINTS) throw range_error("Unexpected number of face keypoints! Expected " + to_string(NB_FACE_KEYPOINTS) + ", got " + to_string(faces.getSize(1)));

            for (int idx = 0; idx < NB_FACE_KEYPOINTS; idx++) {
                node += str({{faces.at({i,idx,0}), faces.at({i,idx,1}), faces.at({i,idx,2})}});
                if (idx != NB_FACE_KEYPOINTS-1) node +=",";
            }
            node += "]";
            if (i != faces.getSize(0) - 1) node +=",";
        }
        node += "}";
    }

    if (!hands.empty())
    {
        node += ",\"hands\":{";
        for (int i = 0; i < hands[0].getSize(0); i++) {
            node += "\"" + to_string(i+1) + "\":{";
            if(hands[0].getSize(1) != NB_HAND_KEYPOINTS) throw range_error("Unexpected number of left hand keypoints! Expected " + to_string(NB_HAND_KEYPOINTS) + ", got " + to_string(hands[0].getSize(1)));
            if(hands[1].getSize(1) != NB_HAND_KEYPOINTS) throw range_error("Unexpected number of right hand keypoints! Expected " + to_string(NB_HAND_KEYPOINTS) + ", got " + to_string(hands[1].getSize(1)));
            // left hand
            node += "\"left\":[";
            for (int idx = 0; idx < NB_HAND_KEYPOINTS; idx++) {
                node += str({{hands[0].at({i,idx,0}), hands[0].at({i,idx,1}), hands[0].at({i,idx,2})}});
                if (idx != NB_HAND_KEYPOINTS-1) node +=",";
            }
            // right hand
            node += "],\"right\":[";
            for (int idx = 0; idx < NB_HAND_KEYPOINTS; idx++) {
                node += str({{hands[1].at({i,idx,0}), hands[1].at({i,idx,1}), hands[1].at({i,idx,2})}});
                if (idx != NB_HAND_KEYPOINTS-1) node +=",";
            }
            node += "]";
            node += "}";
            if (i != hands[0].getSize(0) - 1) node +=",";
        }
        node += "}";
    }

    return node;
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
        ("models", po::value<string>()->default_value("models/"), "path to OpenPose models")
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

    if (!vm.count("path")) {
        cerr << "You must provide a record path.\n";
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    // Configuration OpenPOSE
    ///////////////////////////////////////////////////////////////////////////
    const auto outputSize = op::Point<int>(960, 540);
    // netInputSize
    const auto netInputSize = op::Point<int>(656, 368);
    // faceNetInputSize
    const auto faceNetInputSize = op::Point<int>(368,368); // multiples of 16
    // handNetInputSize
    const auto handNetInputSize = op::Point<int>(368,368); // multiples of 16
    // poseModel
    const auto poseModel = op::PoseModel::COCO_18;
    // keypointScale
    const auto keypointScale = op::ScaleMode::ZeroToOne;
    // heatmaps to add
    const auto heatMapTypes = vector<op::HeatMapType>(); // no heat map
    const auto heatMapScale = op::ScaleMode::UnsignedChar;
    
    // Configure OpenPose
    google::InitGoogleLogging("openPoseTutorialWrapper1");
    op::ConfigureLog::setPriorityThreshold(op::Priority::NoOutput);
    op::Wrapper<std::vector<op::Datum>> opWrapper{op::ThreadManagerMode::Asynchronous};

    // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
    const op::WrapperStructPose wrapperStructPose{netInputSize,
                                                  outputSize, 
                                                  keypointScale, 
                                                  -1, // nb GPU: use all available
                                                  0, // nb GPU start
                                                  1, // Number of scales to average.
                                                  0.3f, // Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1
                                                  op::RenderMode::Cpu, // render a bit faster on CPU
                                                  poseModel,
                                                  true, // blends original image and skeleton
                                                  0.6, // alpha blending pose
                                                  0.7, // alpha blending heatmap
                                                  0, // part to show, 0 shows all
                                                  vm["models"].as<string>(), // models folder
                                                  heatMapTypes, heatMapScale, 
                                                  0.05, // render threshold -- only render part with threshold above it
                                                };

    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{vm["face"].as<bool>(), // enables face detection
                                                  faceNetInputSize, 
                                                  op::RenderMode::Cpu, // render a bit faster on CPU
                                                  0.6, // alpha blending pose
                                                  0.7, // alpha blending heatmap
                                                  0.4, // render threshold -- only render part with threshold above it
                                                };

    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{vm["hand"].as<bool>(), // enables hand detection
                                                  handNetInputSize,
                                                  1, // Analogous to `scale_number` but applied to the hand keypoint detector. Our best results were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4
                                                  0.4f, // hand_scale_range
                                                  true, // hand tracking
                                                  op::RenderMode::Cpu, // render a bit faster on CPU
                                                  0.6, // alpha blending pose
                                                  0.7, // alpha blending heatmap
                                                  0.2, // render threshold -- only render part with threshold above it
                                                };

    // Consumer (comment or use default argument to disable any output)
    const bool displayGui = false;
    const bool guiVerbose = false;
    const bool fullScreen = false;
    const op::WrapperStructOutput wrapperStructOutput{displayGui,
                                                      guiVerbose, 
                                                      fullScreen,
                                                      "", // write keypoints -- path
                                                      op::DataFormat::Yaml,
                                                      "", // Directory to write people pose data in *.json format, compatible with any OpenCV version
                                                      "", // Full file path to write people pose data with *.json COCO validation format
                                                      "", // write_image - Directory to write rendered frames in `write_images_format` image format.
                                                      "png", //image format
                                                      "", // write_video - Full file path to write rendered frames in motion JPEG video format. It might fail if the final path does not finish in `.avi`.
                                                      "", // write_heatmaps -- directory, 
                                                      "png" // heatmaps image format
                                                    };
    // Configure wrapper
    opWrapper.configure(wrapperStructPose,
                        wrapperStructFace, 
                        wrapperStructHand, 
                        op::WrapperStructInput{}, 
                        wrapperStructOutput);

    // openpose -- start
    cout << "Starting the pose estimator thread" << endl;
    opWrapper.start();

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    cout << "Opening " << vm["path"].as<string>() << "/freeplay.bag..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/freeplay.bag", rosbag::bagmode::Read);

    string purple_topic("camera_purple/rgb/image_raw/compressed");
    string yellow_topic("camera_yellow/rgb/image_raw/compressed");
    string env_topic("env_camera/qhd/image_color/compressed");

    std::vector<std::string> topics;
    topics.push_back(purple_topic);
    topics.push_back(yellow_topic);
    topics.push_back(env_topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    //rosbag::View view(bag);

    cout << view.size() << " messages to process" << endl << endl;


    uint total_idx = 0;

    string json("{");

    for (const auto& topic : topics) {
        json += "\"" + topic + "\":{\"frames\":[";

        uint nb_images_with_face = 0;
        uint idx = 0;

        cout << "Processing of topic " << topic << "..." << endl << endl;
        for(rosbag::MessageInstance const m : view)
        {
            if (m.getTopic() == topic || ("/" + m.getTopic() == topic)) {
                idx++;
                total_idx++;

                auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();
                if (compressed_rgb != NULL) {
                    auto cvimg = imdecode(compressed_rgb->data,1);


                    // Create new datum
                    auto datumsPtr = std::make_shared<std::vector<op::Datum>>();
                    datumsPtr->emplace_back();
                    auto& datum = datumsPtr->at(0);
                    
                    // Fill datum
                    datum.cvInputData = cvimg;

                    auto successfullyEmplaced = opWrapper.waitAndEmplace(datumsPtr);
                    // Pop frame
                    std::shared_ptr<std::vector<op::Datum>> datumProcessed;
                    if (successfullyEmplaced && opWrapper.waitAndPop(datumProcessed)) {
                        json += "{";
                        json += "\"ts\":" + to_string(compressed_rgb->header.stamp.toSec()) +",";
                        json += makePoseFrame(
                                            datumProcessed->at(0).poseKeypoints,
                                            vm["face"].as<bool>() ? datumProcessed->at(0).faceKeypoints : NOFACES,
                                            vm["hand"].as<bool>() ? datumProcessed->at(0).handKeypoints : NOHANDS);

                        json += "},";
                        if (datumProcessed->at(0).faceRectangles.size() > 0) {
                            nb_images_with_face++;
                        }

                        //imshow("pose", datumProcessed->at(0).cvOutputData);
                        //waitKey();
                    }
                    else
                        cerr << "Failed to emplace frame" << endl;

                }


                int percent = total_idx * 100 / view.size();
                //if (percent != last_percent) {
                cout << "\x1b[FDone " << percent << "% (" << total_idx << " images)" << endl;
                //    last_percent = percent;
                //}

            }
            
            if(interrupted) {
                cout << "Interrupted." << endl;
                break;
            }
        }

        if(json.back() == ',') {
            json.pop_back();
        }
        json += "]},";

        cout << "\x1b[2FTopic " << topic << " done (" << idx << " frames)" << endl << endl;
    }


    if(json.back() == ',') {
        json.pop_back();
    }

    json += "}";

    std::ofstream fout(vm["path"].as<string>() + "/" + POSES_FILE);
    fout << json;
    cout << "Wrote " << vm["path"].as<string>() + "/" + POSES_FILE << endl;

    bag.close();
}
