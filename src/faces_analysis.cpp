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

#include <openpose/headers.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;


const string POSES_FILE ("poses.yaml");

/**
 * Format of poses.yaml:
 * details of parts idx is here: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

<topic_name>:
    - poses:
            1: # pose index
            # x,y in image coordinate, c is confidence in [0.0,1.0]
            - [x, y, c] # Nose
            - [x, y, c] # Neck
            - [x, y, c] # RShoulder
            - [x, y, c] # RElbow
            - [x, y, c] # RWrist
            - [x, y, c] # LShoulder
            - [x, y, c] # LElbow
            - [x, y, c] # LWrist
            - [x, y, c] # RHip
            - [x, y, c] # RKnee
            - [x, y, c] # RAnkle
            - [x, y, c] # LHip
            - [x, y, c] # LKnee
            - [x, y, c] # LAnkle
            - [x, y, c] # REye
            - [x, y, c] # LEye
            - [x, y, c] # REar
            - [x, y, c] # LEar
            2:
            - ...
      faces:
            1: # face index
            # x,y in image coordinate, c is confidence in [0.0,1.0]
            - [x, y, c]
            - ... # 70 points in total
            2:
            - ...
      hands:
            1: # hand index
                left:
                # x,y in image coordinate, c is confidence in [0.0,1.0]
                - [x, y, c]
                - ... # 20 points in total
                right:
                # x,y in image coordinate, c is confidence in [0.0,1.0]
                - [x, y, c]
                - ... # 20 points in total
            2:
            ...
    - poses: # 2nd frame
        ...


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

YAML::Node makeYamlFrame(const opPosesKeypoints poses, const opFacesKeypoints faces, const opHandsKeypoints hands) {

    YAML::Node node;

    const int NB_POSE_KEYPOINTS = 18; // COCO_18
    const int NB_FACE_KEYPOINTS = 70; // COCO_18
    const int NB_HAND_KEYPOINTS = 21; // COCO_18

    // pose
    for (int i = 0; i < poses.getSize(0); i++) {
        if(poses.getSize(1) != NB_POSE_KEYPOINTS) throw range_error("Unexpected number of pose keypoints!");

        for (int idx = 0; idx < NB_POSE_KEYPOINTS; idx++) {
            node["poses"][i+1][idx+1] = std::vector<float>({poses.at({i,idx,0}), poses.at({i,idx,1}), poses.at({i,idx,2})});
        }
    }

    // face
    if (!faces.empty())
    {
        for (int i = 0; i < faces.getSize(0); i++) {
            if(faces.getSize(1) != NB_FACE_KEYPOINTS) throw range_error("Unexpected number of face keypoints! Expected " + to_string(NB_FACE_KEYPOINTS) + ", got " + to_string(faces.getSize(1)));

            for (int idx = 0; idx < NB_FACE_KEYPOINTS; idx++) {
                node["faces"][i+1][idx+1] = std::vector<float>({faces.at({i,idx,0}), faces.at({i,idx,1}), faces.at({i,idx,2})});
            }
        }
    }

    if (!hands.empty())
    {
        // left hand
        for (int i = 0; i < hands[0].getSize(0); i++) {
            if(hands[0].getSize(1) != NB_HAND_KEYPOINTS) throw range_error("Unexpected number of left hand keypoints! Expected " + to_string(NB_HAND_KEYPOINTS) + ", got " + to_string(hands[0].getSize(1)));


            for (int idx = 0; idx < NB_HAND_KEYPOINTS; idx++) {
                node["hands"]["left"][i+1][idx+1] = std::vector<float>({hands[0].at({i,idx,0}), hands[0].at({i,idx,1}), hands[0].at({i,idx,2})});
            }
        }

        // right hand
        for (int i = 0; i < hands[1].getSize(0); i++) {
            if(hands[1].getSize(1) != NB_HAND_KEYPOINTS) throw("Unexpected number of right hand keypoints! Expected " + to_string(NB_HAND_KEYPOINTS) + ", got " + to_string(hands[1].getSize(1)));

            for (int idx = 0; idx < NB_HAND_KEYPOINTS; idx++) {
                node["hands"]["right"][i+1][idx+1] = std::vector<float>({hands[1].at({i,idx,0}), hands[1].at({i,idx,1}), hands[1].at({i,idx,2})});
            }
        }
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
        ("topic", po::value<string>(), "topic to process (must be of type CompressedImage)")
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

    if (vm.count("topic") == 0) {
        cout << "You must specify a topic to process" << endl;
        return 1;
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
    const auto keypointScale = op::ScaleMode::InputResolution;
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
                                                      "points", // write keypoints -- path
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
   
    //auto estimator = HeadPoseEstimation(vm["model"].as<string>());

    cout << "Opening " << vm["path"].as<string>() << "/freeplay.bag..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/freeplay.bag", rosbag::bagmode::Read);

    string topic(vm["topic"].as<string>());

    std::vector<std::string> topics;
    topics.push_back(topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    //rosbag::View view(bag);

    cout << view.size() << " messages to process" << endl << endl;

    int nb_images_with_face = 0;
    int last_percent=0;


    YAML::Node posesYaml = YAML::Load(topic + ": []");
    try {
        cout << "Opening existing " << POSES_FILE << "...";
        posesYaml = YAML::LoadFile(vm["path"].as<string>() + "/" + POSES_FILE);
        cout << "done." << endl;
    }
    catch (YAML::BadFile bf) {
        cout << "Creating new " << POSES_FILE << endl;
    }
    

    YAML::Node yamlNode = posesYaml[vm["topic"].as<string>()]["frames"];

    auto nbAlreadyProcessed = yamlNode.size();

    if(nbAlreadyProcessed > 0) {
        if(nbAlreadyProcessed == view.size()) {
            cout << "All frames alreay processed. Quitting now." << endl;
            return 0;
        }
        cout << "Already " << nbAlreadyProcessed << " frames processed for this topic. Skipping them." << endl;
    }

    size_t idx = 0;

    for(rosbag::MessageInstance const m : view)
    {
        idx++;

        if(idx <= nbAlreadyProcessed) continue;

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
                auto yamlFrame = makeYamlFrame(
                                          datumProcessed->at(0).poseKeypoints,
                                          vm["face"].as<bool>() ? datumProcessed->at(0).faceKeypoints : NOFACES,
                                          vm["hand"].as<bool>() ? datumProcessed->at(0).handKeypoints : NOHANDS);
                yamlNode.push_back(yamlFrame);

                if (datumProcessed->at(0).faceRectangles.size() > 0) {
                    nb_images_with_face++;
                }

                //imshow("pose", datumProcessed->at(0).cvOutputData);
                //waitKey();
            }
            else
                cerr << "Failed to emplace frame" << endl;

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


    cout << "Found " << nb_images_with_face << " images with faces out of " << idx << " (" << (nb_images_with_face * 100.f)/idx << "%)" << endl;

    if(vm["face"].as<bool>()) {
        posesYaml[topic]["nb_frames_with_face"] = nb_images_with_face;
    }

    std::ofstream fout(vm["path"].as<string>() + "/" + POSES_FILE);
    fout << posesYaml;

    bag.close();
}
