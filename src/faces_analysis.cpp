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

bool interrupted = false;

void my_handler(int s){
    printf("Caught signal %d\n",s);
    interrupted = true; 
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
        ("model", po::value<string>(), "dlib's trained face model")
        ("path", po::value<string>(), "record path (must contain experiment.yaml and freeplay.bag)")
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

    if (vm.count("model") == 0) {
        cout << "You must specify the path to a trained dlib's face model\n"
            << "with the option --model." << endl;
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
                                                  "models/", // models folder
                                                  heatMapTypes, heatMapScale, 
                                                  0.05, // render threshold -- only render part with threshold above it
                                                };

    // Face configuration (use op::WrapperStructFace{} to disable it)
    const op::WrapperStructFace wrapperStructFace{true, // enables face detection
                                                  faceNetInputSize, 
                                                  op::RenderMode::Cpu, // render a bit faster on CPU
                                                  0.6, // alpha blending pose
                                                  0.7, // alpha blending heatmap
                                                  0.4, // render threshold -- only render part with threshold above it
                                                };

    // Hand configuration (use op::WrapperStructHand{} to disable it)
    const op::WrapperStructHand wrapperStructHand{true, // enables hand detection
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

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    
    //auto estimator = HeadPoseEstimation(vm["model"].as<string>());

    cout << "Openning " << vm["path"].as<string>() << "/freeplay.bag..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/freeplay.bag", rosbag::bagmode::Read);

    string image_topic(vm["topic"].as<string>());

    std::vector<std::string> topics;
    topics.push_back(image_topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    //rosbag::View view(bag);

    cout << view.size() << " messages to process" << endl << endl;

    int nb_images = 0;
    int nb_images_with_face = 0;
    vector<int> multiples_faces_frames;
    int nb_msgs = 0;
    int last_percent=0;

    // openpose -- start
    cout << "Starting the pose estimator thread" << endl;
    opWrapper.start();


    for(rosbag::MessageInstance const m : view)
    {
        nb_msgs++;

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
                imshow("pose", datumProcessed->at(0).cvOutputData);
                waitKey(10);
                cout << "Frame successfully processed!" << endl;
            }
            else
                cerr << "Failed to emplace frame" << endl;


            //if (res.size() > 0) {
            //    if(res.size() > 1) {
            //        cout << "Frame " << nb_images << " of topic " << m.getTopic() << ": Found more than one face!" << endl;
            //        multiples_faces_frames.push_back(nb_msgs);
            //    }
            //    nb_images_with_face++;
            //}

            //nb_images++;
            
            
            
            
            //if (show_frame) {
            //    imshow("headpose", estimator._debug);
            //    waitKey(10);
            //}

        }

        int percent = nb_msgs * 100 / view.size();
        if (percent != last_percent) {
            cout << "\x1b[FDone " << percent << "% (" << nb_images << " images)" << endl;
            last_percent = percent;
        }

        if(interrupted) break;
    }

    cout << "Found " << nb_images_with_face << " images with faces out of " << nb_images << " (" << (nb_images_with_face * 100.f)/nb_images << "%)" << endl;


    YAML::Node facesyaml = YAML::Load("faces: []");
    try {
        facesyaml = YAML::LoadFile(vm["path"].as<string>() + "/faces.yaml");
        cout << "Updating existing faces.yaml" << endl;
    }
    catch (YAML::BadFile bf) {
        cout << "Creating new faces.yaml" << endl;
    }
    facesyaml["faces"][vm["topic"].as<string>()]["nb_frames"] = nb_images;
    facesyaml["faces"][vm["topic"].as<string>()]["nb_frames_with_face"] = nb_images_with_face;
    facesyaml["faces"][vm["topic"].as<string>()]["multiple_faces"] = multiples_faces_frames;
    std::ofstream fout(vm["path"].as<string>() + "/faces.yaml");
    fout << facesyaml;
    bag.close();
}
