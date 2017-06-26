#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <gazr/head_pose_estimation.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include <boost/program_options.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <yaml-cpp/yaml.h>


#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

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


    auto estimator = HeadPoseEstimation(vm["model"].as<string>());

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
    for(rosbag::MessageInstance const m : view)
    {
        nb_msgs++;

        auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();
        if (compressed_rgb != NULL) {
            auto cvimg = imdecode(compressed_rgb->data,1);
            auto res = estimator.update(cvimg);
            if (res.size() > 0) {
                if(res.size() > 1) {
                    cout << "Frame " << nb_images << " of topic " << m.getTopic() << ": Found more than one face!" << endl;
                    multiples_faces_frames.push_back(nb_msgs);
                }
                nb_images_with_face++;
            }

            nb_images++;
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
