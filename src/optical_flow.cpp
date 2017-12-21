#include <fstream>
#include <iostream>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <vector>
#include <map>

#include <signal.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/CompressedImage.h>
#include <boost/program_options.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace po = boost::program_options;

//const string BAG_FILE ("freeplay.bag");
const string BAG_FILE ("visual_tracking.bag");


// signal handler to stop the program with ctrl+c
bool interrupted = false;
void my_handler(int s){
    cout << "Caught signal " << s << endl;
    interrupted = true; 
}

Mat showFlow(Mat flow) {
    //extraxt x and y channels
    Mat xy[2]; //X,Y
    split(flow, xy);

    //calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    //convert to BGR and show
    Mat bgr;//CV_32FC3 matrix
    cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    return bgr;
}


int main(int argc, char **argv) {

    ////////////////////////////////////////////////////////////////////// 
    // configure the signal handler to be able to interrupt the tool with
    // ctrl+c

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    ////////////////////////////////////////////////////////////////////// 



    ////////////////////////////////////////////////////////////////////// 
    // Command-line program options
    //
    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("version,v", "shows version and exits")
        ("path", po::value<string>(), "record path (must contain freeplay.bag)")
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
    ////////////////////////////////////////////////////////////////////// 


    // those are the RGB video topics that we want to read from the bag file
    std::vector<std::string> topics {"camera_purple/rgb/image_raw/compressed", 
                                     "camera_yellow/rgb/image_raw/compressed"};


    int total_nb_frames = 0;

    ////////////////////////////////////////////////////////////////////// 
    // Loads freeplay.bag and freeplay.poses.json
    //
    rosbag::Bag bag;
    rosbag::View view;

    cerr << "Opening " << vm["path"].as<string>() << "/" << BAG_FILE << " (this might take up to a few minutes, depending on your hard-drive)..." << endl;
    bag.open(vm["path"].as<string>() + "/" + BAG_FILE, rosbag::bagmode::Read);
    view.addQuery(bag, rosbag::TopicQuery(topics));
    total_nb_frames = view.size();
    if (total_nb_frames == 0) {
        cerr << "Found no image messages for given topic in " << BAG_FILE << ". Aborting." << endl;
        exit(1);
    }
    
    ////////////////////////////////////////////////////////////////////// 

    // Keeps track of how many frames we have seen on each topic
    // This is needed to match the ROS video frame with the corresponding entry
    // in freeplay.poses.json
    map<string, size_t> topicsIndices;

    Size windowSize(960 * topics.size(), 540);
    //Mat image(windowSize, CV_32FC2, Scalar(0,0,0));

    // for each topic, stores a pair <current image, prev image>
    map<string, pair<Mat, Mat>> images;
    
    auto opticalFlow = cv::createOptFlow_DualTVL1();
    // tau = 0.25,
    // lambda = 0.15,
    // theta = 0.3,
    // nscales = 5,
    // warps = 5,
    // epsilon = 0.01,
    // innnerIterations = 30,
    // outerIterations = 10,
    // scaleStep = 0.8,
    // gamma = 0.0,
    // medianFiltering = 5,
    // useInitialFlow = false 

    int idx = 0;
    int last_percent = 0;

    // iterate over each of the ROS messages in 'view' (ie, the 2 RGB video streams)
    for(rosbag::MessageInstance const m : view)
    {
        idx++;

        // iterate over the topics of interest
        for (size_t t_idx = 0; t_idx < topics.size(); t_idx++) {

            auto topic = topics[t_idx];

            if (m.getTopic() == topic || ("/" + m.getTopic() == topic)) {

                auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();

                if (compressed_rgb != NULL) {

                    topicsIndices[topic] += 1;

                    // the ROS video stream are compressed as PNG. Simply
                    // calling OpenCV's imdecode on the raw, compressed data,
                    // returns the uncompressed image.
                    auto camimage = imdecode(compressed_rgb->data,1);
                    cvtColor(camimage, camimage, COLOR_BGR2GRAY);
                    images[topic].second = images[topic].first;
                    images[topic].first = camimage;

                    if(topicsIndices[topic] > 1) {
                        Mat optflow;
                        opticalFlow->calc(images[topic].first, images[topic].second, optflow);

                        imshow(topic, showFlow(optflow));
                        //Rect roi( Point( 960 * t_idx, 0 ), camimage.size() );

                        //optflow.copyTo(image(roi));
                    }
                }

            }

        }

        // once we've seen at least one frame per topic, we update the display
        if (idx % topics.size() == 0) {
            //imshow("Optical flow -- press Space to pause, Esc to quit", image);

            auto k = waitKey(30) & 0xFF;
            if (k == 27) interrupted = true;
            if (k == 32) { // space to pause
                while (true) {
                    if ((waitKey(30) & 0xFF) == 32) break;
                }
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


    bag.close();
}
