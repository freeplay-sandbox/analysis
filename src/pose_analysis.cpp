#include <iostream>
#include <string>
#include <vector>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>


#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>

#include <image_geometry/pinhole_camera_model.h>

#include <boost/program_options.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;

image_geometry::PinholeCameraModel model;

/**
 * Inherits from message_filters::SimpleFilter<M>
 * to use protected signalMessage function 
 */
template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
    public:
        void newMessage(const boost::shared_ptr<M const> &msg)
        {
            this->signalMessage(msg);
        }
};

// Callback for synchronized messages
void callback(const sensor_msgs::CompressedImage::ConstPtr &img_msg, 
              const sensor_msgs::CameraInfo::ConstPtr &info_msg)
{

    //for (size_t i = 0; i < info_msg->D.size(); ++i)
    //      {
    //              cout << info_msg->D[i] << " ";
    //      }
    //cout << endl;

    model.fromCameraInfo(info_msg);
    auto image = imdecode(img_msg->data,1);
    Mat rect;

    model.rectifyImage(image, rect, cv::INTER_LINEAR);

    Mat debug;
    addWeighted( image, 0.5, rect, 0.5, 0.0, debug);
    imshow("rectification", debug);
    waitKey(10);
}

int main(int argc, char **argv) 
{

    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("version,v", "shows version and exits")
        ("topic", po::value<string>(), "topic to process (must be of type CompressedImage)")
        ("models", po::value<string>()->default_value("models/"), "path to OpenPose models")
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

    ros::Time::init();

    cout << "Opening " << vm["path"].as<string>() << "/freeplay.bag..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/freeplay.bag", rosbag::bagmode::Read);

    vector<string> topics;

    string rgb_topic = vm["topic"].as<string>() + "/image_raw/compressed";
    string rgb_info_topic = vm["topic"].as<string>() + "/camera_info";

    topics.push_back(rgb_topic);
    topics.push_back(rgb_info_topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Set up fake subscribers to capture images
    BagSubscriber<sensor_msgs::CompressedImage> rgb_sub;
    //BagSubscriber<sensor_msgs::CompressedImage> depth_sub;
    BagSubscriber<sensor_msgs::CameraInfo> rgb_info_sub;
    //BagSubscriber<sensor_msgs::CameraInfo> depth_info_sub;

    // Use time synchronizer to make sure we get properly synchronized images
    message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CameraInfo> sync(rgb_sub, rgb_info_sub, 25);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    // Load all messages into our stereo dataset
    for(rosbag::MessageInstance const m : view)
    {
        if (m.getTopic() == rgb_topic || ("/" + m.getTopic() == rgb_topic))
        {
            auto img = m.instantiate<sensor_msgs::CompressedImage>();
            if (img != NULL)
                rgb_sub.newMessage(img);
        }

        if (m.getTopic() == rgb_info_topic || ("/" + m.getTopic() == rgb_info_topic))
        {
            auto info = m.instantiate<sensor_msgs::CameraInfo>();
            if (info != NULL)
                rgb_info_sub.newMessage(info);
        }
    }

    bag.close();
}
