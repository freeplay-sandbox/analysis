#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>

#include <image_geometry/pinhole_camera_model.h>

#include "depth_traits.h"

#include <boost/program_options.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;
using namespace depth_image_proc;

image_geometry::PinholeCameraModel rgb_model;
image_geometry::PinholeCameraModel depth_model;

uint nbSyncFrames = 0;

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


/** Registration code taken from https://github.com/ros-perception/image_pipeline/blob/indigo/depth_image_proc/src/nodelets/register.cpp
*/
template<typename T>
cv::Mat convert(cv::Mat depth,
                cv::Size size,
                const Eigen::Affine3d& depth_to_rgb)
{
    cv::Mat registered_image = Mat::zeros(size, CV_16UC1);

    // Extract all the parameters we need
    double inv_depth_fx = 1.0 / depth_model.fx();
    double inv_depth_fy = 1.0 / depth_model.fy();
    double depth_cx = depth_model.cx(), depth_cy = depth_model.cy();
    double depth_Tx = depth_model.Tx(), depth_Ty = depth_model.Ty();
    double rgb_fx = rgb_model.fx(), rgb_fy = rgb_model.fy();
    double rgb_cx = rgb_model.cx(), rgb_cy = rgb_model.cy();
    double rgb_Tx = rgb_model.Tx(), rgb_Ty = rgb_model.Ty();

    // Transform the depth values into the RGB frame
    /// @todo When RGB is higher res, interpolate by rasterizing depth triangles onto the registered image  
    const T* depth_row = reinterpret_cast<const T*>(&depth.data[0]);
    int row_step = depth.step / sizeof(T);
    T* registered_data = reinterpret_cast<T*>(&registered_image.data[0]);
    int raw_index = 0;
    for (int v = 0; v < depth.size().height; ++v, depth_row += row_step)
    {
        for (int u = 0; u < depth.size().width; ++u, ++raw_index)
        {
            T raw_depth = depth_row[u];
            if (!DepthTraits<T>::valid(raw_depth))
                continue;

            double depth = DepthTraits<T>::toMeters(raw_depth);

            /// @todo Combine all operations into one matrix multiply on (u,v,d)
            // Reproject (u,v,Z) to (X,Y,Z,1) in depth camera frame
            Eigen::Vector4d xyz_depth;
            xyz_depth << ((u - depth_cx)*depth - depth_Tx) * inv_depth_fx,
                      ((v - depth_cy)*depth - depth_Ty) * inv_depth_fy,
                      depth,
                      1;

            // Transform to RGB camera frame
            Eigen::Vector4d xyz_rgb = depth_to_rgb * xyz_depth;

            // Project to (u,v) in RGB image
            double inv_Z = 1.0 / xyz_rgb.z();
            int u_rgb = (rgb_fx*xyz_rgb.x() + rgb_Tx)*inv_Z + rgb_cx + 0.5;
            int v_rgb = (rgb_fy*xyz_rgb.y() + rgb_Ty)*inv_Z + rgb_cy + 0.5;

            if (u_rgb < 0 || u_rgb >= (int)registered_image.size().width ||
                    v_rgb < 0 || v_rgb >= (int)registered_image.size().height)
                continue;

            T& reg_depth = registered_data[v_rgb*registered_image.size().width + u_rgb];
            T  new_depth = DepthTraits<T>::fromMeters(xyz_rgb.z());
            // Validity and Z-buffer checks
            if (!DepthTraits<T>::valid(reg_depth) || reg_depth > new_depth)
                reg_depth = new_depth;
        }
    }

    return registered_image;
}

// Callback for synchronized messages
void callback(const sensor_msgs::CompressedImage::ConstPtr &rgb_msg, 
              const sensor_msgs::CameraInfo::ConstPtr &rgb_info_msg,
              const sensor_msgs::CompressedImage::ConstPtr &depth_msg, 
              const sensor_msgs::CameraInfo::ConstPtr &depth_info_msg)
{

    nbSyncFrames++;

    //for (size_t i = 0; i < info_msg->D.size(); ++i)
    //      {
    //              cout << info_msg->D[i] << " ";
    //      }
    //cout << endl;

    rgb_model.fromCameraInfo(rgb_info_msg);
    cout << "RGB Camera calibration set: fx: " << rgb_model.fx() << ", fy: " << rgb_model.fy() << ", cx: " << rgb_model.cx() << ", cy: " << rgb_model.cy() << endl;
     depth_model.fromCameraInfo(depth_info_msg);
    cout << "Depth Camera calibration set: fx: " << depth_model.fx() << ", fy: " << depth_model.fy() << ", cx: " << depth_model.cx() << ", cy: " << depth_model.cy() << endl;

    auto rgb = imdecode(rgb_msg->data, 1);

    // compressedDepth is a PNG with an additional 12 bytes header. Remove the header and
    // let OpenCV decode it.
    // Note that our encoding is 16UC1 and quantization is not perform in that case (cf https://github.com/ros-perception/image_transport_plugins/blob/indigo-devel/compressed_depth_image_transport/src/codec.cpp#L135)
    decltype(depth_msg->data) depth_png(depth_msg->data.begin () + 12, depth_msg->data.end());
    auto depth = imdecode(depth_png, CV_LOAD_IMAGE_UNCHANGED);

    Mat depth_rect;

    depth_model.rectifyImage(depth, depth_rect, cv::INTER_LINEAR);

    // SR300 transformation between IR camera and RGB camera
    Eigen::Affine3d depth_to_rgb = Eigen::Translation3d(0.03,  // x
                                                        0.0038,  // y
                                                        0.0007) * // z
                                   Eigen::Quaterniond(1.0,   // w
                                                      0.0,   // x
                                                      0.0,   // y
                                                      0.0);  // z

    cv::Size resolution = rgb_model.reducedResolution();

    auto registered_depth = convert<uint16_t>(depth_rect, resolution, depth_to_rgb); // uint16_t -> SR300 cameras encoding

    Mat depth8bit;
    registered_depth.convertTo(depth8bit, CV_8UC1, 1./256);

    Mat debug;
    Mat greyscale;
    cvtColor(rgb, greyscale, CV_RGB2GRAY);
    resize(depth8bit, depth8bit, rgb.size());
    depth8bit *= 100;
    debug = greyscale * 0.7 + depth8bit * 0.3;
    imshow("rectification", debug);
    //imshow("depth rect", depth_rect);
    imshow("depth rect", depth8bit);
    
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
        ("ns", po::value<string>(), "camera namespace to process (must have /rgb and /depth sub namespaces)")
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

    string rgb_topic = vm["ns"].as<string>() + "/rgb/image_raw/compressed";
    string rgb_info_topic = vm["ns"].as<string>() + "/rgb/camera_info";
    string depth_topic = vm["ns"].as<string>() + "/depth/image_raw/compressedDepth";
    string depth_info_topic = vm["ns"].as<string>() + "/depth/camera_info";

    topics.push_back(rgb_topic);
    topics.push_back(rgb_info_topic);
    topics.push_back(depth_topic);
    topics.push_back(depth_info_topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // Set up fake subscribers to capture images
    BagSubscriber<sensor_msgs::CompressedImage> rgb_sub;
    BagSubscriber<sensor_msgs::CompressedImage> depth_sub;
    BagSubscriber<sensor_msgs::CameraInfo> rgb_info_sub;
    BagSubscriber<sensor_msgs::CameraInfo> depth_info_sub;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage,
                                                            sensor_msgs::CameraInfo,
                                                            sensor_msgs::CompressedImage,
                                                            sensor_msgs::CameraInfo> SyncPolicy;

    // Use time synchronizer to make sure we get properly synchronized images
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10),
                                                   rgb_sub,
                                                   rgb_info_sub,
                                                   depth_sub,
                                                   depth_info_sub);
    sync.getPolicy()->setMaxIntervalDuration(ros::Duration(1./30)); // SR300 RGB-D cameras are recorded at 30 FPS

    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));

    uint nbRgbFrames = 0;
    uint nbDepthFrames = 0;

    for(rosbag::MessageInstance const m : view)
    {
        if (m.getTopic() == rgb_topic || ("/" + m.getTopic() == rgb_topic))
        {
            auto img = m.instantiate<sensor_msgs::CompressedImage>();
            if (img != NULL) {
                nbRgbFrames++;
                rgb_sub.newMessage(img);
            }
        }

        if (m.getTopic() == rgb_info_topic || ("/" + m.getTopic() == rgb_info_topic))
        {
            auto info = m.instantiate<sensor_msgs::CameraInfo>();
            if (info != NULL)
                rgb_info_sub.newMessage(info);
        }
        if (m.getTopic() == depth_topic || ("/" + m.getTopic() == depth_topic))
        {
            auto img = m.instantiate<sensor_msgs::CompressedImage>();
            if (img != NULL) {
                nbDepthFrames++;
                depth_sub.newMessage(img);
            }
        }

        if (m.getTopic() == depth_info_topic || ("/" + m.getTopic() == depth_info_topic))
        {
            auto info = m.instantiate<sensor_msgs::CameraInfo>();
            if (info != NULL)
                depth_info_sub.newMessage(info);
        }
    }

    cout << "Got " << nbRgbFrames << " RGB frames and " << nbDepthFrames << " depth frames." << endl;
    cout << "Got " << nbSyncFrames << " synchronized RGB-D pairs (" << (100. * nbSyncFrames)/nbRgbFrames << "\% of RGB frames)" << endl;

    bag.close();
}
