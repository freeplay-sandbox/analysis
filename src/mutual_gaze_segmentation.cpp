#include <string>
#include <map>
#include <vector>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <boost/program_options.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#define UPDATE_RATE 1000 // update progress every X messages

using namespace std;
using namespace cv;
namespace enc = sensor_msgs::image_encodings;
namespace po = boost::program_options;

// Compression formats
enum compressionFormat
{
  UNDEFINED = -1, INV_DEPTH
};

// Compression configuration
struct ConfigHeader
{
  // compression format
  compressionFormat format;
  // quantization parameters (used in depth image compression)
  float depthParam[2];
};

/** Taken from https://github.com/ros-perception/image_transport_plugins/blob/indigo-devel/compressed_depth_image_transport/src/codec.cpp
 */
cv::Mat decodeCompressedDepthImage(const sensor_msgs::CompressedImage& message)
{

  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

  // Copy message header
  cv_ptr->header = message.header;

  // Assign image encoding
  std::string image_encoding = message.format.substr(0, message.format.find(';'));
  cv_ptr->encoding = image_encoding;

  // Decode message data
  if (message.data.size() > sizeof(ConfigHeader))
  {

    // Read compression type from stream
    ConfigHeader compressionConfig;
    memcpy(&compressionConfig, &message.data[0], sizeof(compressionConfig));

    // Get compressed image data
    const std::vector<uint8_t> imageData(message.data.begin() + sizeof(compressionConfig), message.data.end());

    // Depth map decoding
    float depthQuantA, depthQuantB;

    // Read quantization parameters
    depthQuantA = compressionConfig.depthParam[0];
    depthQuantB = compressionConfig.depthParam[1];

    if (enc::bitDepth(image_encoding) == 32)
    {
      cv::Mat decompressed;
      try
      {
        // Decode image data
        decompressed = cv::imdecode(imageData, cv::IMREAD_UNCHANGED);
      }
      catch (cv::Exception& e)
      {
        cout << e.what() << endl;
        return Mat();
      }

      size_t rows = decompressed.rows;
      size_t cols = decompressed.cols;

      if ((rows > 0) && (cols > 0))
      {
        cv_ptr->image = Mat(rows, cols, CV_32FC1);

        // Depth conversion
        MatIterator_<float> itDepthImg = cv_ptr->image.begin<float>(),
                            itDepthImg_end = cv_ptr->image.end<float>();
        MatConstIterator_<unsigned short> itInvDepthImg = decompressed.begin<unsigned short>(),
                                          itInvDepthImg_end = decompressed.end<unsigned short>();

        for (; (itDepthImg != itDepthImg_end) && (itInvDepthImg != itInvDepthImg_end); ++itDepthImg, ++itInvDepthImg)
        {
          // check for NaN & max depth
          if (*itInvDepthImg)
          {
            *itDepthImg = depthQuantA / ((float)*itInvDepthImg - depthQuantB);
          }
          else
          {
            *itDepthImg = std::numeric_limits<float>::quiet_NaN();
          }
        }

        // Publish message to user callback
        return cv_ptr->image;
      }
    }
    else
    {
      // Decode raw image
      try
      {
        cv_ptr->image = cv::imdecode(imageData, CV_LOAD_IMAGE_UNCHANGED);
      }
      catch (cv::Exception& e)
      {
        cout << e.what() << endl;
        return Mat();
      }

      size_t rows = cv_ptr->image.rows;
      size_t cols = cv_ptr->image.cols;

      if ((rows > 0) && (cols > 0))
      {
        // Publish message to user callback
        return cv_ptr->image;
      }
    }
  }

  return Mat();
}

void process(Mat rgb, Mat depth) {

    auto t_start = getTickCount();

    //depth.convertTo(depth, CV_32F); // thresholding works on CV_8U or CV_32F but not CV_16U
    //imshow("Input depth", depth);
    //threshold(depth, depth, 0.5, 1.0, THRESH_BINARY_INV);
    depth.convertTo(depth, CV_8U); // masking requires CV_8U. All non-zero values are kept, so '1.0' is fine

    Mat element = getStructuringElement(cv::MORPH_RECT, Size(21,21));
    dilate(depth, depth, element);

    Mat maskedImage;
    rgb.copyTo(maskedImage, depth);

    auto t_end = getTickCount();

    //cout << "Time per frame: " << (t_end-t_start) / getTickFrequency() * 1000. << "ms (" << 1/((t_end-t_start) / getTickFrequency()) << "fps)" << endl;
    //imshow("Input RGB", rgb);
    //imshow("Masked input", maskedImage);
    //waitKey(10);

}

int main(int argc, char **argv) {

    po::positional_options_description p;
    p.add("bag", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produces help message")
            ("version,v", "shows version and exits")
            ("bag", po::value<string>(), "dataset (.bag)")
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

    if (!vm.count("bag")) {
        cerr << "You must provide a .bag file.\n";
        return 1;
    }

    rosbag::Bag bag(vm["bag"].as<string>());
    rosbag::View view(bag);
    auto duration = (view.getEndTime() - view.getBeginTime()).toSec();


    map<string, int> msgs;

    Mat rgb, depth;


    int total_msgs = 0;

    for(rosbag::MessageInstance const m : view) {
        msgs[m.getTopic()] += 1;
        total_msgs++;
    }

    for(auto const& kv : msgs) {
        cout << kv.first << ": " << kv.second << " msgs (" << kv.second / duration << "Hz)" << endl;
    }

    cout << "\n Total duration: " << duration << " secs" << endl;
    cout << "Total messages: " << total_msgs << endl;
    cout << "Starting processing...\n" << endl;

    int progress = 0;

    auto t_start = getTickCount();

    for(rosbag::MessageInstance const m : view)
    {
        progress++;

        if(m.getTopic() == "camera_purple/rgb/image_raw/compressed") {
            auto compressed_rgb = m.instantiate<sensor_msgs::CompressedImage>();
            rgb = imdecode(compressed_rgb->data,1);
        }
        if(m.getTopic() == "camera_purple/depth/image_raw/compressedDepth") {
            auto compressed_depth = m.instantiate<sensor_msgs::CompressedImage>();
            depth = decodeCompressedDepthImage(*compressed_depth);
        }
        if (!rgb.empty() && !depth.empty()) {
            process(rgb, depth);
        }

        if (progress % UPDATE_RATE == 0) {
            auto t_intermediate = getTickCount();
            cout << "Done " << (int) (100. * progress)/total_msgs << "% (" << progress * 1/((t_intermediate-t_start) / getTickFrequency()) << " msgs/s)" << endl;
        }

    }

    auto t_end = getTickCount();
    cout << "Bag processed in " << (t_end-t_start) / getTickFrequency() << "s" << endl;

    bag.close();

}
