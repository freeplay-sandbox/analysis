#include <opencv2/highgui/highgui.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <OpenFace/LandmarkCoreIncludes.h>
#include <OpenFace/FaceAnalyser.h>
#include <OpenFace/GazeEstimation.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using namespace cv2;

// focal lengths
const double FX = 700;
const double FY = 700;

// optical center
const double CX = 300 ;
const double CY = 200;

int main (int argc, char **argv)
{

    po::positional_options_description p;
    p.add("path", 1);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produces help message")
        ("version,v", "shows version and exits")
//        ("models", po::value<string>()->default_value("models/"), "path to OpenPose models")
//        ("bag", po::value<string>()->default_value("freeplay"), "Bag file, without the '.bag' extension")
        ("path", po::value<string>(), "path to the source video")
//        ("face", po::value<bool>()->default_value(true), "detect faces as well")
//        ("hand", po::value<bool>()->default_value(true), "detect hands as well")
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
        cerr << "You must provide a path to the video to process.\n";
        return 1;
    }


    VideoCapture capture(vm["path"].as<string>());


    // Load the models if images found
    LandmarkDetector::FaceModelParameters det_parameters;

    // The modules that are being used for tracking
    cout << "Loading the model" << endl;
    LandmarkDetector::CLNF face_model(det_parameters.model_location);
    cout << "Model loaded" << endl;

    // Load facial feature extractor and AU analyser (make sure it is static)
    FaceAnalysis::FaceAnalyserParameters face_analysis_params;
    face_analysis_params.OptimizeForImages();
    FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

    dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

    for(;;)
    {
        Mat frame;
        capture >> frame; 

        if (frame.empty()) break;

        Mat greyscale_image;
        // Making sure the image is in uchar grayscale
        //cv::Mat_<uchar> grayscale_image = image_reader.GetGrayFrame();
        cvtColor(frame, greyscale_image, CV_BGR2GRAY);

        // Detect faces in an image
        vector<cv::Rect_<double> > face_detections;
        vector<double> confidences;
        LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);

        // Detect landmarks around detected faces
        int face_det = 0;
        // perform landmark detection for every face detected
        for (size_t face = 0; face < face_detections.size(); ++face)
        {
            // if there are multiple detections go through them
            bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_detections[face], face_model, det_parameters);

            // Estimate head pose and eye gaze
            Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, FX, FY, CX, CY);

            // Gaze tracking, absolute gaze direction
            Point3f gaze_direction0(0, 0, -1);
            Point3f gaze_direction1(0, 0, -1);
            Vec2d gaze_angle(0, 0);

            if (face_model.eye_model)
            {
                GazeAnalysis::EstimateGaze(face_model, gaze_direction0, FX, FY, CX, CY, true);
                GazeAnalysis::EstimateGaze(face_model, gaze_direction1, FX, FY, CX, CY, false);
                gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
            }


            face_analyser.PredictStaticAUsAndComputeFeatures(frame, face_model.detected_landmarks);

        }
    }

    return 0;
}

