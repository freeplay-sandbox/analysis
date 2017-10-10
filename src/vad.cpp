#include <iostream>


#include <rosbag/bag.h>
#include <rosbag/view.h>


#include <audio_common_msgs/AudioData.h>

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
        ("topic", po::value<string>(), "topic to process (must be of type AudioData)")
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

    if (!vm.count("path")) {
        cerr << "You must provide a record path.\n";
        return 1;
    }


    cout << "Openning " << vm["path"].as<string>() << "/freeplay.bag..." << endl;
    rosbag::Bag bag(vm["path"].as<string>() + "/freeplay.bag", rosbag::bagmode::Read);

    string audio_topic(vm["topic"].as<string>());

    std::vector<std::string> topics;
    topics.push_back(audio_topic);

    rosbag::View view(bag, rosbag::TopicQuery(topics));
    //rosbag::View view(bag);

    cout << view.size() << " messages to process" << endl << endl;

    int nb_msgs = 0;
    int last_percent=0;

    for(rosbag::MessageInstance const m : view)
    {
        nb_msgs++;

        auto msg = m.instantiate<audio_common_msgs::AudioData>();
        if (msg != NULL) {

            int percent = nb_msgs * 100 / view.size();
            if (percent != last_percent) {
                cout << "\x1b[FDone " << percent << "%" << endl;
                last_percent = percent;
            }
        }

        if(interrupted) break;
    }


    //YAML::Node facesyaml = YAML::Load("faces: []");
    //try {
    //    facesyaml = YAML::LoadFile(vm["path"].as<string>() + "/faces.yaml");
    //    cout << "Updating existing faces.yaml" << endl;
    //}
    //catch (YAML::BadFile bf) {
    //    cout << "Creating new faces.yaml" << endl;
    //}
    //facesyaml["faces"][vm["topic"].as<string>()]["nb_frames"] = nb_images;
    //facesyaml["faces"][vm["topic"].as<string>()]["nb_frames_with_face"] = nb_images_with_face;
    //facesyaml["faces"][vm["topic"].as<string>()]["multiple_faces"] = multiples_faces_frames;
    //std::ofstream fout(vm["path"].as<string>() + "/faces.yaml");
    //fout << facesyaml;
    bag.close();
}
