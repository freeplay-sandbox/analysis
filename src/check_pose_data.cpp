#include <fstream>
#include <iostream>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <string>
#include <thread> // std::this_thread
#include <vector>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>
#include "json.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace nlohmann; // json
namespace po = boost::program_options;


//const string BAG_FILE ("rectified_streams.bag");
const string YAML_BAG_FILE ("freeplay.bag.yaml");
const string POSES_FILE ("freeplay.poses.json");

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
        ("path", po::value<string>(), string("record path (must contain " + YAML_BAG_FILE + " and " + POSES_FILE + ")").c_str())
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

    map<string, unsigned int> topics;

    int total_nb_frames = 0;

    json root;

    auto start = std::chrono::system_clock::now();
    cerr << "Opening " << POSES_FILE << "..." << flush;
    std::ifstream file(vm["path"].as<string>() + "/" + POSES_FILE);
    file >> root;

    auto end = std::chrono::system_clock::now();
    auto elapsed = end - start;

    cerr << "done (took " << std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() << "s)" << endl << endl;

    for (json::iterator topic = root.begin(); topic != root.end(); ++topic) {
        auto nb_frames = topic.value()["frames"].size();
        if (nb_frames == 0) {
            cerr << "!! Found no frames for topic " << topic.key() << " in " << POSES_FILE << "." << endl;
        }
        topics[topic.key()] = nb_frames;
        total_nb_frames += nb_frames;
    }

    YAML::Node yaml_topics = YAML::LoadFile(vm["path"].as<string>() + "/" + YAML_BAG_FILE)["topics"];

    bool ok = true;

    for (const auto& t : topics) {
        for (const auto& topic : yaml_topics) {
            if(topic["topic"].as<string>() == t.first) {
                auto nb_msgs = topic["messages"].as<unsigned int>();

                if(nb_msgs == t.second) {
                    cout << topic["topic"] << ": OK (nb frames = nb poses = " << nb_msgs << endl;
                }
                else {
                    ok = false;
                    cout << topic["topic"] << ": INCONSISTENCY (nb frames = " << nb_msgs << ", nb poses = " << t.second << endl;
                }
            }
        }
    }

    if (ok) return 0;
    return 1;

}
