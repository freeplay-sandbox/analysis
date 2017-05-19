#include <string>
#include <map>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/program_options.hpp>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace std;
using namespace cv;
namespace po = boost::program_options;

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

    cout << "Duration: " << duration << " secs" << endl;

    map<string, int> msgs;

    for(rosbag::MessageInstance const m : view)
    {
        //std_msgs::Int32::ConstPtr i = m.instantiate<std_msgs::Int32>();
        //if (i != NULL)
        //    std::cout << i->data << std::endl;
        msgs[m.getTopic()] += 1;
    }

    for(auto const& kv : msgs) {
        cout << kv.first << ": " << kv.second << " msgs (" << kv.second / duration << "Hz)" << endl;
    }

    bag.close();

}
