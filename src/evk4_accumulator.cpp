#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <memory>
#include <atomic>
#include <csignal>
#include <thread>
#include <fstream>
#include <queue>          // std::queue
extern "C"
{
#include "rm3100_spi_userspace.h"
}

static std::atomic_bool globalShutdown(false);

static void globalShutdownSignalHandler(int signal)
{
    // Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
    if (signal == SIGTERM || signal == SIGINT)
    {
        globalShutdown.store(true);
    }
}

// this class will be used to analyze the events
class EventAnalyzer
{
private:
  std::chrono::time_point<std::chrono::system_clock> lastPrintTime_;

public:
    // class variables to store global information
    int global_counter = 0;                 // this will track how many events we processed
    Metavision::timestamp global_max_t = 0; // this will track the highest timestamp we processed
    Metavision::timestamp first_t = 0;
    Metavision::timestamp last_t = 0;
    int total_events=0;
    std::vector<Metavision::EventCD> events_accumulated;
    double angle = 999;      // radians
    double prev_angle = 999; // radians
    bool mag_ready = false;
    EventAnalyzer()
    {
        std::chrono::time_point<std::chrono::system_clock> t_now = std::chrono::system_clock::now();
        lastPrintTime_ = t_now;

    }
    ~EventAnalyzer() 
    {
    }

    void save_events(std::vector<Metavision::EventCD> events, double current_angle)
    {
    }
    void save_events_thread()
    {
        // std::ofstream events_file("events.dat", std::ios::out | std::ios::binary);
        // std::ofstream mag_timestamps_file("timestamps.txt", std::ios::out);
        // for (auto event:events){
        //     // int64_t ts = event.t;
        //     // uint16_t x = event.x;
        //     // uint16_t y = event.y;
        //     // int8_t pol = event.p;
        //     events_file.write(reinterpret_cast<const char *>(&event.t), sizeof(int64_t));
        //     events_file.write(reinterpret_cast<const char *>(&event.x), sizeof(uint16_t));
        //     events_file.write(reinterpret_cast<const char *>(&event.y), sizeof(uint16_t));
        //     events_file.write(reinterpret_cast<const char *>(&event.p), sizeof(int16_t));
        // }
        // mag_timestamps_file << events.back().t<<" "<<current_angle<<std::endl;
        // events_file.close();
        // mag_timestamps_file.close();
    }

    // this function will be associated to the camera callback
    // it is used to compute statistics on the received events
    void analyze_events(const Metavision::EventCD *begin, const Metavision::EventCD *end)
    {
        // std::cout << "----- New callback! -----" << std::endl;

        // time analysis
        // Note: events are ordered by timestamp in the callback, so the first event will have the lowest timestamp and
        // the last event will have the highest timestamp
        Metavision::timestamp min_t = begin->t;     // get the timestamp of the first event of this callback
        Metavision::timestamp max_t = (end - 1)->t; // get the timestamp of the last event of this callback
        global_max_t = max_t;                       // events are ordered by timestamp, so the current last event has the highest timestamp

        // counting analysis
        // int counter = 0;
        std::copy(begin, end, std::back_inserter(events_accumulated));
        // for (const Metavision::EventCD *ev = begin; ev != end; ++ev)
        // {
        //     // events_accumulated.push_back(*ev);
        //     ++counter; // increasing local counter
        // }
        // global_counter += counter; // increase global counter

        // report
        // std::cout << "There were " << counter << " events in this callback" << std::endl;
        // std::cout << "There were " << global_counter << " total events up to now." << std::endl;
        // std::cout << "The current callback included events from " << min_t << " up to " << max_t << " microseconds."
        //           << std::endl;
        // std::cout << max_t-min_t<<std::endl;

        // std::cout << "----- End of the callback! -----" << std::endl;

        if (mag_ready)
        {
            double current_angle = angle;
            printf("angle:%f\n", current_angle);
            mag_ready = false;
            printf("events:%ld\n", events_accumulated.size());
            // save_events(events_accumulated, current_angle);
            
            // //TODO: make it a separate thread
            // std::thread event_thread([&self]()
            //                         { self.save_events_thread(); });
            prev_angle = current_angle;
            events_accumulated.clear();
            // do things to accumulated events here
        }
    }

    
    // this function will be associated to the camera callback
    // it is used to compute statistics on the received events
    void rate_events_raw(const uint8_t *begin, const uint8_t *end)
    {
        
        std::chrono::time_point<std::chrono::system_clock> t_now = std::chrono::system_clock::now();
        size_t bytes=end-begin;
        total_events+=(int)bytes;
        const double dt = std::chrono::duration<double>(t_now - lastPrintTime_).count();
        if (dt>=1e9){
            lastPrintTime_ = t_now;
            
        }

    }
    void record_mag()
    {
        start_mag();
        start_drdy();
        int revid_id = get_revid_id();
        printf("REVID(correct ID is 22): %X\n", revid_id);
        uint16_t cycle_count = 50;
        change_cycle_count(cycle_count);
        set_continuous_measurement(false);
        uint8_t tmrc_value = 0x92;
        set_tmrc(tmrc_value);
        set_continuous_measurement(true);
        struct Measurements res;
        while (!globalShutdown.load(std::memory_order_relaxed))
        {
            res = get_measurement(100000, true);
            // printf("x:%ld y:%ld pass:%d\n",res.x,res.y,res.ret);
            angle = atan2((double)res.y, (double)res.x) + M_PIl;
            mag_ready = true;
        }
        end_mag();
        end_drdy();
    }
};
// main loop
int main(int argc, char *argv[])
{
    std::cout << "Main thread" << std::endl;
    // Install signal handler for global shutdown.
    struct sigaction shutdownAction;

    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags = 0;
    sigemptyset(&shutdownAction.sa_mask);
    sigaddset(&shutdownAction.sa_mask, SIGTERM);
    sigaddset(&shutdownAction.sa_mask, SIGINT);
    if (sigaction(SIGTERM, &shutdownAction, NULL) == -1)
    {
        return (EXIT_FAILURE);
    }

    if (sigaction(SIGINT, &shutdownAction, NULL) == -1)
    {
        return (EXIT_FAILURE);
    }
    std::cout << "Main thread" << std::endl;

    Metavision::Camera cam;       // create the camera
    EventAnalyzer event_analyzer; // create the event analyzer
    std::thread event_thread([&event_analyzer]()
                             { event_analyzer.record_mag(); });
    std::cout << "Mag thread started!" << std::endl;

    if (argc >= 2)
    {
        // if we passed a file path, open it
        cam = Metavision::Camera::from_file(argv[1]);
    }
    else
    {
        // open the first available camera
        cam = Metavision::Camera::from_first_available();
    }

    // // to analyze the events, we add a callback that will be called periodically to give access to the latest events
    // cam.cd().add_callback([&event_analyzer](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end)
    //                       { event_analyzer.analyze_events(ev_begin, ev_end); });
    // to analyze the events, we add a callback that will be called periodically to give access to the latest events
    cam.raw_data().add_callback([&event_analyzer](const uint8_t *ev_begin, const uint8_t *ev_end)
                          { event_analyzer.rate_events_raw(ev_begin, ev_end); });
    // start the camera
    cam.start();
    std::cout << "Camera started!" << std::endl;
    // keep running while the camera is on or the recording is not finished
    while (cam.is_running() && (!globalShutdown.load(std::memory_order_relaxed)))
    {
        // std::cout << "Camera is running!" << std::endl;
    }

    // the recording is finished, stop the camera.
    // Note: we will never get here with a live camera
    cam.stop();
    event_thread.join();

    // print the global statistics
    double length_in_seconds = event_analyzer.global_max_t / 1000000.0;
    std::cout << "There were " << event_analyzer.global_counter << " events in total." << std::endl;
    std::cout << "The total duration was " << length_in_seconds << " seconds." << std::endl;
    if (length_in_seconds >= 1)
    { // no need to print this statistics if the video was too short
        std::cout << "There were " << event_analyzer.global_counter / (event_analyzer.global_max_t / 1000000.0)
                  << " events per seconds on average." << std::endl;
    }
}