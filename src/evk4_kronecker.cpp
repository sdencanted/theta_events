#include <torch/script.h> // One-stop header.
#include <memory>
#include <string>
#include <stdexcept>
#include <chrono>
#include <vector>
#include <iostream>

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDADevice.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "evt3_types.h"
#include "key.h"
torch::NoGradGuard no_grad;

using timestamp_t = uint64_t;
namespace event_camera_codecs
{
    namespace evt3
    {
        class CudaAllocException : public std::exception
        {
        public:
            char *what()
            {
                return (char *)"Cuda Alloc Exception";
            }
        };

        // this class will be used to analyze the events
        class EventAnalyzer
        {
        public:
            EventAnalyzer(std::string hot_pixels_file)
            {
                if (hot_pixels_file != "")
                {
                    std::string line;
                    int val;
                    std::vector<int> hot_pixels_vector;

                    // Create an input filestream
                    std::ifstream myFile(hot_pixels_file);

                    // Make sure the file is open
                    if (!myFile.is_open())
                        throw std::runtime_error("Could not open file");

                    // Read data, line by line
                    while (std::getline(myFile, line))
                    {
                        // Create a stringstream of the current line
                        std::stringstream ss(line);

                        // Extract each integer
                        while (ss >> val)
                        {
                            hot_pixels_vector.push_back(val);

                            // If the next token is a comma, ignore it and move on
                            if (ss.peek() == ',')
                                ss.ignore();
                        }
                    }

                    // Close file
                    myFile.close();
                    hot_pixels_ = torch::from_blob(hot_pixels_vector.data(), hot_pixels_vector.size(), at::TensorOptions().dtype(torch::kInt32)).to(device_).to(torch::kInt64);

                }
                else
                {
                    hot_pixels_ = torch::empty({0, 25});
                }

                std::cout << torch::cuda::device_count() << std::endl;
                if (!cudaAllocMapped(&ex_list_arr_, 4000000))
                {
                    std::cout << "could not allocate cuda mem x" << std::endl;
                    throw CudaAllocException();
                };
                if (!cudaAllocMapped(&ey_list_arr_, 4000000))
                {
                    std::cout << "could not allocate cuda mem y" << std::endl;
                    throw CudaAllocException();
                };
                if (!cudaAllocMapped(&mat_data_, width_*height_))
                {
                    std::cout << "could not allocate cuda mem mat_data" << std::endl;
                    throw CudaAllocException();
                };
                c10::Device dev = at::cuda::getDeviceFromPtr(ex_list_arr_);
                std::cout << "CUDA Mem allocated " << dev << std::endl;
            }
            ~EventAnalyzer()
            {
                CUDA(cudaFreeHost(ex_list_arr_));
                CUDA(cudaFreeHost(ey_list_arr_));
                CUDA(cudaFreeHost(mat_data_));
            }
            // class variables to store global information
            int global_counter = 0;                 // this will track how many events we processed
            Metavision::timestamp global_max_t = 0; // this will track the highest timestamp we processed
            Metavision::timestamp first_t = 0;
            Metavision::timestamp last_t = 0;

            inline size_t findValidTime(const Event *buffer, size_t numRead, size_t offset = 0)
            {
                size_t i = offset;
                bool hasValidHighTime(false);
                for (; !hasValidTime_ && i < numRead; i++)
                {
                    switch (buffer[i].code)
                    {
                    case Code::TIME_LOW:
                    {
                        if (hasValidHighTime)
                        {
                            hasValidTime_ = true; // will break out of loop
                            break;
                        }
                    }
                    break;
                    case Code::TIME_HIGH:
                    {
                        hasValidHighTime = true;
                    }
                    break;
                    default: // ignore all but time codes
                        break;
                    }
                }
                return (i);
            }

            // time in microseconds
            inline timestamp_t makeTime(timestamp_t high, uint16_t low) { return ((high | low)); }

            inline static timestamp_t update_high_time(uint16_t t, timestamp_t timeHigh)
            {
                // shift right and remove rollover bits to get last time_high
                const timestamp_t lastHigh = (timeHigh >> 12) & ((1ULL << 12) - 1);
                // sometimes the high stamp goes back a little without a rollover really happening
                const timestamp_t MIN_DISTANCE = 10;
                if (t < lastHigh && lastHigh - t > MIN_DISTANCE)
                {
                    // new high time is smaller than old high time, bump high time bits
                    // printf("%x %lx %lx\n", t, lastHigh, lastHigh - t);
                    // std::cout << "rollover detected: new " << t << " old: " << lastHigh << std::endl;
                    timeHigh += (1 << 24); // add to rollover bits
                }
                else if (t < lastHigh)
                {
                    // std::cout << "rollover averted: new " << t << " old: " << lastHigh << std::endl;
                }
                // wipe out lower 24 bits of timeHigh (leaving only rollover bits)
                // and replace non-rollover bits with new time base
                timeHigh = (timeHigh & (~((1ULL << 24) - 1))) | (static_cast<timestamp_t>(t) << 12);
                return (timeHigh);
            }

            // this function will be associated to the camera callback
            // it is used to compute statistics on the received events
            void decode_packet(const uint8_t *begin, const uint32_t num_bytes)
            {
                // std::cout << "----- New callback! -----" << std::endl;

                // int num_threads = 4;
                // int valid_times[num_threads] = {0};
                const size_t numRead = num_bytes / sizeof(Event);
                const Event *buffer = reinterpret_cast<const Event *>(begin);
                size_t first_idx = findValidTime(buffer, numRead);
                for (size_t i = first_idx; i < numRead; i++)
                {
                    switch (buffer[i].code)
                    {
                    case Code::ADDR_X:
                    {
                        const AddrX *e = reinterpret_cast<const AddrX *>(&buffer[i]);
                        if (e->x < width_ && ey_ < height_)
                        {
                            ex_list_arr_[num_events_] = e->x;
                            ey_list_arr_[num_events_] = ey_;
                            num_events_++;
                            // ey_list_.push_back(ey_);
                            // ex_list_.push_back(e->x);
                        }
                    }
                    break;
                    case Code::ADDR_Y:
                    {
                        const AddrY *e = reinterpret_cast<const AddrY *>(&buffer[i]);
                        ey_ = e->y; // save for later
                    }
                    break;
                    case Code::TIME_LOW:
                    {
                        const TimeLow *e = reinterpret_cast<const TimeLow *>(&buffer[i]);
                        timeLow_ = e->t;
                        if (timeLimit_ == 0)
                        {
                            timeLimit_ = makeTime(timeHigh_, timeLow_) + 33333;
                        }
                        else if (makeTime(timeHigh_, timeLow_) > timeLimit_)
                        {
                            // consume the list so far
                            // std::cout << "timestamp hit, accumulating events from " << makeTime(timeHigh_, timeLow_) << std::endl;
                            if (num_events_ > 0)
                            {
                                accumulate_events_omp();
                                num_events_ = 0;
                                // ex_list_.clear();
                                // ey_list_.clear();
                            }
                            timeLimit_ = makeTime(timeHigh_, timeLow_) + 33333;
                        }
                    }
                    break;
                    case Code::TIME_HIGH:
                    {
                        const TimeHigh *e = reinterpret_cast<const TimeHigh *>(&buffer[i]);
                        timeHigh_ = update_high_time(e->t, timeHigh_);
                    }
                    break;
                    case Code::VECT_BASE_X:
                    {
                        const VectBaseX *b = reinterpret_cast<const VectBaseX *>(&buffer[i]);
                        currentPolarity_ = b->pol;
                        currentBaseX_ = b->x;
                        break;
                    }
                    case Code::VECT_8:
                    {
                        const Vect8 *b = reinterpret_cast<const Vect8 *>(&buffer[i]);
                        for (int i = 0; i < 8; i++)
                        {
                            if (b->valid & (1 << i))
                            {
                                const uint16_t ex = currentBaseX_ + i;
                                if (ex < width_ && ey_ < height_)
                                {
                                    // ey_list_.push_back(ey_);
                                    // ex_list_.push_back(ex);

                                    ex_list_arr_[num_events_] = ex;
                                    ey_list_arr_[num_events_] = ey_;
                                    num_events_++;
                                }
                            }
                        }
                        currentBaseX_ += 8;
                        break;
                    }
                    case Code::VECT_12:
                    {
                        const Vect12 *b = reinterpret_cast<const Vect12 *>(&buffer[i]);
                        for (int i = 0; i < 12; i++)
                        {
                            if (b->valid & (1 << i))
                            {
                                const uint16_t ex = currentBaseX_ + i;
                                if (ex < width_ && ey_ < height_)
                                {
                                    ex_list_arr_[num_events_] = ex;
                                    ey_list_arr_[num_events_] = ey_;
                                    num_events_++;
                                    // ey_list_.push_back(ey_);
                                    // ex_list_.push_back(e->x);
                                }
                            }
                        }
                        currentBaseX_ += 12;
                        break;
                    }
                    case Code::EXT_TRIGGER:
                    {
                        break;
                    }
                    case Code::OTHERS:
                    {
#if 0
                            const Others * e = reinterpret_cast<const Others *>(&buffer[i]);
                            const SubType subtype = static_cast<SubType>(e->subtype);
                            if (subtype != SubType::MASTER_END_OF_FRAME) {
                                std::cout << "ignoring OTHERS code: " << toString(subtype) << std::endl;
                            }
#endif
                    }
                    break;
                    // ------- the CONTINUED codes are used in conjunction with
                    // the OTHERS code, so ignore as well
                    case Code::CONTINUED_4:
                    case Code::CONTINUED_12:
                    {
                    }
                    break;
                    default:
                        // ------- all the vector codes are not generated
                        // by the Gen3 sensor I have....
                        std::cout << "evt3 event camera decoder got unsupported code: "
                                  << static_cast<int>(buffer[i].code) << std::endl;
                        throw std::runtime_error("got unsupported code!");
                        break;
                    }
                }
            }
            void warmup(){
                for(int i=0;i<1000;i++){
                    at::Tensor key_torch = torch::from_blob(ey_list_arr_, i, at::TensorOptions().dtype(torch::kInt32).device(device_));
                }
            }
            void accumulate_events_omp()
            {
                // if(!cudaAllocMapped(ex_list_.data(),ex_list_.size())){
                //     std::cerr <<"mapping failed x"<<std::endl;
                // }
                // if(!cudaAllocMapped(ey_list_.data(),ey_list_.size())){
                //     std::cerr <<"mapping failed y"<<std::endl;
                // }

                // c10::Device dev=at::cuda::getDeviceFromPtr(ex_list_arr_);
                // std::cout<<"test 2 CUDA Mem allocated "<< dev<<std::endl;
                try
                {
                    float duration, duration_accum, duration_nonzero, duration_index, duration_rescale;
                    float duration_accum_blob, duration_accum_index, duration_accum_creation;
                    float duration_probe1, duration_probe2, duration_probe3;
                    cudaEvent_t start;
                    cudaEvent_t stop_accumulation, stop_accumulation_blob, stop_accumulation_index, stop_accumulation_creation;
                    cudaEvent_t stop_nonzero, stop_index, stop;
                    cudaEvent_t probe1, probe2, probe3, probe4;

                    cudaEventCreate(&start);
                    cudaEventCreate(&stop_accumulation);
                    cudaEventCreate(&stop_accumulation_blob);
                    cudaEventCreate(&stop_accumulation_index);
                    cudaEventCreate(&stop_accumulation_creation);
                    cudaEventCreate(&stop_nonzero);
                    cudaEventCreate(&stop_index);
                    cudaEventCreate(&stop);
                    cudaEventCreate(&probe1);
                    cudaEventCreate(&probe2);
                    cudaEventCreate(&probe3);
                    cudaEventCreate(&probe4);
                    // printf("test\n");
                    // auto start = std::chrono::high_resolution_clock::now();
                    cudaEventRecord(start);
                    // printf("test\n");

                    // --------------------OMP accum--------------------
                    //                 int16_t a_local[1280 * 720] = {0};
                    // #pragma omp parallel num_threads(4) // try to aim for highest frame rate
                    //                 {
                    // #pragma omp for schedule(static) reduction(+ : a_local)
                    //                     for (int j = 0; j < ex_list_.size(); j++)
                    //                     {
                    //                         uint32_t key = ey_list_[j] * 1280 + ex_list_[j];
                    //                         // If event is not one of the hot pixels, push back
                    //                         if (!std::binary_search(hot_pixels_.begin(), hot_pixels_.end(), key))
                    //                         {
                    //                             a_local[key]++;
                    //                         }
                    //                     }
                    //                 } // #pragma omp parallel num_threads(4)
                    //                 at::Tensor kronecker_img = torch::from_blob(a_local, 1280 * 720, at::TensorOptions().dtype(torch::kInt16)).to(device_);

                    // --------------------Torch accum--------------------
                    // at::Tensor x_torch = torch::from_blob(ex_list_arr_, num_events_, at::TensorOptions().dtype(torch::kInt32).device(device_));
                    // at::Tensor y_torch = torch::from_blob(ey_list_arr_, num_events_, at::TensorOptions().dtype(torch::kInt32).device(device_));
                    get_key(num_events_,ex_list_arr_,ey_list_arr_);
                    cudaEventRecord(stop_accumulation_blob);
                    cudaEventSynchronize(stop_accumulation_blob);
                    // cudaDeviceSynchronize();
                    // at::Tensor key_torch = (x_torch + y_torch * width_);
                    at::Tensor key_torch = torch::from_blob(ey_list_arr_, num_events_, at::TensorOptions().dtype(torch::kInt32).device(device_));

                    cudaEventRecord(stop_accumulation_index);
                    cudaEventSynchronize(stop_accumulation_index);
                    at::Tensor vals_torch = torch::ones({key_torch.size(0)}, at::TensorOptions().dtype(torch::kInt32).device(device_));
                    at::Tensor kronecker_img = torch::zeros({height_ * width_}, at::TensorOptions().dtype(torch::kInt32).device(device_));
                    cudaEventRecord(stop_accumulation_creation);
                    cudaEventSynchronize(stop_accumulation_creation);
                    kronecker_img.index_add_(0, key_torch, vals_torch);
                    if (hot_pixels_.numel()>0)
                        kronecker_img.index_fill_(0,hot_pixels_,0);
                    cudaEventRecord(stop_accumulation);
                    cudaEventSynchronize(stop_accumulation);
                    // at::Tensor nonzero_mask = (kronecker_img != 0).to(torch::kBool);
                    cudaEventRecord(stop_nonzero);
                    cudaEventSynchronize(stop_nonzero);
                    // at::Tensor nonzero_voxel = kronecker_img.masked_select(nonzero_mask);
                    cudaEventRecord(stop_index);
                    cudaEventSynchronize(stop_index);
                    // auto minmax =nonzero_voxel.aminmax();
                    // int16_t event_min = nonzero_voxel.min().item<int32_t>();
                    // event_min = std::max(0, event_min - 1);

                    // float dividend = nonzero_voxel.to(torch::kFloat).quantile(0.9).item<float>() - event_min;
                    float dividend = kronecker_img.max().item<float>();
                    // float dividend = nonzero_voxel.max().item<float>();
                    float scale = 255 / std::max((float)1.0, dividend);

                    cudaEventRecord(probe1);
                    cudaEventSynchronize(probe1);
                    // kronecker_img = (kronecker_img * scale)-event_min;
                    // kronecker_img = (kronecker_img * scale);
                    rescale(kronecker_img.data_ptr<int>(),mat_data_,scale);
                    cudaEventRecord(probe2);
                    cudaEventSynchronize(probe2);
                    // kronecker_img = kronecker_img.to(torch::kUInt8);
                    cudaEventRecord(probe3);
                    cudaEventSynchronize(probe3);
                    // kronecker_img = kronecker_img.to(torch::kCPU);
                    cudaEventRecord(probe4);
                    cudaEventSynchronize(probe4);
                    // cv::Mat mat = cv::Mat(height_, width_, CV_8U, kronecker_img.data_ptr<uchar>());
                    cv::Mat mat = cv::Mat(height_, width_, CV_8U,mat_data_);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    // duration

                    cudaEventElapsedTime(&duration, start, stop);
                    cudaEventElapsedTime(&duration_accum_blob, start, stop_accumulation_blob);
                    cudaEventElapsedTime(&duration_accum_index, stop_accumulation_blob, stop_accumulation_index);
                    cudaEventElapsedTime(&duration_accum_creation, stop_accumulation_index, stop_accumulation_creation);
                    cudaEventElapsedTime(&duration_accum, stop_accumulation_creation, stop_accumulation);
                    cudaEventElapsedTime(&duration_nonzero, stop_accumulation, stop_nonzero);
                    cudaEventElapsedTime(&duration_index, stop_nonzero, stop_index);
                    cudaEventElapsedTime(&duration_rescale, stop_index, stop);
                    cudaEventElapsedTime(&duration_probe1, probe1, probe2);
                    cudaEventElapsedTime(&duration_probe2, probe2, probe3);
                    cudaEventElapsedTime(&duration_probe3, probe3, probe4);

                    cudaEventDestroy(start);
                    cudaEventDestroy(stop_accumulation_blob);
                    cudaEventDestroy(stop_accumulation_index);
                    cudaEventDestroy(stop_accumulation_creation);
                    cudaEventDestroy(stop_accumulation);
                    cudaEventDestroy(stop_nonzero);
                    cudaEventDestroy(stop_index);
                    cudaEventDestroy(stop);
                    cudaEventDestroy(probe1);
                    cudaEventDestroy(probe2);
                    cudaEventDestroy(probe3);
                    cudaEventDestroy(probe4);
                    recent_time_ += duration;
                    recent_time_accum_ += duration_accum;
                    recent_time_accum_blob_ += duration_accum_blob;
                    recent_time_accum_index_ += duration_accum_index;
                    recent_time_accum_creation_ += duration_accum_creation;
                    recent_time_nonzero_ += duration_nonzero;
                    recent_time_index_ += duration_index;
                    recent_time_rescale_ += duration_rescale;
                    recent_probe1_ += duration_probe1;
                    recent_probe2_ += duration_probe2;
                    recent_probe3_ += duration_probe3;

                    if (img_idx_ % 100 == 99)
                    {
                        std::cout << (int)(img_idx_ / 100) << std::endl;
                        std::cout << "avg time taken: " << recent_time_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken accum blob: " << recent_time_accum_blob_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken accum index: " << recent_time_accum_index_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken accum creation: " << recent_time_accum_creation_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken accum: " << recent_time_accum_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken nonzero: " << recent_time_nonzero_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken index: " << recent_time_index_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken rescale: " << recent_time_rescale_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken probe1: " << recent_probe1_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken probe2: " << recent_probe2_ / 100 << " milliseconds" << std::endl;
                        std::cout << "avg time taken probe3: " << recent_probe3_ / 100 << " milliseconds" << std::endl;

                        total_time_ += recent_time_;
                        total_time_accum_blob_ += recent_time_accum_blob_;
                        total_time_accum_index_ += recent_time_accum_index_;
                        total_time_accum_creation_ += recent_time_accum_creation_;
                        total_time_accum_ += recent_time_accum_;
                        total_time_nonzero_ += recent_time_nonzero_;
                        total_time_index_ += recent_time_index_;
                        total_time_rescale_ += recent_time_rescale_;
                        recent_time_ = 0;
                        recent_time_accum_blob_ = 0;
                        recent_time_accum_index_ = 0;
                        recent_time_accum_creation_ = 0;
                        recent_time_accum_ = 0;
                        recent_time_nonzero_ = 0;
                        recent_time_index_ = 0;
                        recent_time_rescale_ = 0;
                        recent_probe1_ = 0;
                        recent_probe2_ = 0;
                        recent_probe3_ = 0;
                    }

                    std::string img_name = (boost::format("output/frame_%010d.png") % img_idx_).str();
                    cv::imwrite(img_name, mat);
                    img_idx_++;
                    // printf("test12\n");
                }
                catch (c10::Error e)
                {
                    // Block of code to handle errors
                    std::cout << e.what();
                    throw e;
                }
                // catch (...) {
                //     // Block of code to handle errors
                //     std::cout << "unhandled exception";
                // }
            }
            void print_results()
            {

                total_time_ += recent_time_ / ((img_idx_ % 100) + 1);
                total_time_accum_blob_ += recent_time_accum_blob_ / ((img_idx_ % 100) + 1);
                total_time_accum_index_ += recent_time_accum_index_ / ((img_idx_ % 100) + 1);
                total_time_accum_creation_ += recent_time_accum_creation_ / ((img_idx_ % 100) + 1);
                total_time_accum_ += recent_time_accum_ / ((img_idx_ % 100) + 1);
                total_time_nonzero_ += recent_time_nonzero_ / ((img_idx_ % 100) + 1);
                total_time_index_ += recent_time_index_ / ((img_idx_ % 100) + 1);
                total_time_rescale_ += recent_time_rescale_ / ((img_idx_ % 100) + 1);
                std::cout << "total avg time taken: " << total_time_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken accum blob: " << total_time_accum_blob_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken accum index: " << total_time_accum_index_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken accum creation: " << total_time_accum_creation_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken accum: " << total_time_accum_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken nonzero: " << total_time_nonzero_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken index: " << total_time_index_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
                std::cout << "total avg time taken rescale: " << total_time_rescale_ / (img_idx_ + 1 / 100) << " milliseconds" << std::endl;
            }
            // --------------------- variables
            uint16_t ey_{0};             // current y coordinate
            uint16_t timeLow_{0};        // time stamp low
            timestamp_t timeHigh_{0};    // time stamp high + rollover bits
            uint8_t currentPolarity_{0}; // polarity for vector event
            uint16_t currentBaseX_{0};   // X coordinate basis for vector event
            bool hasValidTime_{false};   // false until time is valid
            uint16_t width_{1280};       // sensor geometry
            uint16_t height_{720};       // sensor geometry

            timestamp_t timeLimit_{0};

            torch::DeviceType device_ = torch::kCUDA;

            size_t num_events_{0};
            int32_t *ex_list_arr_ = NULL;
            int32_t *ey_list_arr_ = NULL;
            uint8_t *mat_data_ = NULL;

            at::Tensor hot_pixels_;
            int img_idx_{0};
            float recent_time_{0};
            float recent_time_accum_{0};
            float recent_time_accum_blob_{0};
            float recent_time_accum_index_{0};
            float recent_time_accum_creation_{0};
            float recent_time_nonzero_{0};
            float recent_time_index_{0};
            float recent_time_rescale_{0};
            float recent_probe1_{0};
            float recent_probe2_{0};
            float recent_probe3_{0};
            float total_time_{0};
            float total_time_accum_{0};
            float total_time_accum_blob_{0};
            float total_time_accum_index_{0};
            float total_time_accum_creation_{0};
            float total_time_nonzero_{0};
            float total_time_index_{0};
            float total_time_rescale_{0};
        };

    } // namespace evt3
} // namespace event_camera_codecs
// main loop
int main(int argc, char *argv[])
{
    Metavision::Camera cam; // create the camera

    std::string hot_pixels_file = "";
    if (argc >= 3)
    {
        hot_pixels_file = argv[2];
    }
    event_camera_codecs::evt3::EventAnalyzer event_analyzer(hot_pixels_file); // create the event analyzer
    event_analyzer.warmup();

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

    // to analyze the events, we add a callback that will be called periodically to give access to the latest events
    // cam.cd().add_callback([&event_analyzer](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
    //     event_analyzer.decode_packet(ev_begin, ev_end);
    // });

    cam.raw_data().add_callback([&event_analyzer](const uint8_t *ev_begin, const uint32_t num_bytes)
                                { event_analyzer.decode_packet(ev_begin, num_bytes); });
    // start the camera
    cam.start();

    std::cout << "Camera started!" << std::endl;
    // keep running while the camera is on or the recording is not finished
    while (cam.is_running())
    {
        // std::cout << "Camera is running!" << std::endl;
    }

    // the recording is finished, stop the camera.
    // Note: we will never get here with a live camera
    cam.stop();
    event_analyzer.print_results();
}