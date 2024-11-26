#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <omp.h>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

// Uncomment this line if compiling on a system with CUDA support
// #define USE_CUDA

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

bool is_integer(const std::string& s)
{
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+')
    {
        if (s.size() == 1) return false; // String is only '+' or '-'
        i = 1;
    }
    for (; i < s.size(); ++i)
    {
        if (!std::isdigit(static_cast<unsigned char>(s[i])))
        {
            return false;
        }
    }
    return true;
}

// Function to get the frame count of a video
int get_frame_count(const std::string& video_path)
{
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
    {
        std::cerr << "Error: Unable to open video file " << video_path << std::endl;
        return 0;
    }
    int frame_count = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    capture.release();
    return frame_count;
}

// Mutex for synchronized console output
std::mutex cout_mutex;

// SafeQueue class for thread-safe communication
template <typename T>
class SafeQueue
{
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool finished_ = false;

public:
    void enqueue(const T& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        cv_.notify_one();
    }

    bool dequeue(T& value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty() || finished_; });

        if (queue_.empty())
            return false;

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void set_finished()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();
    }
};

// Load class names from a file
std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("classes1.txt");
    std::string line;
    while (getline(ifs, line))
    {
        if (!line.empty())
        {
            class_list.push_back(line);
        }
    }
    return class_list;
}

// Load YOLO model
void load_net(cv::dnn::Net& net, bool is_cuda)
{
    auto model = cv::dnn::readNet("yolov5s.onnx");
#ifdef USE_CUDA
    if (is_cuda)
    {
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Using CUDA\n";
        }
    }
    else
#endif
    {
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Using CPU\n";
        }
    }
    net = model;
}

// Perform detection on a single frame
void detect(cv::Mat& frame, cv::dnn::Net& net, const std::vector<std::string>& class_list)
{
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float CONFIDENCE_THRESHOLD = 0.4;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = frame.cols / INPUT_WIDTH;
    float y_factor = frame.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    const int rows = 25200;  // Use the hardcoded value as in the original code

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* scores = data + 5;
            cv::Mat scores_mat(1, class_list.size(), CV_32FC1, scores);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores_mat, 0, &max_class_score, 0, &class_id_point);

            if (max_class_score > SCORE_THRESHOLD)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id_point.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((x - 0.5 * w) * x_factor);
                int top = static_cast<int>((y - 0.5 * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += dimensions;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255)
    };

    for (int idx : indices)
    {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        float confidence = confidences[idx];
        const cv::Scalar& color = colors[class_id % colors.size()];

        cv::rectangle(frame, box, color, 3);

        std::ostringstream label_stream;
        label_stream << class_list[class_id] << " " << std::fixed << std::setprecision(1)
                     << confidence * 100 << "%";
        std::string label = label_stream.str();

        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        cv::Point text_origin(box.x, box.y - text_size.height - 5);

        if (text_origin.y < 0) text_origin.y = 0;

        cv::rectangle(frame,
                      text_origin + cv::Point(0, baseline),
                      text_origin + cv::Point(text_size.width, -text_size.height),
                      color, cv::FILLED);
        cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    }
}

// Forward declaration of process_video_multithreaded
void process_video_multithreaded(const std::string& video_path, bool is_cuda,
                                 const std::vector<std::string>& class_list,
                                 int num_threads,
                                 SafeQueue<std::pair<std::string, cv::Mat>>& display_queue);

// Process multiple videos with threads allocated proportionally to their lengths
void process_videos(const std::vector<std::string>& video_paths, bool is_cuda,
                    const std::vector<std::string>& class_list, int total_threads,
                    SafeQueue<std::pair<std::string, cv::Mat>>& display_queue)
{
    int num_videos = video_paths.size();
    std::vector<int> frame_counts(num_videos);
    int total_frames = 0;

    // Get frame counts for each video
    for (int i = 0; i < num_videos; ++i)
    {
        frame_counts[i] = get_frame_count(video_paths[i]);
        total_frames += frame_counts[i];
    }

    // Allocate threads proportionally
    std::vector<int> threads_per_video(num_videos, 1); // Initialize with 1 thread each

    int threads_allocated = 0;
    for (int i = 0; i < num_videos; ++i)
    {
        if (total_frames > 0)
        {
            threads_per_video[i] = std::max(1, static_cast<int>(
                static_cast<double>(frame_counts[i]) / total_frames * total_threads));
        }
        else
        {
            threads_per_video[i] = 1;
        }
        threads_allocated += threads_per_video[i];
    }

    // Adjust if allocated threads exceed total_threads
    if (threads_allocated > total_threads)
    {
        double scaling_factor = static_cast<double>(total_threads) / threads_allocated;
        threads_allocated = 0;
        for (int i = 0; i < num_videos; ++i)
        {
            threads_per_video[i] = std::max(1, static_cast<int>(threads_per_video[i] * scaling_factor));
            threads_allocated += threads_per_video[i];
        }
    }

    // Distribute any remaining threads
    while (threads_allocated < total_threads)
    {
        int max_frames_index = std::distance(frame_counts.begin(),
                                             std::max_element(frame_counts.begin(), frame_counts.end()));
        threads_per_video[max_frames_index]++;
        threads_allocated++;
    }

    // Print thread allocation
    for (int i = 0; i < num_videos; ++i)
    {
        std::cout << "Video " << video_paths[i] << " allocated " << threads_per_video[i] << " threads.\n";
    }

    // Set the number of threads for the outer parallel region
    int threads_for_outer = std::min(num_videos, total_threads);

    omp_set_num_threads(threads_for_outer);

    // Launch parallel video processing
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_videos; ++i)
    {
        const auto& video_path = video_paths[i];
        int num_threads = threads_per_video[i];
        process_video_multithreaded(video_path, is_cuda, class_list, num_threads, display_queue);
    }
}

// Process a single video using multiple threads (frame-level parallelism)
void process_video_multithreaded(const std::string& video_path, bool is_cuda,
                                 const std::vector<std::string>& class_list,
                                 int num_threads,
                                 SafeQueue<std::pair<std::string, cv::Mat>>& display_queue)
{
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error: Unable to open video file " << video_path << std::endl;
        return;
    }

    // Read frames from the video
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (capture.read(frame))
    {
        frames.push_back(frame.clone());
    }
    int total_frames = frames.size();

    std::vector<cv::Mat> processed_frames(total_frames);

    // Start timing measurement inside the function
    auto start_processing = std::chrono::high_resolution_clock::now();

    // Parallel region
    #pragma omp parallel num_threads(num_threads)
    {
        // Each thread has its own net_local
        cv::dnn::Net net_local;
        auto model_load_start = std::chrono::high_resolution_clock::now();
        load_net(net_local, is_cuda);
        auto model_load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> model_load_time = model_load_end - model_load_start;

        // Print model loading time per thread
        #pragma omp critical
        {
            std::cout << "Video " << video_path << " - Thread " << omp_get_thread_num()
                      << " loaded the model in " << model_load_time.count() << " seconds.\n";
        }

        // Parallel for loop
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total_frames; ++i)
        {
            cv::Mat processed_frame = frames[i].clone();

            // Detection on the frame
            detect(processed_frame, net_local, class_list);

            // Store the processed frame
            processed_frames[i] = processed_frame;
        }
    }

    auto end_processing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> processing_duration = end_processing - start_processing;

    // Print processing time including model loading
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Video " << video_path << " processed in: "
                  << processing_duration.count() << " seconds.\n";
    }

    // Enqueue the processed frames for display
    for (const auto& processed_frame : processed_frames)
    {
        display_queue.enqueue({video_path, processed_frame});
    }

    // Enqueue an empty frame to signal video end
    display_queue.enqueue({video_path, cv::Mat()});
    capture.release();
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage:\n";
        std::cerr << argv[0] << " <video1> [video2 ...] [total_threads] [cuda]\n";
        return 1;
    }

    bool is_cuda = false;
    std::vector<std::string> video_paths;
    int total_threads = 1; // Default to 1 thread

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "cuda")
        {
            is_cuda = true;
        }
        else if (is_integer(arg))
        {
            total_threads = std::stoi(arg);
        }
        else
        {
            video_paths.push_back(arg);
        }
    }

    if (video_paths.empty())
    {
        std::cerr << "No video files specified.\n";
        return 1;
    }

    // Limit total_threads to the maximum available
    omp_set_max_active_levels(2);

    // Limit total_threads to the maximum available
    int max_threads = omp_get_max_threads();
    if (total_threads > max_threads)
    {
        total_threads = max_threads;
        std::cout << "Adjusted total threads to maximum available: " << max_threads << "\n";
    }

    std::cout << "Maximum number of threads available: " << max_threads << "\n";

    // Set the total number of threads
    omp_set_num_threads(total_threads);
    std::vector<std::string> class_list = load_class_list();
    SafeQueue<std::pair<std::string, cv::Mat>> display_queue;

    auto start_total = std::chrono::high_resolution_clock::now();

    // Process multiple videos using threads allocated proportionally
    process_videos(video_paths, is_cuda, class_list, total_threads, display_queue);

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_execution_duration = end_total - start_total;

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Total execution time: " << total_execution_duration.count() << " seconds.\n";
    }

    // Display processed frames sequentially
    std::map<std::string, std::string> windows;
    for (const auto& video_path : video_paths)
    {
        windows[video_path] = "Processed Video: " + video_path;
        cv::namedWindow(windows[video_path], cv::WINDOW_NORMAL);
    }

    std::pair<std::string, cv::Mat> display_data;
    while (true)
    {
        if (display_queue.dequeue(display_data))
        {
            const auto& video_path = display_data.first;
            const auto& frame = display_data.second;

            if (frame.empty())
            {
                cv::destroyWindow(windows[video_path]);
                windows.erase(video_path);
                if (windows.empty())
                    break;
                continue;
            }

            cv::imshow(windows[video_path], frame);
            if (cv::waitKey(1) == 27) // ESC to exit
                break;
        }
    }

    return 0;
}