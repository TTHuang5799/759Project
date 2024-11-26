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
#include <cuda_runtime.h>

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

__global__ void process_frame_cuda_kernel(const uchar* input, uchar* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        // Example processing: Convert to grayscale if RGB
        if (channels == 3)
        {
            uchar r = input[idx];
            uchar g = input[idx + 1];
            uchar b = input[idx + 2];
            uchar gray = static_cast<uchar>(0.299f * r + 0.587f * g + 0.114f * b);
            output[y * width + x] = gray;
        }
    }
}

// CUDA-based frame processing function
void process_frame_cuda(const cv::Mat& input_frame, cv::Mat& output_frame)
{
    int width = input_frame.cols;
    int height = input_frame.rows;
    int channels = input_frame.channels();

    size_t frame_size = width * height * channels * sizeof(uchar);

    // Allocate device memory
    uchar *d_input, *d_output;
    cudaMalloc(&d_input, frame_size);
    cudaMalloc(&d_output, width * height * sizeof(uchar));

    // Copy input frame to device
    cudaMemcpy(d_input, input_frame.data, frame_size, cudaMemcpyHostToDevice);

    // Configure kernel
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                    (height + threads_per_block.y - 1) / threads_per_block.y);

    // Launch kernel
    process_frame_cuda_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, width, height, channels);

    // Synchronize and copy result back to host
    cudaDeviceSynchronize();
    cudaMemcpy(output_frame.data, d_output, width * height * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
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
    const int rows = 25200;

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
        const cv::Scalar& color = colors[class_id % colors.size()];

        cv::rectangle(frame, box, color, 3);

        std::ostringstream label_stream;
        label_stream << class_list[class_id] << " " << std::fixed << std::setprecision(1)
                     << confidences[idx] * 100 << "%";
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

// Process a single video
void process_video(const std::string& video_path, bool is_cuda, const std::vector<std::string>& class_list, SafeQueue<std::pair<std::string, cv::Mat>>& display_queue)
{
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error: Unable to open video file " << video_path << std::endl;
        return;
    }

    int total_frames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    if (total_frames <= 0)
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error: No frames available in video: " << video_path << std::endl;
        return;
    }

    // Measure and load YOLO model
    auto start_model_load = std::chrono::high_resolution_clock::now();
    cv::dnn::Net net;
    load_net(net, is_cuda);
    auto end_model_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> model_load_duration = end_model_load - start_model_load;

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Model loaded for video: " << video_path << " in " << model_load_duration.count() << " seconds.\n";
    }

    auto start_processing = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> frames(total_frames);
    std::vector<cv::Mat> processed_frames(total_frames);

    // Sequentially read frames
    for (int i = 0; i < total_frames; ++i)
    {
        capture.read(frames[i]);
    }

    // Parallelize frame processing
    #pragma omp parallel for
    for (int i = 0; i < total_frames; ++i)
    {
        if (!frames[i].empty())
        {
            if (is_cuda)
            {
                // Process frame using CUDA
                processed_frames[i] = cv::Mat::zeros(frames[i].size(), CV_8UC1);
                process_frame_cuda(frames[i], processed_frames[i]);
            }
            else
            {
                // Process frame using CPU (e.g., grayscale conversion as an example)
                processed_frames[i] = cv::Mat::zeros(frames[i].size(), CV_8UC1);
                cv::cvtColor(frames[i], processed_frames[i], cv::COLOR_BGR2GRAY);
            }

            // Detection step (shared for both CUDA and CPU processed frames)
            detect(processed_frames[i], net, class_list);

            // Enqueue the processed frame for display
            display_queue.enqueue({video_path, processed_frames[i]});
        }
    }

    auto end_processing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> processing_duration = end_processing - start_processing;

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Video processed in: " << processing_duration.count() << " seconds.\n";
    }

    // Enqueue an empty frame to signal video end
    display_queue.enqueue({video_path, cv::Mat()});
    capture.release();
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <video1> [video2 ...] [cuda]\n";
        return 1;
    }

    bool is_cuda = false;
    std::vector<std::string> video_paths;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "cuda")
        {
            is_cuda = true;
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

    // Dynamically set the number of threads to match the number of input videos
    int num_threads = video_paths.size();
    omp_set_num_threads(num_threads);
    std::cout << "Number of threads set to: " << num_threads << "\n";

    // Display whether CUDA or CPU is being used
    if (is_cuda)
    {
        int device_count;
        cudaGetDeviceCount(&device_count);

        if (device_count > 0)
        {
            std::cout << "CUDA-enabled GPU(s) detected: " << device_count << "\n";
            std::cout << "Using CUDA for video processing.\n";
        }
        else
        {
            std::cerr << "No CUDA-capable device detected. Falling back to CPU.\n";
            is_cuda = false;
        }
    }

    std::vector<std::string> class_list = load_class_list();
    SafeQueue<std::pair<std::string, cv::Mat>> display_queue;

    auto start_total = std::chrono::high_resolution_clock::now();

    // Launch parallel video processing
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (const auto& video_path : video_paths)
            {
                #pragma omp task
                {
                    process_video(video_path, is_cuda, class_list, display_queue);
                }
            }
        }
    }

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
