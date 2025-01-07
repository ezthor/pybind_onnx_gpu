#include "goldwire_evaluator.h"
#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <signal.h>

namespace py = pybind11;

// GPU配置常量
constexpr int GPU_ID = 0;          // 使用的GPU ID
constexpr float GPU_MEM_GB = 4.0f; // GPU显存限制（GB）
constexpr int TEST_ITERATIONS = 10; // 每个线程的测试次数

// 用于统计线程执行次数
std::atomic<int> successful_inferences(0);
std::atomic<int> failed_inferences(0);

// 全局停止标志
std::atomic<bool> g_stop_flag(false);

// 信号处理函数
void signal_handler(int signum) {
    if (signum == SIGINT) {
        std::cout << "\nReceived Ctrl+C, stopping..." << std::endl;
        g_stop_flag = true;
    }
}

void thread_func(GoldWireSeg::Evaluator& evaluator, const cv::Mat& image, int thread_id) {
    cv::Mat mask;
    cv::Mat dummy_tiff;
    
    try {
        for (int i = 0; i < TEST_ITERATIONS && !g_stop_flag; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto status = GoldWireSeg::Status::SUCCESS;
            
            // 只在调用Python代码时获取GIL
            {
                py::gil_scoped_acquire acquire;
                status = evaluator.evaluateSingle(image, dummy_tiff, mask);
            }  // GIL 在这里自动释放
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            
            if (status == GoldWireSeg::Status::SUCCESS) {
                successful_inferences++;
                std::cout << "Thread " << thread_id << " - Iteration " << i + 1 
                         << "/" << TEST_ITERATIONS 
                         << " completed in " << elapsed.count() << "s" << std::endl;
                
                if (i == 0) {
                    cv::imwrite("thread_" + std::to_string(thread_id) + "_result.bmp", mask);
                }
            } else {
                failed_inferences++;
                std::cerr << "Thread " << thread_id << " - Iteration " << i + 1 
                         << "/" << TEST_ITERATIONS << " failed with status: " 
                         << static_cast<int>(status) << std::endl;
            }
            
            // 在没有GIL的情况下检查停止标志和休眠
            if (g_stop_flag) {
                std::cout << "Thread " << thread_id << " stopping..." << std::endl;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " error: " << e.what() << std::endl;
    }
}

// 简单的线程使用示例
void simple_thread_test(const std::string& model_path, 
                       const std::string& image_path,
                       int gpu_id = 0, 
                       float gpu_mem_gb = 4.0f) {
    GoldWireSeg::Evaluator evaluator(model_path, gpu_id, gpu_mem_gb);
    cv::Mat image = cv::imread(image_path);
    cv::Mat mask;
    
    std::thread t1([&]() {
        evaluator.threadSafeInference(image, mask);
    });
    
    t1.join();
}

// 详细的测试功能
void detailed_test(const std::string& model_path, 
                  const std::string& image_path,
                  int gpu_id = 0, 
                  float gpu_mem_gb = 4.0f) {
    std::atomic<int> successful_inferences(0);
    std::atomic<int> failed_inferences(0);
    std::atomic<bool> stop_flag(false);
    
    GoldWireSeg::Evaluator evaluator1(model_path, gpu_id, gpu_mem_gb);
    GoldWireSeg::Evaluator evaluator2(model_path, gpu_id, gpu_mem_gb);
    
    cv::Mat image = cv::imread(image_path);
    
    std::thread t1([&]() {
        evaluator1.testInference(image, 1, successful_inferences, 
                               failed_inferences, stop_flag);
    });
    
    std::thread t2([&]() {
        evaluator2.testInference(image, 2, successful_inferences, 
                               failed_inferences, stop_flag);
    });
    
    t1.join();
    t2.join();
    
    // 输出统计信息...
}

int main(int argc, char* argv[]) {
    std::string model_path = "../weight/yolov8l-seg-640-origintype-3000-dynamic.onnx";
    std::string image_path = "../data/test.bmp";

    if (argc > 1) {
        image_path = argv[1];
    }

    // 初始化Python解释器
    py::scoped_interpreter guard{};

    // 设置信号处理
    signal(SIGINT, signal_handler);

    try {
        std::cout << "Starting multi-instance inference test..." << std::endl;
        
        // 读取图像
        cv::Mat bmp_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (bmp_image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }
        std::cout << "Image loaded successfully" << std::endl;

        // 创建两个评估器实例，分别使用不同的GPU（如果有多GPU）或同一个GPU
        GoldWireSeg::Evaluator evaluator1(model_path, 0, GPU_MEM_GB);  // GPU 0
        std::cout << "Evaluator 1 initialized" << std::endl;
        
        GoldWireSeg::Evaluator evaluator2(model_path, 0, GPU_MEM_GB);  // 也使用 GPU 0
        std::cout << "Evaluator 2 initialized" << std::endl;

        // 确保在创建线程前释放GIL
        py::gil_scoped_release release;
        
        std::cout << "Starting threads with separate evaluators..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 每个一个线程
        std::thread t1(thread_func, std::ref(evaluator1), std::ref(bmp_image), 1);
        std::thread t2(thread_func, std::ref(evaluator2), std::ref(bmp_image), 2);
        
        t1.join();
        t2.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = end_time - start_time;

        // 如果是因为Ctrl+C停止的，输出相应信息
        if (g_stop_flag) {
            std::cout << "\nTest stopped by user." << std::endl;
        }

        // 输出统计信息
        std::cout << "\nTest completed!" << std::endl;
        std::cout << "Total time: " << total_time.count() << "s" << std::endl;
        std::cout << "Total successful inferences: " << successful_inferences << std::endl;
        std::cout << "Total failed inferences: " << failed_inferences << std::endl;
        std::cout << "Average time per inference: " 
                  << total_time.count() / (successful_inferences + failed_inferences) 
                  << "s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}