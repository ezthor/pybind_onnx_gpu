#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "goldwire_status.h"
#include <pybind11/pybind11.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

namespace py = pybind11;

namespace GoldWireSeg {

class Evaluator {
public:
    Evaluator(const std::string& model_path, int gpu_id = 0, float gpu_mem_gb = 4.0);
    ~Evaluator();

    Status evaluateSingle(const cv::Mat& bmp_image, 
                         const cv::Mat& tiff_image,
                         cv::Mat& mask);

    Status evaluateBatch(const std::vector<cv::Mat>& bmp_images,
                        const std::vector<cv::Mat>& tiff_images,
                        std::vector<cv::Mat>& masks);

    // 线程安全的推理接口
    Status threadSafeInference(const cv::Mat& image, cv::Mat& mask);

    // 用于测试的详细推理接口
    Status testInference(const cv::Mat& image, int thread_id, 
                        std::atomic<int>& successful_count,
                        std::atomic<int>& failed_count,
                        const std::atomic<bool>& stop_flag,
                        int iterations = 10);

    // GIL 管理方法
    void acquireGIL() {
        if (!gil_acquired_) {
            gil_acquire_ = std::make_unique<py::gil_scoped_acquire>();
            gil_acquired_ = true;
        }
    }

    void releaseGIL() {
        if (gil_acquired_) {
            gil_acquire_.reset();
            gil_acquired_ = false;
        }
    }

private:
    std::string model_path_;
    py::object model_instance_;
    std::mutex eval_mutex_;
    std::unique_ptr<py::gil_scoped_acquire> gil_acquire_;
    bool gil_acquired_{false};
    std::mutex gil_mutex_;  // 保护 GIL 操作的互斥锁
};

} 