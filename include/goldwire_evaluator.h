#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "goldwire_status.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace GoldWireSeg {

class Evaluator {
public:
    // 默认GPU ID为0，默认显存限制为4GB
    Evaluator(const std::string& model_path, int gpu_id = 0, float gpu_mem_gb = 4.0);
    ~Evaluator();

    Status evaluateSingle(const cv::Mat& bmp_image, 
                         const cv::Mat& tiff_image,
                         cv::Mat& mask);

    Status evaluateBatch(const std::vector<cv::Mat>& bmp_images,
                        const std::vector<cv::Mat>& tiff_images,
                        std::vector<cv::Mat>& masks);

private:
    std::string model_path_;
    py::object model_instance_;
};

} 