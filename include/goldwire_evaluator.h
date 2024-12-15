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
    Evaluator(const std::string& model_path);
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