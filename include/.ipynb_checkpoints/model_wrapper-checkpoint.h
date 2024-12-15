#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp> // 添加 OpenCV 头文件
#include<pybind11/numpy.h>

namespace py = pybind11;

py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat & input);

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat & input);

class ModelWrapper {
public:
    ModelWrapper(const std::string& model_path = "./best.pt");
    ~ModelWrapper();

    cv::Mat predict(const cv::Mat& img); // 修改参数类型
    

private:
    py::object pInstance; // Python类实例
};