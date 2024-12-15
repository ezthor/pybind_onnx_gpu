#include "model_wrapper.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <opencv2/opencv.hpp> // 添加 OpenCV 头文件

namespace py = pybind11;

ModelWrapper::ModelWrapper(const std::string& model_path) {
    try {
        // 导入sys模块并修改sys.path
        py::module sys = py::module::import("sys");
        sys.attr("path").cast<py::list>().append("../python");

        // 导入自定义的model模块
        py::object model_module = py::module::import("onnx_model");

        // 获取MODEL类
        py::object model_class = model_module.attr("YOLO_ONNX");

        // 创建MODEL类的实例
        pInstance = model_class(model_path);

    } catch (const std::exception &e) {
        std::cerr << "Error initializing ModelWrapper: " << e.what() << std::endl;
    }
}

ModelWrapper::~ModelWrapper() {
    // 不再在此处终止解释器
}

cv::Mat ModelWrapper::predict(const cv::Mat& img) {
    std::vector<std::string> result;
    cv::Mat result_mask;

    try {
        // 将const cv::Mat& img转换为py::object
        // Check if the image is continuous
        if (!img.isContinuous()) {
            throw std::runtime_error("Only continuous Mats are supported");
        }

        // Determine the number of channels
        int channels = img.channels();

        // Create a numpy array from the cv::Mat
        py::array_t<uint8_t> array;
        if (channels == 1) {
            // Grayscale image
            array = py::array_t<uint8_t>({img.rows, img.cols}, img.data);
        } else if (channels == 3) {
            // Color image (BGR)
            array = py::array_t<uint8_t>({img.rows, img.cols, channels}, img.data);
        } else {
            throw std::runtime_error("Unsupported number of channels");
        }

        // 将py::array_t<uint8_t>转换为const py::object&
        const py::object& py_img = array;

        // 调用Python的predict方法
        py::object py_result = pInstance.attr("predict")(py_img);

        // 检查py_result是否为numpy数组
        if (py::isinstance<py::array>(py_result)) {
            py::array py_array = py_result.cast<py::array>();

            // 获取数组的维度数量和形状
            size_t ndim = py_array.ndim();
            const ssize_t* shape_ptr = py_array.shape();
            std::vector<ssize_t> shape(shape_ptr, shape_ptr + ndim);

            // 检查数组是否为2维
            if (shape.size() != 2) {
                throw std::runtime_error("Expected a 2D numpy array");
            }

            // 将numpy数组转换为cv::Mat
            cv::Mat mask(shape[0], shape[1], CV_8UC1, py_array.mutable_data());

            // 根据需要处理mask，例如保存到文件
            cv::imwrite("mask_image.bmp", mask);
            result_mask = mask;

        } else {
            std::cerr << "Error: Python predict function did not return a numpy array." << std::endl;
        }

    } catch (const py::error_already_set &e) {
        std::cerr << "Error in predict: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error in predict: " << e.what() << std::endl;
    }

    return result_mask;
}