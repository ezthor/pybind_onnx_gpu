#include "goldwire_evaluator.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace GoldWireSeg {

Evaluator::Evaluator(const std::string& model_path, int gpu_id, float gpu_mem_gb) 
    : model_path_(model_path) {
    try {
        // 导入sys模块并修改sys.path
        py::module sys = py::module::import("sys");
        sys.attr("path").cast<py::list>().append("../python");

        // 导入模块并指定GPU配置
        py::module model_module = py::module::import("onnx_model");
        py::object model_class = model_module.attr("YOLO_ONNX");
        // 传入GPU ID和显存限制
        model_instance_ = model_class(model_path, gpu_id, gpu_mem_gb);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Evaluator: " << e.what() << std::endl;
        throw;
    }
}

Evaluator::~Evaluator() {}

Status Evaluator::evaluateSingle(const cv::Mat& bmp_image, 
                                const cv::Mat& tiff_image,
                                cv::Mat& mask) {
    try {
        if (bmp_image.empty()) {
            return Status::ERR_INPUT;
        }

        if (!bmp_image.isContinuous()) {
            return Status::ERR_INPUT;
        }

        if (bmp_image.channels() != 3) {
            return Status::ERR_INPUT;
        }

        // 转换图像为numpy数组
        py::array_t<uint8_t> array({bmp_image.rows, bmp_image.cols, bmp_image.channels()}, 
                                 bmp_image.data);

        // 调用Python模型进行预测
        py::object py_result = model_instance_.attr("predict")(array);

        // 转换结果为cv::Mat
        if (py::isinstance<py::array>(py_result)) {
            py::array py_array = py_result.cast<py::array>();
            if (py_array.ndim() != 2) {
                return Status::ERR_LOGIC;
            }

            const ssize_t* shape = py_array.shape();
            mask = cv::Mat(shape[0], shape[1], CV_8UC1, py_array.mutable_data());
            mask = mask.clone();
            return Status::SUCCESS;
        }

        return Status::ERR_LOGIC;
    } catch (const py::error_already_set &e) {
        std::cerr << "Error in evaluateSingle: " << e.what() << std::endl;
        return Status::ERR_SYSTEM;
    } catch (const std::exception &e) {
        std::cerr << "Error in evaluateSingle: " << e.what() << std::endl;
        return Status::ERR_UNKNOWN;
    }
}

Status Evaluator::evaluateBatch(const std::vector<cv::Mat>& bmp_images,
                               const std::vector<cv::Mat>& tiff_images,
                               std::vector<cv::Mat>& masks) {
    try {
        if (bmp_images.empty()) {
            return Status::ERR_INPUT;
        }

        masks.clear();

        // 将所有图像转换为numpy数组列表
        py::list image_list;
        for (const auto& bmp_image : bmp_images) {
            if (bmp_image.empty() || !bmp_image.isContinuous() || bmp_image.channels() != 3) {
                return Status::ERR_INPUT;
            }

            py::array_t<uint8_t> array({bmp_image.rows, bmp_image.cols, bmp_image.channels()}, 
                                     bmp_image.data);
            image_list.append(array);
        }

        // 调用Python的批处理方法
        py::object py_result = model_instance_.attr("predict_batch")(image_list);

        // 检查返回结果
        if (!py::isinstance<py::list>(py_result)) {
            return Status::ERR_LOGIC;
        }

        // 转换结果
        py::list result_list = py_result.cast<py::list>();
        for (size_t i = 0; i < result_list.size(); ++i) {
            py::array py_array = result_list[i].cast<py::array>();
            if (py_array.ndim() != 2) {
                return Status::ERR_LOGIC;
            }

            const ssize_t* shape = py_array.shape();
            cv::Mat mask(shape[0], shape[1], CV_8UC1, py_array.mutable_data());
            masks.push_back(mask.clone());
        }

        return Status::SUCCESS;
    } catch (const py::error_already_set &e) {
        std::cerr << "Error in evaluateBatch: " << e.what() << std::endl;
        return Status::ERR_SYSTEM;
    } catch (const std::exception &e) {
        std::cerr << "Error in evaluateBatch: " << e.what() << std::endl;
        return Status::ERR_UNKNOWN;
    }
}

} 