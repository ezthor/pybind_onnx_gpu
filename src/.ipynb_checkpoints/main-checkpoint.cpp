#include "model_wrapper.h"
#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

namespace py = pybind11;

int main(int argc, char* argv[]) {
    std::string model_path = "../weight/yolov8l-seg-640-origintype-3000.onnx";
    std::string image_path = "../data/test.bmp";

    if (argc > 1) {
        image_path = argv[1];
    }

    // 初始化Python解释器
    py::scoped_interpreter guard{};

    // 安装Ultralytics库
    // py::module os = py::module::import("os");
    // os.attr("system")("pip install ultralytics");

    // 读取图像
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    // 初始化模型
    ModelWrapper model(model_path);



    // 测试推理100次的时间
    int count = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (count < 100) {
        count++;
        // 创建一个img的深拷贝
        // cv::Mat img_copy = img.clone();
        cv::Mat result_mask = model.predict(img); // 传递py::object类型的图像
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Total inference time for 100 runs: " << elapsed_seconds.count() << "s\n";
    std::cout << "Average time per inference: " << (elapsed_seconds.count() / 100) << "s\n";
    std::cout << "please check result at /build/output.txt"<<std::endl; 

    // 输出最后一次推理的结果
    cv::Mat mask = model.predict(img);
    cv::imwrite("mask_image_main.bmp", mask);



    return 0;
}