#include "goldwire_evaluator.h"
#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>
#include <iostream>
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

    try {
        // 读取图像
        cv::Mat bmp_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (bmp_image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        // 创建评估器实例
        GoldWireSeg::Evaluator evaluator(model_path);

        // 测试单张图片推理
        cv::Mat mask;
        cv::Mat dummy_tiff; // 空的tiff图像
        auto status = evaluator.evaluateSingle(bmp_image, dummy_tiff, mask);
        
        if (status == GoldWireSeg::Status::SUCCESS) {
            cv::imwrite("single_result_mask.bmp", mask);
            std::cout << "Single image inference completed successfully" << std::endl;
        } else {
            std::cerr << "Single image inference failed with status: " 
                      << static_cast<int>(status) << std::endl;
        }

        // 在main.cpp中设置批大小
        const int BATCH_SIZE = 16;  // 可以根据需要调整这个值

        // 创建批处理输入
        std::vector<cv::Mat> bmp_images(BATCH_SIZE, bmp_image);  // 使用同一张图片复制BATCH_SIZE次
        std::vector<cv::Mat> tiff_images(BATCH_SIZE);
        std::vector<cv::Mat> masks;

        // 测试批处理性能
        auto start = std::chrono::high_resolution_clock::now();
        status = evaluator.evaluateBatch(bmp_images, tiff_images, masks);
        auto end = std::chrono::high_resolution_clock::now();

        if (status == GoldWireSeg::Status::SUCCESS) {
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "Batch processing completed successfully" << std::endl;
            std::cout << "Total inference time for " << BATCH_SIZE << " runs: " 
                      << elapsed_seconds.count() << "s\n";
            std::cout << "Average time per inference: " 
                      << (elapsed_seconds.count() / BATCH_SIZE) << "s\n";
        } else {
            std::cerr << "Batch processing failed with status: " 
                      << static_cast<int>(status) << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}