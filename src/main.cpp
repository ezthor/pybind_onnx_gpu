#include "goldwire_evaluator.h"
#include <pybind11/embed.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>

namespace py = pybind11;

// GPU配置常量
constexpr int GPU_ID = 0;          // 使用的GPU ID , TODO: 请根据实际需要修改
constexpr float GPU_MEM_GB = 4.0f; // GPU显存限制（GB） , TODO: 请根据实际需要修改 , 若出错可放宽不确定超出显存时是否出错

// 性能测试配置常量
constexpr int BATCH_SIZE = 16;      // 批处理大小
constexpr int WARMUP_ROUNDS = 10;   // 预热轮数
constexpr int TEST_ROUNDS = 100;    // 测试轮数

int main(int argc, char* argv[]) {
    try {
        // 初始化Python解释器
        py::scoped_interpreter guard{};

        // 添加Python路径
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append(r'C:/Users/admin/AppData/Local/Programs/Python/Python38/lib/site-packages')");
        
        // 添加Python模块搜索路径 , 此部分出错可注释
        py::module::import("sys").attr("path").cast<py::list>().append("..\\python");
        std::cout << "Python path: " << "..\\python" << std::endl;

        // 使用相对路径 , TODO: 请根据实际需要修改为绝对路径
        std::string model_path = "..\\weight\\yolov8l-seg-640-origintype-3000-dynamic.onnx";
        std::string image_path = "..\\data\\test.bmp";

        if (argc > 1) {
            image_path = argv[1];
        }

        // 读取图像
        cv::Mat bmp_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (bmp_image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return 1;
        }

        // 创建评估器实例，使用常量配置
        GoldWireSeg::Evaluator evaluator(model_path, GPU_ID, GPU_MEM_GB);

        // 测试单张图片推理
        cv::Mat mask;
        cv::Mat dummy_tiff;
        auto status = evaluator.evaluateSingle(bmp_image, dummy_tiff, mask);
        
        if (status == GoldWireSeg::Status::SUCCESS) {
            cv::imwrite("single_result_mask.bmp", mask);
            std::cout << "Single image inference completed successfully" << std::endl;
        } else {
            std::cerr << "Single image inference failed with status: " 
                      << static_cast<int>(status) << std::endl;
        }

        // 准备批处理数据
        std::vector<cv::Mat> bmp_images(BATCH_SIZE, bmp_image);
        std::vector<cv::Mat> tiff_images(BATCH_SIZE);
        std::vector<cv::Mat> masks;

        // 预热阶段
        std::cout << "\nWarming up with " << WARMUP_ROUNDS << " rounds..." << std::endl;
        for (int i = 0; i < WARMUP_ROUNDS; ++i) {
            status = evaluator.evaluateBatch(bmp_images, tiff_images, masks);
            if (status != GoldWireSeg::Status::SUCCESS) {
                std::cerr << "Warmup failed at round " << i << std::endl;
                return 1;
            }
        }

        // 正式测试阶段
        std::cout << "\nStarting performance test with " << TEST_ROUNDS << " rounds..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < TEST_ROUNDS; ++i) {
            status = evaluator.evaluateBatch(bmp_images, tiff_images, masks);
            if (status != GoldWireSeg::Status::SUCCESS) {
                std::cerr << "Test failed at round " << i << std::endl;
                return 1;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        
        // 输出性能统计
        double total_time = elapsed_seconds.count();
        double avg_batch_time = total_time / TEST_ROUNDS;
        double avg_image_time = avg_batch_time / BATCH_SIZE;
        
        std::cout << "\nPerformance Statistics:" << std::endl;
        std::cout << "Total test time: " << total_time << "s" << std::endl;
        std::cout << "Average time per batch: " << avg_batch_time * 1000 << "ms" << std::endl;
        std::cout << "Average time per image: " << avg_image_time * 1000 << "ms" << std::endl;
        std::cout << "Throughput: " << (BATCH_SIZE * TEST_ROUNDS) / total_time << " images/s" << std::endl;

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}