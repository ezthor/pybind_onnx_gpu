一种在C++中通过pybind11调用Python模块的方法

此处实现的测试功能为调用python中的onnxruntime-gpu模块，实现金线分割项目

实际需求为使用torch模块运行点云补全模型，亦测试通过

# GoldWire Segmentation Project

金线分割项目测试说明文档

## 环境配置

### CUDA & Cudnn 测试可运行版本
| CUDA Version | Python Version | Cudnn Version | GPU      |
| ------------ | -------------- | ------------- | -------- |
| CUDA 11.8    | 3.8            | 8.x           | RTX 3080 |




### Python 依赖安装

```bash

pip install onnx==1.17.0
pip install onnxruntime-gpu==1.19.0
# torch没有版本需求，仅前后处理模块需求
pip install torch
pip install opencv-python
```

### 编译项目

```bash
cd Goldwire_Segmentation
rm -rf build  
mkdir build && cd build
cmake ..
make -j
```

### 运行项目

```bash
./GoldWireSegmentation
```



## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

This means:
- ✅ You can freely use this code for non-commercial purposes
- ✅ You can modify and share this code
- ❌ You cannot use this code for commercial purposes
- ❌ You cannot sublicense or sell this code

