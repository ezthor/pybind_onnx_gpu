# GoldWire Segmentation Project

金线分割项目测试说明文档

## 环境配置

### CUDA & Cudnn 测试可运行版本
| CUDA Version | Python Version | Cudnn Version | GPU      |
| ------------ | -------------- | ------------- | -------- |
| CUDA 11.8    | 3.8            | 8.x           | RTX 3080 |




### Python 依赖安装

```bash
pip install torch==2.4.1
pip install onnx==1.17.0
pip install onnxruntime-gpu==1.19.0
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

