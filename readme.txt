首先在终端运行 , 需要根据本机cuda版本和cudnn版本确定，cuda<12时使用1.18.0
pip install onnxruntime-gpu==1.18.0
然后进入build文件夹（可删除后重新创建）
cmake .. && make -j
运行
./EmbedPythonExample 

python代码位于/python/onnx_model.py , 被调用的代码
cpp的包装接口位于/src/model_wrapper.cpp 
cpp的测试位于/src/main.cpp