cmake_minimum_required(VERSION 3.12)
project(GoldWireSegmentation)

set(CMAKE_CXX_STANDARD 11)

# 添加pybind11子模块（或使用已安装的pybind11）
# 如果没有将pybind11作为子模块，可以使用以下方式引入
# find_package(pybind11 REQUIRED)

# 将pybind11放在项目的第三方目录下
add_subdirectory(third_party/pybind11)

# 查找Python安装
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

# 修改可执行文件名称
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接pybind11、Python和OpenCV库
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    pybind11::embed 
    ${Python3_LIBRARIES} 
    ${OpenCV_LIBS}
)