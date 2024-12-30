cmake_minimum_required(VERSION 3.12)
project(GoldWireSegmentation)

set(CMAKE_CXX_STANDARD 11)

# 在Windows下设置Python路径
if(WIN32)
    # 设置Python根目录
    set(Python3_ROOT_DIR "C:/Users/admin/AppData/Local/Programs/Python/Python38")
    
    # 添加包路径
    list(APPEND CMAKE_PREFIX_PATH
        "C:/Users/admin/AppData/Local/Programs/Python/Python38/DLLs"
        "C:/Users/admin/AppData/Local/Programs/Python/Python38/lib"
        "C:/Users/admin/AppData/Local/Programs/Python/Python38/lib/site-packages"
        "C:/Users/admin/AppData/Roaming/Python/Python38/site-packages"
    )
    
    # 添加site-packages路径
    set(PYTHON_SITE_PACKAGES "C:/Users/admin/AppData/Local/Programs/Python/Python38/lib/site-packages")
    message(STATUS "Using Python site-packages: ${PYTHON_SITE_PACKAGES}")
endif()

add_definitions(-DPYBIND11_SIMPLE_GIL_MANAGEMENT)

# 将pybind11放在项目的第三方目录下
add_subdirectory(third_party/pybind11)

# 查找Python安装
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找OpenCV库
set(OpenCV_DIR "D:/DBWB/code/VisionEngine/opencv460/build")
find_package(OpenCV REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    pybind11::embed 
    ${Python3_LIBRARIES} 
    ${OpenCV_LIBS}
)

# Windows下设置输出目录和Python路径
if(WIN32)
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    
    # 打印Python路径用于调试
    message(STATUS "Python3_ROOT_DIR: ${Python3_ROOT_DIR}")
    message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
endif()