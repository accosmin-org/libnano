cmake_minimum_required(VERSION 3.5)
project(example VERSION 0.0.1 LANGUAGES CXX)

option(NANO_BUILD_QT5_APP "Build Qt5-based interface (if Qt5 libraries are installed)" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(info src/info.cpp)
add_executable(tensor src/tensor.cpp)
add_executable(minimize src/minimize.cpp)

find_package(NANO 1.0 REQUIRED)

target_link_libraries(info NANO::NANO)
target_link_libraries(tensor NANO::NANO)
target_link_libraries(minimize NANO::NANO)

if(NANO_BUILD_QT5_APP)
    if(APPLE)
        # todo: find a better way to properly use Qt5 installed with brew
        find_package(Qt5 5.6 COMPONENTS Core Widgets QUIET)
        if(NOT Qt5_FOUND)
            set(Qt5_DIR "/usr/local/opt/qt/lib/cmake/Qt5")
        endif()
    endif()
    find_package(Qt5 5.6 COMPONENTS Core Widgets QUIET)
    if(Qt5_FOUND)
        message(STATUS "Building the Qt5-based app")
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)

        if(CMAKE_VERSION VERSION_LESS "3.7.0")
            set(CMAKE_INCLUDE_CURRENT_DIR ON)
        endif()

        add_executable(LibNano src/qt5ui.cpp src/qt5mainwindow.cpp)

        target_include_directories(LibNano PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
        target_link_libraries(LibNano PRIVATE NANO::NANO Qt5::Widgets)
    else()
        message(WARNING "Skipping building the Qt5-based app, cannot find Qt5")
    endif()
else()
    message(STATUS "Skipping building the Qt5-based app, disabled")
endif()
