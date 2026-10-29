cmake_minimum_required(VERSION 3.16)

project(MultilayerPerceptron)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ENABLE_GCOV "Enable gcov" OFF)
option(BUILD_TESTS "Build tests" OFF)

if(ENABLE_GCOV)

    add_compile_options(
        --coverage
        -fprofile-arcs
        -ftest-coverage
    )

    add_link_options(
        --coverage
    )

endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS
    Widgets
    Charts
    Concurrent
)

set(CORE_SOURCES
    src/model/matrix_model.cpp
    src/model/graph_model.cpp
)

set(APP_SOURCES

    src/main.cpp

    src/controller/controller.cpp
    src/controller/controller.h

    src/view/mainwindow.cpp
    src/view/mainwindow.h
    src/view/mainwindow.ui

    src/view/paint_widget.h
)

add_executable(
    MultilayerPerceptron

    ${APP_SOURCES}
    ${CORE_SOURCES}
)

target_include_directories(
    MultilayerPerceptron

    PRIVATE

        src
        src/model
        src/view
        src/controller
)

target_compile_options(
    MultilayerPerceptron

    PRIVATE

        -O3
        -march=native
)

target_link_libraries(
    MultilayerPerceptron

    PRIVATE

        Qt6::Widgets
        Qt6::Charts
        Qt6::Concurrent
)

set_target_properties(
    MultilayerPerceptron

    PROPERTIES

        AUTOUIC_SEARCH_PATHS
            "src/view"
)

if(BUILD_TESTS)

    find_package(GTest REQUIRED)

    add_executable(
        test_perceptron
        EXCLUDE_FROM_ALL

        src/tests.cpp

        ${CORE_SOURCES}
    )

    target_include_directories(
        test_perceptron

        PRIVATE

            src
            src/model
    )

    target_compile_options(
        test_perceptron

        PRIVATE

            -O3
            -march=native
    )

    target_link_libraries(
        test_perceptron

        PRIVATE

            GTest::gtest_main
            GTest::gtest
            pthread
    )

endif()

find_package(Doxygen)

if(DOXYGEN_FOUND)

    set(
        DOXYGEN_PROJECT_NAME
        "Multilayer Perceptron"
    )

    set(
        DOXYGEN_OUTPUT_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/docs"
    )

    set(
        DOXYGEN_GENERATE_LATEX
        NO
    )

    set(
        DOXYGEN_USE_MDFILE_AS_MAINPAGE
        "${CMAKE_CURRENT_SOURCE_DIR}/MAINPAGE.md"
    )

    doxygen_add_docs(
        dvi

        src
        REPORT.md
        MAINPAGE.md

        COMMENT
        "Generating Documentation"
    )

endif()