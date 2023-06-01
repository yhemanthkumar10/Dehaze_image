cmake_minimum_required(VERSION 3.24)
project(Dehaze50)

set(CMAKE_CXX_STANDARD 23)

add_executable(Dehaze50 train_network_functions.cpp createmodel.cpp generatedata.cpp smoothness.cpp utilities.cpp train_network.cpp)
