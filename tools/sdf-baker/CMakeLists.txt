cmake_minimum_required(VERSION 3.21)
project(ply_sdf_baker LANGUAGES CXX)

find_package(OpenVDB CONFIG REQUIRED)

add_executable(ply_sdf_baker main.cpp)
target_compile_features(ply_sdf_baker PRIVATE cxx_std_17)
target_link_libraries(ply_sdf_baker PRIVATE OpenVDB::openvdb)
