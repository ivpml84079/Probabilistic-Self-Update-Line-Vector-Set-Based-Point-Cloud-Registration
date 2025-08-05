cmake_minimum_required(VERSION 3.10)
project(PSULVSB)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(teaserpp REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP)

include_directories(${PCL_INCLUDE_DIRS})
# Change this line to include your own executable file
add_executable(PSULVSB PSULVSB.cc)

# Link to teaserpp & Eigen3
target_link_libraries(PSULVSB Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_io OpenMP::OpenMP_CXX ${PCL_LIBRARIES})

# Copy the data files to build directory
file(COPY ../example_data/
        DESTINATION ./example_data/
        FILES_MATCHING
        PATTERN *.ply)