# CMakeList.txt: проект CMake для CMM-Speech-Recognition; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#
cmake_minimum_required (VERSION 3.12)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()




project ("HMM-Speech-Recognition" VERSION 1.0)


#Install CXXOPTS
include(FetchContent)
set(CXXOPTS_VERSION 2.2.0)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v${CXXOPTS_VERSION}
)
FetchContent_MakeAvailable(cxxopts)






#Install JSON
set(JSON_VERSION 3.11.2)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v${JSON_VERSION}
)
FetchContent_MakeAvailable(nlohmann_json)




file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(HMM-Speech-Recognition main.cpp ${SRC_FILES})
target_include_directories(HMM-Speech-Recognition PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(HMM-Speech-Recognition PRIVATE cxxopts nlohmann_json::nlohmann_json)



