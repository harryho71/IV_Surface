cmake_minimum_required(VERSION 3.15)
project(iv_surface CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================== Dependencies ====================

# Boost (numerical algorithms) - optional
find_package(Boost 1.75 QUIET)

# Eigen (linear algebra, matrix operations) - use local third_party if not found system-wide
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
	# Use bundled Eigen3
	set(Eigen3_FOUND TRUE)
	set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
	message(STATUS "Using local Eigen3 from third_party")
endif()

# NLopt (nonlinear optimization for calibration) - use bundled if not found
find_package(nlopt QUIET)
if(NOT nlopt_FOUND)
	# Use bundled NLopt
	set(nlopt_FOUND TRUE)
	set(NLOPT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlopt/include")
	set(NLOPT_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlopt/lib/libnlopt.dll.a")
	message(STATUS "Using local NLopt from third_party")
endif()

# ==================== C++ Library ====================

# Core C++ library (static)
add_library(iv_surface_static STATIC
	src/cpp/src/arbitrage.cpp
	src/cpp/src/pricing.cpp
	src/cpp/src/volatility.cpp
	src/cpp/src/interpolation.cpp
	src/cpp/src/optimization.cpp
	src/cpp/src/utils.cpp
	src/cpp/src/total_variance.cpp
)

target_include_directories(iv_surface_static
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
)

# Conditionally link libraries if found
if(Boost_FOUND)
	target_include_directories(iv_surface_static PUBLIC ${Boost_INCLUDE_DIRS})
	target_link_libraries(iv_surface_static PUBLIC Boost::boost)
endif()

if(Eigen3_FOUND)
	target_include_directories(iv_surface_static PUBLIC ${EIGEN3_INCLUDE_DIR})
	# If using local Eigen, just add include path (Eigen is header-only)
	# Otherwise link to Eigen3::Eigen if it's an imported target
	if(TARGET Eigen3::Eigen)
		target_link_libraries(iv_surface_static PUBLIC Eigen3::Eigen)
	endif()
endif()

if(nlopt_FOUND)
	target_include_directories(iv_surface_static PUBLIC ${NLOPT_INCLUDE_DIR})
	if(TARGET nlopt::nlopt_cxx)
		target_link_libraries(iv_surface_static PUBLIC nlopt::nlopt_cxx)
	else()
		# For bundled NLopt, link directly
		target_link_libraries(iv_surface_static PUBLIC ${NLOPT_LIBRARIES})
	endif()
	target_compile_definitions(iv_surface_static PUBLIC HAVE_NLOPT)
	message(STATUS "NLopt found and enabled for model calibration")
endif()

# ==================== C++ Unit Tests ====================

enable_testing()

add_executable(test_cpp_pricing
	tests/cpp/test_pricing.cpp
)

target_link_libraries(test_cpp_pricing
	PRIVATE
		iv_surface_static
)

add_test(NAME cpp_pricing COMMAND test_cpp_pricing)

add_executable(test_cpp_interpolation
	tests/cpp/test_interpolation.cpp
)

target_link_libraries(test_cpp_interpolation
	PRIVATE
		iv_surface_static
)

add_test(NAME cpp_interpolation COMMAND test_cpp_interpolation)

add_executable(test_cpp_calibration
	tests/cpp/test_calibration.cpp
)

target_link_libraries(test_cpp_calibration
	PRIVATE
		iv_surface_static
)

add_test(NAME cpp_calibration COMMAND test_cpp_calibration)

# ==================== Compiler Optimization ====================

if(MSVC)
	# Visual Studio
	target_compile_options(iv_surface_static PRIVATE /O2 /arch:AVX2 /fp:fast)
else()
	# GCC / Clang
	target_compile_options(iv_surface_static PRIVATE -O3 -march=native -ffast-math)
    
	# Parallel execution (OpenMP)
	find_package(OpenMP QUIET)
	if(OpenMP_FOUND)
		target_link_libraries(iv_surface_static PUBLIC OpenMP::OpenMP_CXX)
	endif()
endif()

# ==================== Logging ====================

message(STATUS "Build Configuration")
message(STATUS "  CMake version:    ${CMAKE_VERSION}")
message(STATUS "  C++ Compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Boost version:    ${Boost_VERSION}")
message(STATUS "  Eigen3 found:     ${Eigen3_FOUND}")
message(STATUS "  NLopt found:      ${nlopt_FOUND}")

# ==================== SABR CLI Tool ====================

add_executable(sabr_cli
        src/cpp/src/sabr_cli.cpp
)

target_link_libraries(sabr_cli
        PRIVATE
                iv_surface_static
)
