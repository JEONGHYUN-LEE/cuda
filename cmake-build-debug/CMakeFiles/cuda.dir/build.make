# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jeonghyeonlee/Project/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jeonghyeonlee/Project/cuda/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/cuda.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda.dir/flags.make

CMakeFiles/cuda.dir/main.cpp.o: CMakeFiles/cuda.dir/flags.make
CMakeFiles/cuda.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jeonghyeonlee/Project/cuda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda.dir/main.cpp.o -c /Users/jeonghyeonlee/Project/cuda/main.cpp

CMakeFiles/cuda.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jeonghyeonlee/Project/cuda/main.cpp > CMakeFiles/cuda.dir/main.cpp.i

CMakeFiles/cuda.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jeonghyeonlee/Project/cuda/main.cpp -o CMakeFiles/cuda.dir/main.cpp.s

# Object files for target cuda
cuda_OBJECTS = \
"CMakeFiles/cuda.dir/main.cpp.o"

# External object files for target cuda
cuda_EXTERNAL_OBJECTS =

cuda: CMakeFiles/cuda.dir/main.cpp.o
cuda: CMakeFiles/cuda.dir/build.make
cuda: CMakeFiles/cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jeonghyeonlee/Project/cuda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda.dir/build: cuda

.PHONY : CMakeFiles/cuda.dir/build

CMakeFiles/cuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda.dir/clean

CMakeFiles/cuda.dir/depend:
	cd /Users/jeonghyeonlee/Project/cuda/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jeonghyeonlee/Project/cuda /Users/jeonghyeonlee/Project/cuda /Users/jeonghyeonlee/Project/cuda/cmake-build-debug /Users/jeonghyeonlee/Project/cuda/cmake-build-debug /Users/jeonghyeonlee/Project/cuda/cmake-build-debug/CMakeFiles/cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda.dir/depend

