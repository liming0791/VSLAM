# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liming/MySLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liming/MySLAM/build

# Include any dependencies generated for this target.
include CMakeFiles/ORB_SLAM2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ORB_SLAM2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ORB_SLAM2.dir/flags.make

CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o: CMakeFiles/ORB_SLAM2.dir/flags.make
CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o: ../src/ORBextractor.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liming/MySLAM/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o -c /home/liming/MySLAM/src/ORBextractor.cpp

CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liming/MySLAM/src/ORBextractor.cpp > CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.i

CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liming/MySLAM/src/ORBextractor.cpp -o CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.s

CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.requires:
.PHONY : CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.requires

CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.provides: CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.requires
	$(MAKE) -f CMakeFiles/ORB_SLAM2.dir/build.make CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.provides.build
.PHONY : CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.provides

CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.provides.build: CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o

CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o: CMakeFiles/ORB_SLAM2.dir/flags.make
CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o: ../src/ORBmatcher.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liming/MySLAM/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o -c /home/liming/MySLAM/src/ORBmatcher.cpp

CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liming/MySLAM/src/ORBmatcher.cpp > CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.i

CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liming/MySLAM/src/ORBmatcher.cpp -o CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.s

CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.requires:
.PHONY : CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.requires

CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.provides: CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.requires
	$(MAKE) -f CMakeFiles/ORB_SLAM2.dir/build.make CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.provides.build
.PHONY : CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.provides

CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.provides.build: CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o

CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o: CMakeFiles/ORB_SLAM2.dir/flags.make
CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o: ../src/PnPSolver.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liming/MySLAM/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o -c /home/liming/MySLAM/src/PnPSolver.cpp

CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liming/MySLAM/src/PnPSolver.cpp > CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.i

CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liming/MySLAM/src/PnPSolver.cpp -o CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.s

CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.requires:
.PHONY : CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.requires

CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.provides: CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.requires
	$(MAKE) -f CMakeFiles/ORB_SLAM2.dir/build.make CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.provides.build
.PHONY : CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.provides

CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.provides.build: CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o

# Object files for target ORB_SLAM2
ORB_SLAM2_OBJECTS = \
"CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o" \
"CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o" \
"CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o"

# External object files for target ORB_SLAM2
ORB_SLAM2_EXTERNAL_OBJECTS =

../lib/libORB_SLAM2.so: CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o
../lib/libORB_SLAM2.so: CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o
../lib/libORB_SLAM2.so: CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o
../lib/libORB_SLAM2.so: CMakeFiles/ORB_SLAM2.dir/build.make
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_videostab.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_video.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_ts.a
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_superres.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_stitching.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_photo.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_ocl.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_objdetect.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_nonfree.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_ml.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_legacy.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_imgproc.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_highgui.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_gpu.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_flann.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_features2d.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_core.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_contrib.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_calib3d.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libpangolin.so
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_nonfree.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_ocl.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_gpu.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_photo.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_objdetect.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_legacy.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_video.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_ml.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_calib3d.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_features2d.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_highgui.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_imgproc.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_flann.so.2.4.9
../lib/libORB_SLAM2.so: /usr/local/lib/libopencv_core.so.2.4.9
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libGL.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libSM.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libICE.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libX11.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libXext.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libdc1394.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libavcodec.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libavformat.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libavutil.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libswscale.so
../lib/libORB_SLAM2.so: /usr/lib/libOpenNI.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libpng.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libz.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libtiff.so
../lib/libORB_SLAM2.so: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../lib/libORB_SLAM2.so: CMakeFiles/ORB_SLAM2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../lib/libORB_SLAM2.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ORB_SLAM2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ORB_SLAM2.dir/build: ../lib/libORB_SLAM2.so
.PHONY : CMakeFiles/ORB_SLAM2.dir/build

CMakeFiles/ORB_SLAM2.dir/requires: CMakeFiles/ORB_SLAM2.dir/src/ORBextractor.cpp.o.requires
CMakeFiles/ORB_SLAM2.dir/requires: CMakeFiles/ORB_SLAM2.dir/src/ORBmatcher.cpp.o.requires
CMakeFiles/ORB_SLAM2.dir/requires: CMakeFiles/ORB_SLAM2.dir/src/PnPSolver.cpp.o.requires
.PHONY : CMakeFiles/ORB_SLAM2.dir/requires

CMakeFiles/ORB_SLAM2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ORB_SLAM2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ORB_SLAM2.dir/clean

CMakeFiles/ORB_SLAM2.dir/depend:
	cd /home/liming/MySLAM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liming/MySLAM /home/liming/MySLAM /home/liming/MySLAM/build /home/liming/MySLAM/build /home/liming/MySLAM/build/CMakeFiles/ORB_SLAM2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ORB_SLAM2.dir/depend

