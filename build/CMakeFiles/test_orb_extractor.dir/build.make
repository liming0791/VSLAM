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
CMAKE_SOURCE_DIR = /home/liming/VSLAM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liming/VSLAM/build

# Include any dependencies generated for this target.
include CMakeFiles/test_orb_extractor.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_orb_extractor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_orb_extractor.dir/flags.make

CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o: CMakeFiles/test_orb_extractor.dir/flags.make
CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o: ../test/test_orb_extractor.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liming/VSLAM/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o -c /home/liming/VSLAM/test/test_orb_extractor.cpp

CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liming/VSLAM/test/test_orb_extractor.cpp > CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.i

CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liming/VSLAM/test/test_orb_extractor.cpp -o CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.s

CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.requires:
.PHONY : CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.requires

CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.provides: CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_orb_extractor.dir/build.make CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.provides.build
.PHONY : CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.provides

CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.provides.build: CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o

# Object files for target test_orb_extractor
test_orb_extractor_OBJECTS = \
"CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o"

# External object files for target test_orb_extractor
test_orb_extractor_EXTERNAL_OBJECTS =

../bin/test_orb_extractor: CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o
../bin/test_orb_extractor: CMakeFiles/test_orb_extractor.dir/build.make
../bin/test_orb_extractor: ../lib/libVSLAM.so
../bin/test_orb_extractor: /usr/local/lib/libopencv_videostab.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_ts.a
../bin/test_orb_extractor: /usr/local/lib/libopencv_superres.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_stitching.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_contrib.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_nonfree.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_ocl.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_gpu.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_photo.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_objdetect.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_legacy.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_video.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_ml.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_calib3d.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_features2d.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_highgui.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_imgproc.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_flann.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libopencv_core.so.2.4.9
../bin/test_orb_extractor: /usr/local/lib/libpangolin.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libpython2.7.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libdc1394.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libavcodec.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libavformat.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libavutil.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libswscale.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libz.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../bin/test_orb_extractor: /usr/local/lib/libvtkIOParallelXML-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkInteractionImage-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingMath-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkGeovisCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkproj4-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOImport-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOPLY-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOParallel-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIONetCDF-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkjsoncpp-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkViewsContext2D-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingLOD-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersSMP-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOVideo-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOExodus-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkexoIIc-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingImage-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersProgrammable-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOSQL-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtksqlite-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersParallelImaging-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersParallel-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOAMR-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersAMR-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkParallelCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOEnSight-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkViewsInfovis-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkViewsCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkInteractionWidgets-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkInteractionStyle-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkInfovisLayout-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersModeling-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersImaging-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkChartsCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingStencil-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOInfovis-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOLegacy-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkInfovisCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtklibxml2-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingContextOpenGL2-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersGeneric-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersSelection-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersHyperTree-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOLSDyna-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOExport-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingAnnotation-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingColor-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingContext2D-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingLabel-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingFreeType-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkfreetype-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOMINC-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersHybrid-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkNetCDF_cxx-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkNetCDF-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkhdf5_hl-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkhdf5-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOMovie-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkoggtheora-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingMorphological-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingGeneral-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingSources-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersTexture-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingStatistics-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingVolumeOpenGL2-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingVolume-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersVerdict-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkverdict-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkDomainsChemistryOpenGL2-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingOpenGL2-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingHybrid-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOImage-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkDICOMParser-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkmetaio-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkpng-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtktiff-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkjpeg-7.0.so.1
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libXt.so
../bin/test_orb_extractor: /usr/local/lib/libvtkglew-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkDomainsChemistry-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkRenderingCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonColor-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersExtraction-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersStatistics-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingFourier-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkImagingCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkalglib-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersGeometry-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOXML-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOGeometry-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOXMLParser-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkexpat-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersFlowPaths-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersSources-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersGeneral-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonComputationalGeometry-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkFiltersCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkIOCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkzlib-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonExecutionModel-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonDataModel-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonTransforms-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonMisc-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonMath-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonSystem-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtkCommonCore-7.0.so.1
../bin/test_orb_extractor: /usr/local/lib/libvtksys-7.0.so.1
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_common.so
../bin/test_orb_extractor: /usr/local/lib/libflann_cpp_s.a
../bin/test_orb_extractor: /usr/local/lib/libpcl_kdtree.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_containers.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_utils.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_octree.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_features.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_kinfu.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_segmentation.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_octree.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_io.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_search.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_sample_consensus.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_filters.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_features.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_ml.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_segmentation.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_surface.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_registration.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_recognition.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_keypoints.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_tracking.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_visualization.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_people.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_outofcore.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_stereo.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_cuda_features.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_cuda_segmentation.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_cuda_sample_consensus.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_common.so
../bin/test_orb_extractor: /usr/local/lib/libflann_cpp_s.a
../bin/test_orb_extractor: /usr/local/lib/libpcl_kdtree.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_containers.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_utils.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_octree.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_features.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_kinfu.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_segmentation.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_octree.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_io.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_search.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_sample_consensus.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_filters.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_features.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_ml.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_segmentation.so
../bin/test_orb_extractor: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_surface.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_registration.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_recognition.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_keypoints.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_tracking.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_visualization.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_people.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_outofcore.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_stereo.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_cuda_features.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_cuda_segmentation.so
../bin/test_orb_extractor: /usr/local/lib/libpcl_cuda_sample_consensus.so
../bin/test_orb_extractor: /usr/lib/libOpenNI.so
../bin/test_orb_extractor: CMakeFiles/test_orb_extractor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/test_orb_extractor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_orb_extractor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_orb_extractor.dir/build: ../bin/test_orb_extractor
.PHONY : CMakeFiles/test_orb_extractor.dir/build

CMakeFiles/test_orb_extractor.dir/requires: CMakeFiles/test_orb_extractor.dir/test/test_orb_extractor.cpp.o.requires
.PHONY : CMakeFiles/test_orb_extractor.dir/requires

CMakeFiles/test_orb_extractor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_orb_extractor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_orb_extractor.dir/clean

CMakeFiles/test_orb_extractor.dir/depend:
	cd /home/liming/VSLAM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liming/VSLAM /home/liming/VSLAM /home/liming/VSLAM/build /home/liming/VSLAM/build /home/liming/VSLAM/build/CMakeFiles/test_orb_extractor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_orb_extractor.dir/depend

