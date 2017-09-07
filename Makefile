# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/kamil/Projects/RabbitCL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/kamil/Projects/RabbitCL

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/opt/local/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/opt/local/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/opt/local/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/opt/local/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/opt/local/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/opt/local/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/opt/local/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local/fast

# Special rule for the target test
test:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running tests..."
	/opt/local/bin/ctest --force-new-ctest-process $(ARGS)
.PHONY : test

# Special rule for the target test
test/fast: test

.PHONY : test/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/opt/local/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/kamil/Projects/RabbitCL/CMakeFiles /Users/kamil/Projects/RabbitCL/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/kamil/Projects/RabbitCL/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named mozart

# Build rule for target.
mozart: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 mozart
.PHONY : mozart

# fast build rule for target.
mozart/fast:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/build
.PHONY : mozart/fast

#=============================================================================
# Target rules for targets named mozart_test

# Build rule for target.
mozart_test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 mozart_test
.PHONY : mozart_test

# fast build rule for target.
mozart_test/fast:
	$(MAKE) -f CMakeFiles/mozart_test.dir/build.make CMakeFiles/mozart_test.dir/build
.PHONY : mozart_test/fast

#=============================================================================
# Target rules for targets named googletest

# Build rule for target.
googletest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 googletest
.PHONY : googletest

# fast build rule for target.
googletest/fast:
	$(MAKE) -f CMakeFiles/googletest.dir/build.make CMakeFiles/googletest.dir/build
.PHONY : googletest/fast

#=============================================================================
# Target rules for targets named gmock

# Build rule for target.
gmock: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gmock
.PHONY : gmock

# fast build rule for target.
gmock/fast:
	$(MAKE) -f build/googletest-build/googlemock/CMakeFiles/gmock.dir/build.make build/googletest-build/googlemock/CMakeFiles/gmock.dir/build
.PHONY : gmock/fast

#=============================================================================
# Target rules for targets named gmock_main

# Build rule for target.
gmock_main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gmock_main
.PHONY : gmock_main

# fast build rule for target.
gmock_main/fast:
	$(MAKE) -f build/googletest-build/googlemock/CMakeFiles/gmock_main.dir/build.make build/googletest-build/googlemock/CMakeFiles/gmock_main.dir/build
.PHONY : gmock_main/fast

#=============================================================================
# Target rules for targets named gtest_main

# Build rule for target.
gtest_main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gtest_main
.PHONY : gtest_main

# fast build rule for target.
gtest_main/fast:
	$(MAKE) -f build/googletest-build/googlemock/gtest/CMakeFiles/gtest_main.dir/build.make build/googletest-build/googlemock/gtest/CMakeFiles/gtest_main.dir/build
.PHONY : gtest_main/fast

#=============================================================================
# Target rules for targets named gtest

# Build rule for target.
gtest: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gtest
.PHONY : gtest

# fast build rule for target.
gtest/fast:
	$(MAKE) -f build/googletest-build/googlemock/gtest/CMakeFiles/gtest.dir/build.make build/googletest-build/googlemock/gtest/CMakeFiles/gtest.dir/build
.PHONY : gtest/fast

src/activation.o: src/activation.cpp.o

.PHONY : src/activation.o

# target to build an object file
src/activation.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/activation.cpp.o
.PHONY : src/activation.cpp.o

src/activation.i: src/activation.cpp.i

.PHONY : src/activation.i

# target to preprocess a source file
src/activation.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/activation.cpp.i
.PHONY : src/activation.cpp.i

src/activation.s: src/activation.cpp.s

.PHONY : src/activation.s

# target to generate assembly for a file
src/activation.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/activation.cpp.s
.PHONY : src/activation.cpp.s

src/cost.o: src/cost.cpp.o

.PHONY : src/cost.o

# target to build an object file
src/cost.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/cost.cpp.o
.PHONY : src/cost.cpp.o

src/cost.i: src/cost.cpp.i

.PHONY : src/cost.i

# target to preprocess a source file
src/cost.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/cost.cpp.i
.PHONY : src/cost.cpp.i

src/cost.s: src/cost.cpp.s

.PHONY : src/cost.s

# target to generate assembly for a file
src/cost.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/cost.cpp.s
.PHONY : src/cost.cpp.s

src/dense.o: src/dense.cpp.o

.PHONY : src/dense.o

# target to build an object file
src/dense.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/dense.cpp.o
.PHONY : src/dense.cpp.o

src/dense.i: src/dense.cpp.i

.PHONY : src/dense.i

# target to preprocess a source file
src/dense.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/dense.cpp.i
.PHONY : src/dense.cpp.i

src/dense.s: src/dense.cpp.s

.PHONY : src/dense.s

# target to generate assembly for a file
src/dense.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/dense.cpp.s
.PHONY : src/dense.cpp.s

src/dense_config.o: src/dense_config.cpp.o

.PHONY : src/dense_config.o

# target to build an object file
src/dense_config.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/dense_config.cpp.o
.PHONY : src/dense_config.cpp.o

src/dense_config.i: src/dense_config.cpp.i

.PHONY : src/dense_config.i

# target to preprocess a source file
src/dense_config.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/dense_config.cpp.i
.PHONY : src/dense_config.cpp.i

src/dense_config.s: src/dense_config.cpp.s

.PHONY : src/dense_config.s

# target to generate assembly for a file
src/dense_config.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/dense_config.cpp.s
.PHONY : src/dense_config.cpp.s

src/function/categorical_cross_entropy.o: src/function/categorical_cross_entropy.cpp.o

.PHONY : src/function/categorical_cross_entropy.o

# target to build an object file
src/function/categorical_cross_entropy.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/categorical_cross_entropy.cpp.o
.PHONY : src/function/categorical_cross_entropy.cpp.o

src/function/categorical_cross_entropy.i: src/function/categorical_cross_entropy.cpp.i

.PHONY : src/function/categorical_cross_entropy.i

# target to preprocess a source file
src/function/categorical_cross_entropy.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/categorical_cross_entropy.cpp.i
.PHONY : src/function/categorical_cross_entropy.cpp.i

src/function/categorical_cross_entropy.s: src/function/categorical_cross_entropy.cpp.s

.PHONY : src/function/categorical_cross_entropy.s

# target to generate assembly for a file
src/function/categorical_cross_entropy.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/categorical_cross_entropy.cpp.s
.PHONY : src/function/categorical_cross_entropy.cpp.s

src/function/dot.o: src/function/dot.cpp.o

.PHONY : src/function/dot.o

# target to build an object file
src/function/dot.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/dot.cpp.o
.PHONY : src/function/dot.cpp.o

src/function/dot.i: src/function/dot.cpp.i

.PHONY : src/function/dot.i

# target to preprocess a source file
src/function/dot.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/dot.cpp.i
.PHONY : src/function/dot.cpp.i

src/function/dot.s: src/function/dot.cpp.s

.PHONY : src/function/dot.s

# target to generate assembly for a file
src/function/dot.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/dot.cpp.s
.PHONY : src/function/dot.cpp.s

src/function/element_add.o: src/function/element_add.cpp.o

.PHONY : src/function/element_add.o

# target to build an object file
src/function/element_add.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_add.cpp.o
.PHONY : src/function/element_add.cpp.o

src/function/element_add.i: src/function/element_add.cpp.i

.PHONY : src/function/element_add.i

# target to preprocess a source file
src/function/element_add.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_add.cpp.i
.PHONY : src/function/element_add.cpp.i

src/function/element_add.s: src/function/element_add.cpp.s

.PHONY : src/function/element_add.s

# target to generate assembly for a file
src/function/element_add.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_add.cpp.s
.PHONY : src/function/element_add.cpp.s

src/function/element_add_assign.o: src/function/element_add_assign.cpp.o

.PHONY : src/function/element_add_assign.o

# target to build an object file
src/function/element_add_assign.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_add_assign.cpp.o
.PHONY : src/function/element_add_assign.cpp.o

src/function/element_add_assign.i: src/function/element_add_assign.cpp.i

.PHONY : src/function/element_add_assign.i

# target to preprocess a source file
src/function/element_add_assign.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_add_assign.cpp.i
.PHONY : src/function/element_add_assign.cpp.i

src/function/element_add_assign.s: src/function/element_add_assign.cpp.s

.PHONY : src/function/element_add_assign.s

# target to generate assembly for a file
src/function/element_add_assign.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_add_assign.cpp.s
.PHONY : src/function/element_add_assign.cpp.s

src/function/element_mul.o: src/function/element_mul.cpp.o

.PHONY : src/function/element_mul.o

# target to build an object file
src/function/element_mul.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_mul.cpp.o
.PHONY : src/function/element_mul.cpp.o

src/function/element_mul.i: src/function/element_mul.cpp.i

.PHONY : src/function/element_mul.i

# target to preprocess a source file
src/function/element_mul.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_mul.cpp.i
.PHONY : src/function/element_mul.cpp.i

src/function/element_mul.s: src/function/element_mul.cpp.s

.PHONY : src/function/element_mul.s

# target to generate assembly for a file
src/function/element_mul.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/element_mul.cpp.s
.PHONY : src/function/element_mul.cpp.s

src/function/reduce_avg.o: src/function/reduce_avg.cpp.o

.PHONY : src/function/reduce_avg.o

# target to build an object file
src/function/reduce_avg.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/reduce_avg.cpp.o
.PHONY : src/function/reduce_avg.cpp.o

src/function/reduce_avg.i: src/function/reduce_avg.cpp.i

.PHONY : src/function/reduce_avg.i

# target to preprocess a source file
src/function/reduce_avg.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/reduce_avg.cpp.i
.PHONY : src/function/reduce_avg.cpp.i

src/function/reduce_avg.s: src/function/reduce_avg.cpp.s

.PHONY : src/function/reduce_avg.s

# target to generate assembly for a file
src/function/reduce_avg.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/reduce_avg.cpp.s
.PHONY : src/function/reduce_avg.cpp.s

src/function/relu.o: src/function/relu.cpp.o

.PHONY : src/function/relu.o

# target to build an object file
src/function/relu.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/relu.cpp.o
.PHONY : src/function/relu.cpp.o

src/function/relu.i: src/function/relu.cpp.i

.PHONY : src/function/relu.i

# target to preprocess a source file
src/function/relu.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/relu.cpp.i
.PHONY : src/function/relu.cpp.i

src/function/relu.s: src/function/relu.cpp.s

.PHONY : src/function/relu.s

# target to generate assembly for a file
src/function/relu.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/relu.cpp.s
.PHONY : src/function/relu.cpp.s

src/function/scale.o: src/function/scale.cpp.o

.PHONY : src/function/scale.o

# target to build an object file
src/function/scale.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/scale.cpp.o
.PHONY : src/function/scale.cpp.o

src/function/scale.i: src/function/scale.cpp.i

.PHONY : src/function/scale.i

# target to preprocess a source file
src/function/scale.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/scale.cpp.i
.PHONY : src/function/scale.cpp.i

src/function/scale.s: src/function/scale.cpp.s

.PHONY : src/function/scale.s

# target to generate assembly for a file
src/function/scale.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/scale.cpp.s
.PHONY : src/function/scale.cpp.s

src/function/softmax.o: src/function/softmax.cpp.o

.PHONY : src/function/softmax.o

# target to build an object file
src/function/softmax.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/softmax.cpp.o
.PHONY : src/function/softmax.cpp.o

src/function/softmax.i: src/function/softmax.cpp.i

.PHONY : src/function/softmax.i

# target to preprocess a source file
src/function/softmax.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/softmax.cpp.i
.PHONY : src/function/softmax.cpp.i

src/function/softmax.s: src/function/softmax.cpp.s

.PHONY : src/function/softmax.s

# target to generate assembly for a file
src/function/softmax.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/softmax.cpp.s
.PHONY : src/function/softmax.cpp.s

src/function/squared_error.o: src/function/squared_error.cpp.o

.PHONY : src/function/squared_error.o

# target to build an object file
src/function/squared_error.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/squared_error.cpp.o
.PHONY : src/function/squared_error.cpp.o

src/function/squared_error.i: src/function/squared_error.cpp.i

.PHONY : src/function/squared_error.i

# target to preprocess a source file
src/function/squared_error.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/squared_error.cpp.i
.PHONY : src/function/squared_error.cpp.i

src/function/squared_error.s: src/function/squared_error.cpp.s

.PHONY : src/function/squared_error.s

# target to generate assembly for a file
src/function/squared_error.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/squared_error.cpp.s
.PHONY : src/function/squared_error.cpp.s

src/function/tanh.o: src/function/tanh.cpp.o

.PHONY : src/function/tanh.o

# target to build an object file
src/function/tanh.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/tanh.cpp.o
.PHONY : src/function/tanh.cpp.o

src/function/tanh.i: src/function/tanh.cpp.i

.PHONY : src/function/tanh.i

# target to preprocess a source file
src/function/tanh.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/tanh.cpp.i
.PHONY : src/function/tanh.cpp.i

src/function/tanh.s: src/function/tanh.cpp.s

.PHONY : src/function/tanh.s

# target to generate assembly for a file
src/function/tanh.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/function/tanh.cpp.s
.PHONY : src/function/tanh.cpp.s

src/gradient_descent.o: src/gradient_descent.cpp.o

.PHONY : src/gradient_descent.o

# target to build an object file
src/gradient_descent.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/gradient_descent.cpp.o
.PHONY : src/gradient_descent.cpp.o

src/gradient_descent.i: src/gradient_descent.cpp.i

.PHONY : src/gradient_descent.i

# target to preprocess a source file
src/gradient_descent.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/gradient_descent.cpp.i
.PHONY : src/gradient_descent.cpp.i

src/gradient_descent.s: src/gradient_descent.cpp.s

.PHONY : src/gradient_descent.s

# target to generate assembly for a file
src/gradient_descent.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/gradient_descent.cpp.s
.PHONY : src/gradient_descent.cpp.s

src/input.o: src/input.cpp.o

.PHONY : src/input.o

# target to build an object file
src/input.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/input.cpp.o
.PHONY : src/input.cpp.o

src/input.i: src/input.cpp.i

.PHONY : src/input.i

# target to preprocess a source file
src/input.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/input.cpp.i
.PHONY : src/input.cpp.i

src/input.s: src/input.cpp.s

.PHONY : src/input.s

# target to generate assembly for a file
src/input.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/input.cpp.s
.PHONY : src/input.cpp.s

src/input_config.o: src/input_config.cpp.o

.PHONY : src/input_config.o

# target to build an object file
src/input_config.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/input_config.cpp.o
.PHONY : src/input_config.cpp.o

src/input_config.i: src/input_config.cpp.i

.PHONY : src/input_config.i

# target to preprocess a source file
src/input_config.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/input_config.cpp.i
.PHONY : src/input_config.cpp.i

src/input_config.s: src/input_config.cpp.s

.PHONY : src/input_config.s

# target to generate assembly for a file
src/input_config.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/input_config.cpp.s
.PHONY : src/input_config.cpp.s

src/layer.o: src/layer.cpp.o

.PHONY : src/layer.o

# target to build an object file
src/layer.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/layer.cpp.o
.PHONY : src/layer.cpp.o

src/layer.i: src/layer.cpp.i

.PHONY : src/layer.i

# target to preprocess a source file
src/layer.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/layer.cpp.i
.PHONY : src/layer.cpp.i

src/layer.s: src/layer.cpp.s

.PHONY : src/layer.s

# target to generate assembly for a file
src/layer.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/layer.cpp.s
.PHONY : src/layer.cpp.s

src/layer_config.o: src/layer_config.cpp.o

.PHONY : src/layer_config.o

# target to build an object file
src/layer_config.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/layer_config.cpp.o
.PHONY : src/layer_config.cpp.o

src/layer_config.i: src/layer_config.cpp.i

.PHONY : src/layer_config.i

# target to preprocess a source file
src/layer_config.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/layer_config.cpp.i
.PHONY : src/layer_config.cpp.i

src/layer_config.s: src/layer_config.cpp.s

.PHONY : src/layer_config.s

# target to generate assembly for a file
src/layer_config.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/layer_config.cpp.s
.PHONY : src/layer_config.cpp.s

src/matrix.o: src/matrix.cpp.o

.PHONY : src/matrix.o

# target to build an object file
src/matrix.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/matrix.cpp.o
.PHONY : src/matrix.cpp.o

src/matrix.i: src/matrix.cpp.i

.PHONY : src/matrix.i

# target to preprocess a source file
src/matrix.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/matrix.cpp.i
.PHONY : src/matrix.cpp.i

src/matrix.s: src/matrix.cpp.s

.PHONY : src/matrix.s

# target to generate assembly for a file
src/matrix.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/matrix.cpp.s
.PHONY : src/matrix.cpp.s

src/matrix_helpers.o: src/matrix_helpers.cpp.o

.PHONY : src/matrix_helpers.o

# target to build an object file
src/matrix_helpers.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/matrix_helpers.cpp.o
.PHONY : src/matrix_helpers.cpp.o

src/matrix_helpers.i: src/matrix_helpers.cpp.i

.PHONY : src/matrix_helpers.i

# target to preprocess a source file
src/matrix_helpers.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/matrix_helpers.cpp.i
.PHONY : src/matrix_helpers.cpp.i

src/matrix_helpers.s: src/matrix_helpers.cpp.s

.PHONY : src/matrix_helpers.s

# target to generate assembly for a file
src/matrix_helpers.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/matrix_helpers.cpp.s
.PHONY : src/matrix_helpers.cpp.s

src/random_matrix_generator.o: src/random_matrix_generator.cpp.o

.PHONY : src/random_matrix_generator.o

# target to build an object file
src/random_matrix_generator.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/random_matrix_generator.cpp.o
.PHONY : src/random_matrix_generator.cpp.o

src/random_matrix_generator.i: src/random_matrix_generator.cpp.i

.PHONY : src/random_matrix_generator.i

# target to preprocess a source file
src/random_matrix_generator.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/random_matrix_generator.cpp.i
.PHONY : src/random_matrix_generator.cpp.i

src/random_matrix_generator.s: src/random_matrix_generator.cpp.s

.PHONY : src/random_matrix_generator.s

# target to generate assembly for a file
src/random_matrix_generator.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/random_matrix_generator.cpp.s
.PHONY : src/random_matrix_generator.cpp.s

src/scalar.o: src/scalar.cpp.o

.PHONY : src/scalar.o

# target to build an object file
src/scalar.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/scalar.cpp.o
.PHONY : src/scalar.cpp.o

src/scalar.i: src/scalar.cpp.i

.PHONY : src/scalar.i

# target to preprocess a source file
src/scalar.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/scalar.cpp.i
.PHONY : src/scalar.cpp.i

src/scalar.s: src/scalar.cpp.s

.PHONY : src/scalar.s

# target to generate assembly for a file
src/scalar.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/scalar.cpp.s
.PHONY : src/scalar.cpp.s

src/sequence.o: src/sequence.cpp.o

.PHONY : src/sequence.o

# target to build an object file
src/sequence.cpp.o:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/sequence.cpp.o
.PHONY : src/sequence.cpp.o

src/sequence.i: src/sequence.cpp.i

.PHONY : src/sequence.i

# target to preprocess a source file
src/sequence.cpp.i:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/sequence.cpp.i
.PHONY : src/sequence.cpp.i

src/sequence.s: src/sequence.cpp.s

.PHONY : src/sequence.s

# target to generate assembly for a file
src/sequence.cpp.s:
	$(MAKE) -f CMakeFiles/mozart.dir/build.make CMakeFiles/mozart.dir/src/sequence.cpp.s
.PHONY : src/sequence.cpp.s

test/mozart_test.o: test/mozart_test.cpp.o

.PHONY : test/mozart_test.o

# target to build an object file
test/mozart_test.cpp.o:
	$(MAKE) -f CMakeFiles/mozart_test.dir/build.make CMakeFiles/mozart_test.dir/test/mozart_test.cpp.o
.PHONY : test/mozart_test.cpp.o

test/mozart_test.i: test/mozart_test.cpp.i

.PHONY : test/mozart_test.i

# target to preprocess a source file
test/mozart_test.cpp.i:
	$(MAKE) -f CMakeFiles/mozart_test.dir/build.make CMakeFiles/mozart_test.dir/test/mozart_test.cpp.i
.PHONY : test/mozart_test.cpp.i

test/mozart_test.s: test/mozart_test.cpp.s

.PHONY : test/mozart_test.s

# target to generate assembly for a file
test/mozart_test.cpp.s:
	$(MAKE) -f CMakeFiles/mozart_test.dir/build.make CMakeFiles/mozart_test.dir/test/mozart_test.cpp.s
.PHONY : test/mozart_test.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install/strip"
	@echo "... install"
	@echo "... list_install_components"
	@echo "... rebuild_cache"
	@echo "... install/local"
	@echo "... test"
	@echo "... mozart"
	@echo "... edit_cache"
	@echo "... mozart_test"
	@echo "... googletest"
	@echo "... gmock"
	@echo "... gmock_main"
	@echo "... gtest_main"
	@echo "... gtest"
	@echo "... src/activation.o"
	@echo "... src/activation.i"
	@echo "... src/activation.s"
	@echo "... src/cost.o"
	@echo "... src/cost.i"
	@echo "... src/cost.s"
	@echo "... src/dense.o"
	@echo "... src/dense.i"
	@echo "... src/dense.s"
	@echo "... src/dense_config.o"
	@echo "... src/dense_config.i"
	@echo "... src/dense_config.s"
	@echo "... src/function/categorical_cross_entropy.o"
	@echo "... src/function/categorical_cross_entropy.i"
	@echo "... src/function/categorical_cross_entropy.s"
	@echo "... src/function/dot.o"
	@echo "... src/function/dot.i"
	@echo "... src/function/dot.s"
	@echo "... src/function/element_add.o"
	@echo "... src/function/element_add.i"
	@echo "... src/function/element_add.s"
	@echo "... src/function/element_add_assign.o"
	@echo "... src/function/element_add_assign.i"
	@echo "... src/function/element_add_assign.s"
	@echo "... src/function/element_mul.o"
	@echo "... src/function/element_mul.i"
	@echo "... src/function/element_mul.s"
	@echo "... src/function/reduce_avg.o"
	@echo "... src/function/reduce_avg.i"
	@echo "... src/function/reduce_avg.s"
	@echo "... src/function/relu.o"
	@echo "... src/function/relu.i"
	@echo "... src/function/relu.s"
	@echo "... src/function/scale.o"
	@echo "... src/function/scale.i"
	@echo "... src/function/scale.s"
	@echo "... src/function/softmax.o"
	@echo "... src/function/softmax.i"
	@echo "... src/function/softmax.s"
	@echo "... src/function/squared_error.o"
	@echo "... src/function/squared_error.i"
	@echo "... src/function/squared_error.s"
	@echo "... src/function/tanh.o"
	@echo "... src/function/tanh.i"
	@echo "... src/function/tanh.s"
	@echo "... src/gradient_descent.o"
	@echo "... src/gradient_descent.i"
	@echo "... src/gradient_descent.s"
	@echo "... src/input.o"
	@echo "... src/input.i"
	@echo "... src/input.s"
	@echo "... src/input_config.o"
	@echo "... src/input_config.i"
	@echo "... src/input_config.s"
	@echo "... src/layer.o"
	@echo "... src/layer.i"
	@echo "... src/layer.s"
	@echo "... src/layer_config.o"
	@echo "... src/layer_config.i"
	@echo "... src/layer_config.s"
	@echo "... src/matrix.o"
	@echo "... src/matrix.i"
	@echo "... src/matrix.s"
	@echo "... src/matrix_helpers.o"
	@echo "... src/matrix_helpers.i"
	@echo "... src/matrix_helpers.s"
	@echo "... src/random_matrix_generator.o"
	@echo "... src/random_matrix_generator.i"
	@echo "... src/random_matrix_generator.s"
	@echo "... src/scalar.o"
	@echo "... src/scalar.i"
	@echo "... src/scalar.s"
	@echo "... src/sequence.o"
	@echo "... src/sequence.i"
	@echo "... src/sequence.s"
	@echo "... test/mozart_test.o"
	@echo "... test/mozart_test.i"
	@echo "... test/mozart_test.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

