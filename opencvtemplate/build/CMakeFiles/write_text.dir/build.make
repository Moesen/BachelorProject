# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\Programs\cmake\bin\cmake.exe

# The command to remove a file.
RM = D:\Programs\cmake\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\GIT\Bachelor\opencvtemplate

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\GIT\Bachelor\opencvtemplate\build

# Include any dependencies generated for this target.
include CMakeFiles/write_text.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/write_text.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/write_text.dir/flags.make

CMakeFiles/write_text.dir/writetext.cpp.obj: CMakeFiles/write_text.dir/flags.make
CMakeFiles/write_text.dir/writetext.cpp.obj: CMakeFiles/write_text.dir/includes_CXX.rsp
CMakeFiles/write_text.dir/writetext.cpp.obj: ../writetext.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\GIT\Bachelor\opencvtemplate\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/write_text.dir/writetext.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\write_text.dir\writetext.cpp.obj -c E:\GIT\Bachelor\opencvtemplate\writetext.cpp

CMakeFiles/write_text.dir/writetext.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/write_text.dir/writetext.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\GIT\Bachelor\opencvtemplate\writetext.cpp > CMakeFiles\write_text.dir\writetext.cpp.i

CMakeFiles/write_text.dir/writetext.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/write_text.dir/writetext.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\GIT\Bachelor\opencvtemplate\writetext.cpp -o CMakeFiles\write_text.dir\writetext.cpp.s

# Object files for target write_text
write_text_OBJECTS = \
"CMakeFiles/write_text.dir/writetext.cpp.obj"

# External object files for target write_text
write_text_EXTERNAL_OBJECTS =

write_text.exe: CMakeFiles/write_text.dir/writetext.cpp.obj
write_text.exe: CMakeFiles/write_text.dir/build.make
write_text.exe: CMakeFiles/write_text.dir/linklibs.rsp
write_text.exe: CMakeFiles/write_text.dir/objects1.rsp
write_text.exe: CMakeFiles/write_text.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=E:\GIT\Bachelor\opencvtemplate\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable write_text.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\write_text.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/write_text.dir/build: write_text.exe

.PHONY : CMakeFiles/write_text.dir/build

CMakeFiles/write_text.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\write_text.dir\cmake_clean.cmake
.PHONY : CMakeFiles/write_text.dir/clean

CMakeFiles/write_text.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\GIT\Bachelor\opencvtemplate E:\GIT\Bachelor\opencvtemplate E:\GIT\Bachelor\opencvtemplate\build E:\GIT\Bachelor\opencvtemplate\build E:\GIT\Bachelor\opencvtemplate\build\CMakeFiles\write_text.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/write_text.dir/depend
