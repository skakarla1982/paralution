# *************************************************************************
#
#    PARALUTION   www.paralution.com
#
#    Copyright (C) 2015  PARALUTION Labs UG (haftungsbeschränkt) & Co. KG
#                        Am Hasensprung 6, 76571 Gaggenau
#                        Handelsregister: Amtsgericht Mannheim, HRA 706051
#                        Vertreten durch:
#                        PARALUTION Labs Verwaltungs UG (haftungsbeschränkt)
#                        Am Hasensprung 6, 76571 Gaggenau
#                        Handelsregister: Amtsgericht Mannheim, HRB 721277
#                        Geschäftsführer: Dimitar Lukarski, Nico Trost
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************



# PARALUTION version 1.1.0 
# Srinivas K
macro(HIP_COMPILE generated_files)
  hip_compile_base(hip_compile OBJ ${generated_files} ${ARGN})
endmacro()

macro(HIP_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xOPTIONS")
      set( _found_options TRUE )
    elseif(
        "x${arg}" STREQUAL "xWIN32" OR
        "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
        "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
        "x${arg}" STREQUAL "xSTATIC" OR
        "x${arg}" STREQUAL "xSHARED" OR
        "x${arg}" STREQUAL "xMODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()

macro(hip_compile_base hip_target format generated_files)
  # Update a counter in this directory, to keep phony target names unique.
  set(_hip_target "${hip_target}")
  get_property(_counter DIRECTORY PROPERTY _hip_internal_phony_counter)
  if(_counter)
    math(EXPR _counter "${_counter} + 1")
  else()
    set(_counter 1)
  endif()
  string(APPEND _hip_target "_${_counter}")
  set_property(DIRECTORY PROPERTY _hip_internal_phony_counter ${_counter})

  # Separate the sources from the options
  HIP_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})

  # Create custom commands and targets for each file.
  #HIP_WRAP_SRCS( ${_hip_target} ${format} _generated_files ${_sources}
  #                ${_cmake_options} OPTIONS ${_options} PHONY)

  set( ${generated_files} ${_generated_files})

endmacro()

# Srinivas K


set(PARALUTION_SOURCES)
set(PARALUTION_PUBLIC_HEADERS
  paralution.hpp
)

# Copy headers to binary include dir
foreach(i ${PARALUTION_PUBLIC_HEADERS})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${i}" "${INC_BUILD_DIR}/${i}" COPYONLY)
endforeach()

# Fill PARALUTION lists with sources and headers for each subdirectory
# base
include(base/CMakeLists.txt)
foreach(i ${BASE_SOURCES})
  list(APPEND PARALUTION_SOURCES base/${i})
endforeach()
foreach(i ${BASE_PUBLIC_HEADERS})
  list(APPEND PARALUTION_PUBLIC_HEADERS base/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/base/${i}" "${INC_BUILD_DIR}/base/${i}" COPYONLY)
endforeach()

# solvers
include(solvers/CMakeLists.txt)
foreach(i ${SOLVERS_SOURCES})
  list(APPEND PARALUTION_SOURCES solvers/${i})
endforeach()
foreach(i ${SOLVERS_PUBLIC_HEADERS})
  list(APPEND PARALUTION_PUBLIC_HEADERS solvers/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solvers/${i}" "${INC_BUILD_DIR}/solvers/${i}" COPYONLY)
endforeach()

# utils
include(utils/CMakeLists.txt)
foreach(i ${UTILS_SOURCES})
  list(APPEND PARALUTION_SOURCES utils/${i})
endforeach()
foreach(i ${UTILS_PUBLIC_HEADERS})
  list(APPEND PARALUTION_PUBLIC_HEADERS utils/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/utils/${i}" "${INC_BUILD_DIR}/utils/${i}" COPYONLY)
endforeach()

# plug-ins
include(plug-ins/CMakeLists.txt)
foreach(i ${PLUG-INS_SOURCES})
  list(APPEND PARALUTION_SOURCES plug-ins/${i})
endforeach()
foreach(i ${PLUG-INS_PUBLIC_HEADERS})
  list(APPEND PARALUTION_PUBLIC_HEADERS plug-ins/${i})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/plug-ins/${i}" "${PLUGINS_BUILD_DIR}/${i}" COPYONLY)
endforeach()

#######################################################################################################################

# Replace CUDA files in PARALUTION list since it is compiled by NVCC
if (SUPPORT_CUDA)
  foreach(i ${PARALUTION_SOURCES})

    if (${i} MATCHES ".cu")
      CUDA_COMPILE(cu ${i})
      list(REMOVE_ITEM PARALUTION_SOURCES ${i})
      list(APPEND PARALUTION_SOURCES ${cu})
    endif()
    
  endforeach()
endif()

# Srinivas K
if (SUPPORT_HIP)
  foreach(i ${PARALUTION_SOURCES})

    if (${i} MATCHES ".cu")
            HIP_COMPILE(cu ${i})
      list(REMOVE_ITEM PARALUTION_SOURCES ${i})
      list(APPEND PARALUTION_SOURCES ${cu})
    endif()

  endforeach()
endif()

# Create paralution library
if (SUPPORT_MIC)
  SET(CMAKE_AR "xiar")
  SET(CMAKE_LINKER "xild")
  SET(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> -qoffload-build rcs <TARGET> <LINK_FLAGS> <OBJECTS>")
  SET(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -qoffload-build rcs <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

add_library(${PARALUTION_LIBRARY} ${LIBRARY_TYPE} ${PARALUTION_SOURCES})

#######################################################################################################################

# Link platform specific libraries
if (SUPPORT_CUDA)
  target_link_libraries(${PARALUTION_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
endif()

if (SUPPORT_HIP)
        SET(HIP_LIBRARIES "-L/opt/rocm/lib -lhip_hcc -lhipblas -lhipsparse")
        target_link_libraries(${PARALUTION_LIBRARY} ${HIP_LIBRARIES})
endif()


if (SUPPORT_OCL)
  target_link_libraries(${PARALUTION_LIBRARY} ${OPENCL_LIBRARIES})
endif()

if (SUPPORT_MIC)
  target_link_libraries(${PARALUTION_LIBRARY} -openmp)
endif()

if (SUPPORT_MKL)
  target_link_libraries(${PARALUTION_LIBRARY} ${MKL_LIBRARIES})
endif()
