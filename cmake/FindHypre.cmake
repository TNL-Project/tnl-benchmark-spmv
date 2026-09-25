################################################################################
#
# \file      FindHypre.cmake
# \copyright 2012-2015 J. Bakosi,
#            2016-2018 Los Alamos National Security, LLC.,
#            2019-2021 Triad National Security, LLC.
#            All rights reserved. See the LICENSE file for details.
# \brief     Find the Hypre library from LLNL
#
################################################################################

# Hypre: https://github.com/LLNL/hypre
#
#  HYPRE_FOUND - System has Hypre
#  HYPRE_INCLUDE_DIRS - The Hypre include directory
#  HYPRE_LIBRARIES - The libraries needed to use Hypre
#  HYPRE_WITH_CUDA - ON if Hypre was built with CUDA (HYPRE_USING_CUDA in HYPRE_config.h)
#
#  Set HYPRE_ROOT before calling find_package to a path to add an additional
#  search path, e.g.,
#
#  Usage:
#
#  set(HYPRE_ROOT "/path/to/custom/hypre") # prefer over system
#  find_package(Hypre)
#  if(HYPRE_FOUND)
#    target_link_libraries (TARGET ${HYPRE_LIBRARIES})
#  endif()

function(_HYPRE_GET_VERSION _OUT_ver _version_hdr)
    file(
        STRINGS ${_version_hdr}
        _contents
        REGEX "#define HYPRE_RELEASE_VERSION[ \t]+"
    )
    if(_contents)
        string(REGEX REPLACE "\"" "" _cont "${_contents}")
        string(
            REGEX REPLACE ".*#define HYPRE_RELEASE_VERSION[ \t]+([0-9.]+).*"
            "\\1"
            ${_OUT_ver}
            "${_cont}"
        )
        if(NOT ${${_OUT_ver}} MATCHES "[0-9]+")
            message(
                FATAL_ERROR
                "Version parsing failed for HYPRE_RELEASE_VERSION in ${_version_hdr}!"
            )
        endif()
        set(${_OUT_ver} ${${_OUT_ver}} PARENT_SCOPE)
    elseif(EXISTS ${_version_hdr})
        message(FATAL_ERROR "No HYPRE_RELEASE_VERSION in ${_version_hdr}")
    else()
        message(FATAL_ERROR "Include file ${_version_hdr} does not exist")
    endif()
endfunction()

# Set _OUT_defined to TRUE if "#define <_macro> 1" is in the given header
function(_HYPRE_CONFIG_DEFINED _OUT_defined _config_hdr _macro)
    file(STRINGS ${_config_hdr} _contents REGEX "^#define ${_macro}[ \t]+1")
    if(_contents)
        set(${_OUT_defined} TRUE PARENT_SCOPE)
    else()
        set(${_OUT_defined} FALSE PARENT_SCOPE)
    endif()
endfunction()

# If HYPRE_ROOT changed since the last configuration, forget the cached
# results so that the new location is searched
if(NOT "${HYPRE_ROOT}" STREQUAL "${_HYPRE_ROOT_LAST}")
    unset(HYPRE_INCLUDE_DIR CACHE)
    unset(HYPRE_LIBRARY CACHE)
    unset(HYPRE_UMPIRE_LIBRARY CACHE)
    unset(HYPRE_CAMP_LIBRARY CACHE)
    unset(HYPRE_FMT_LIBRARY CACHE)
endif()
set(_HYPRE_ROOT_LAST "${HYPRE_ROOT}" CACHE INTERNAL "HYPRE_ROOT used in the last configuration")

# If already in cache, be silent
if(HYPRE_INCLUDE_DIRS AND HYPRE_LIBRARIES)
    set(HYPRE_FIND_QUIETLY TRUE)
endif()

if(HYPRE_ROOT)
    set(HYPRE_SEARCH_OPTS NO_DEFAULT_PATH)
else()
    set(HYPRE_ROOT "/usr")
endif()

find_path(
    HYPRE_INCLUDE_DIR
    NAMES HYPRE.h
    PATH_SUFFIXES hypre
    HINTS ${HYPRE_ROOT}/include ${HYPRE_ROOT}/include/hypre ${HYPRE_SEARCH_OPTS}
)

if(HYPRE_INCLUDE_DIR)
    _hypre_get_version(HYPRE_VERSION ${HYPRE_INCLUDE_DIR}/HYPRE_config.h)
    set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
    # Hypre built with CUDA has "#define HYPRE_USING_CUDA 1" in HYPRE_config.h
    _hypre_config_defined(_hypre_using_cuda ${HYPRE_INCLUDE_DIR}/HYPRE_config.h HYPRE_USING_CUDA)
    if(_hypre_using_cuda)
        set(HYPRE_WITH_CUDA ON)
    else()
        set(HYPRE_WITH_CUDA OFF)
    endif()
else()
    set(HYPRE_VERSION 0.0.0)
    set(HYPRE_INCLUDE_DIRS "")
    set(HYPRE_WITH_CUDA OFF)
endif()

find_library(HYPRE_LIBRARY NAMES HYPRE HINTS ${HYPRE_ROOT}/lib ${HYPRE_ROOT}/lib64 ${HYPRE_SEARCH_OPTS})

set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})

# A static libHYPRE.a does not carry its dependencies, so add the libraries
# Hypre was configured with (as recorded in HYPRE_config.h) explicitly.
if(HYPRE_INCLUDE_DIR AND HYPRE_LIBRARY)
    set(_hypre_config_hdr ${HYPRE_INCLUDE_DIR}/HYPRE_config.h)

    # Umpire (and its dependencies camp and fmt) - installed along with Hypre
    _hypre_config_defined(_hypre_using_umpire ${_hypre_config_hdr} HYPRE_USING_UMPIRE)
    if(_hypre_using_umpire)
        foreach(_lib umpire camp fmt)
            string(TOUPPER ${_lib} _LIB)
            find_library(
                HYPRE_${_LIB}_LIBRARY
                NAMES ${_lib}
                HINTS ${HYPRE_ROOT}/lib ${HYPRE_ROOT}/lib64 ${HYPRE_SEARCH_OPTS}
            )
            mark_as_advanced(HYPRE_${_LIB}_LIBRARY)
            if(HYPRE_${_LIB}_LIBRARY)
                list(APPEND HYPRE_LIBRARIES ${HYPRE_${_LIB}_LIBRARY})
            elseif(NOT _lib STREQUAL "fmt")
                message(WARNING "Hypre was built with Umpire but lib${_lib} was not found in ${HYPRE_ROOT}.")
            endif()
        endforeach()
    endif()

    # CUDA libraries
    if(HYPRE_WITH_CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            list(APPEND HYPRE_LIBRARIES CUDA::cudart)
            foreach(_lib cusparse curand cublas cusolver)
                string(TOUPPER ${_lib} _LIB)
                _hypre_config_defined(_hypre_using_lib ${_hypre_config_hdr} HYPRE_USING_${_LIB})
                if(_hypre_using_lib)
                    list(APPEND HYPRE_LIBRARIES CUDA::${_lib})
                    if(_lib STREQUAL "cublas")
                        list(APPEND HYPRE_LIBRARIES CUDA::cublasLt)
                    endif()
                endif()
            endforeach()
        else()
            message(WARNING "Hypre was built with CUDA but the CUDA toolkit was not found.")
        endif()
    endif()
endif()

# Handle the QUIETLY and REQUIRED arguments and set HYPRE_FOUND to TRUE if
# all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Hypre
    REQUIRED_VARS HYPRE_LIBRARY HYPRE_INCLUDE_DIRS
    VERSION_VAR HYPRE_VERSION
)

if(HYPRE_FOUND)
    message(STATUS "Hypre built with CUDA: ${HYPRE_WITH_CUDA}")
    message(STATUS "Hypre libraries: ${HYPRE_LIBRARIES}")
endif()

mark_as_advanced(HYPRE_INCLUDE_DIRS HYPRE_LIBRARIES)
