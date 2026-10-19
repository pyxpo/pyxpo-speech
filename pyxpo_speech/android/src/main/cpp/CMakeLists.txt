# libpyxpo_speech.so: whisper.cpp (and its ggml) linked in statically, behind a small JNI layer.
#
# The whisper.cpp source is a pinned release tarball, checked against its sha256 and kept in
# ~/.cache/pyxpo (downloaded once, shared by every app and ABI). Point PYXPO_WHISPER_DIR (a CMake
# variable or an environment variable) at a whisper.cpp checkout to build that instead, for example offline.
# ggml symbols stay hidden, so another module's ggml (llama.cpp) can live in the same app.
cmake_minimum_required(VERSION 3.18)
project(pyxpo_speech C CXX)

set(WHISPER_VERSION 1.9.5)
set(WHISPER_SHA256 ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720)
set(WHISPER_URL https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v${WHISPER_VERSION}.tar.gz)

if(NOT PYXPO_WHISPER_DIR AND DEFINED ENV{PYXPO_WHISPER_DIR})
    set(PYXPO_WHISPER_DIR "$ENV{PYXPO_WHISPER_DIR}")
endif()

if(NOT PYXPO_WHISPER_DIR)
    if(DEFINED ENV{XDG_CACHE_HOME})
        set(cache "$ENV{XDG_CACHE_HOME}/pyxpo")
    elseif(DEFINED ENV{HOME})
        set(cache "$ENV{HOME}/.cache/pyxpo")
    else()
        set(cache "${CMAKE_BINARY_DIR}/pyxpo-cache")
    endif()
    set(PYXPO_WHISPER_DIR "${cache}/whisper.cpp-${WHISPER_VERSION}")
    if(NOT EXISTS "${PYXPO_WHISPER_DIR}/CMakeLists.txt")
        file(MAKE_DIRECTORY "${cache}")
        # one ABI at a time: Gradle configures them in parallel
        file(LOCK "${cache}/whisper.cpp.lock" GUARD FILE TIMEOUT 1800)
        if(NOT EXISTS "${PYXPO_WHISPER_DIR}/CMakeLists.txt")
            set(tarball "${cache}/whisper.cpp-${WHISPER_VERSION}.tar.gz")
            # skipped when the file is there and matches the hash
            file(DOWNLOAD "${WHISPER_URL}" "${tarball}" EXPECTED_HASH SHA256=${WHISPER_SHA256} TLS_VERIFY ON
                 STATUS got)
            list(GET got 0 code)
            if(NOT code EQUAL 0)
                file(REMOVE "${tarball}")
                message(FATAL_ERROR "pyxpo-speech: could not download whisper.cpp ${WHISPER_VERSION} (${got}). "
                        "Download ${WHISPER_URL} yourself, check its sha256 is ${WHISPER_SHA256}, "
                        "unpack it and set PYXPO_WHISPER_DIR to the folder.")
            endif()
            set(tmp "${cache}/whisper.cpp-${WHISPER_VERSION}.tmp")
            file(REMOVE_RECURSE "${tmp}")
            file(ARCHIVE_EXTRACT INPUT "${tarball}" DESTINATION "${tmp}")
            file(RENAME "${tmp}/whisper.cpp-${WHISPER_VERSION}" "${PYXPO_WHISPER_DIR}")
            file(REMOVE_RECURSE "${tmp}")
        endif()
        file(LOCK "${cache}/whisper.cpp.lock" RELEASE)
    endif()
endif()

# Debug app builds still compile whisper optimized: unoptimized it is about ten times slower.
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)      # the build host is not the phone
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)      # ggml's own thread pool; no libomp.so to ship
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
add_subdirectory("${PYXPO_WHISPER_DIR}" whisper EXCLUDE_FROM_ALL)

add_library(pyxpo_speech SHARED speech_jni.cpp)
target_link_libraries(pyxpo_speech PRIVATE whisper)
target_compile_features(pyxpo_speech PRIVATE cxx_std_17)
target_compile_definitions(pyxpo_speech PRIVATE PYXPO_WHISPER_VERSION="${WHISPER_VERSION}")
# 16 KB pages: Android 15+ devices and Google Play need it; NDK 27 links for 4 KB by default
target_link_options(pyxpo_speech PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections -Wl,-z,max-page-size=16384)
