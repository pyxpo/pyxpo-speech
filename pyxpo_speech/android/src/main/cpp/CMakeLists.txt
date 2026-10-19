# libpyxpo_speech.so: a small JNI layer over whisper.cpp and its ggml, built as shared libraries so ggml can
# pick, on the phone, the CPU build that uses the phone's fastest instructions (dot product, fp16, int8 matmul,
# SVE, SME). One build for generic armv8.0 is several times slower on a recent phone.
#
# The whisper.cpp source is a pinned release tarball, checked against its sha256 and kept in
# ~/.cache/pyxpo (downloaded once, shared by every app and ABI). Point PYXPO_WHISPER_DIR (a CMake
# variable or an environment variable) at a whisper.cpp checkout to build that instead, for example offline.
# Every library is named libpyxpo_speech_*.so, so another module's ggml (llama.cpp) can live in the same app:
# the files do not clash in the APK, and each library finds the ggml it was linked with.
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

# 16 KB pages: Android 15+ devices and Google Play need it; NDK 27 links for 4 KB by default
string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-z,max-page-size=16384")
string(APPEND CMAKE_MODULE_LINKER_FLAGS " -Wl,-z,max-page-size=16384")

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)   # CPU backends are loaded at run time (speech_jni.cpp)
# arm64 and x86_64: one CPU backend per instruction set level; other ABIs: one plain CPU backend
if(ANDROID_ABI STREQUAL "arm64-v8a" OR ANDROID_ABI STREQUAL "x86_64")
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
endif()
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)      # the build host is not the phone
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)      # ggml's own thread pool; no libomp.so to ship
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
add_subdirectory("${PYXPO_WHISPER_DIR}" whisper EXCLUDE_FROM_ALL)

# ggml builds every CPU backend as a dependency of ggml; GGML_AVAILABLE_BACKENDS lists them, oldest CPU first.
# parakeet (another speech model in whisper.cpp) is unused, but the Android Gradle plugin builds and packs
# every shared library target, so it is renamed too.
set(cpu_backends "")
foreach(target ggml ggml-base whisper parakeet ${GGML_AVAILABLE_BACKENDS})
    if(NOT TARGET ${target})   # left in the cache by an earlier configure with other options
        continue()
    endif()
    string(REPLACE "-" "_" name "pyxpo_speech_${target}")
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name})
    if(target MATCHES "^ggml-cpu")
        list(PREPEND cpu_backends "lib${name}.so")
    endif()
endforeach()
list(JOIN cpu_backends "," cpu_backends)

add_library(pyxpo_speech SHARED speech_jni.cpp)
target_link_libraries(pyxpo_speech PRIVATE whisper ggml)
target_compile_features(pyxpo_speech PRIVATE cxx_std_17)
target_compile_definitions(pyxpo_speech PRIVATE PYXPO_WHISPER_VERSION="${WHISPER_VERSION}"
                           PYXPO_CPU_BACKENDS="${cpu_backends}")
target_link_options(pyxpo_speech PRIVATE -Wl,--gc-sections)
