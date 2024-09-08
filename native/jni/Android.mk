LOCAL_PATH := $(call my-dir)


include $(CLEAR_VARS)
LOCAL_MODULE    := info_get
LOCAL_SRC_FILES := execute.cpp fps.cpp state.cpp temp.cpp
APP_STL := c++_static
APP_ABI := arm64-v8a
include $(BUILD_EXECUTABLE)


include $(CLEAR_VARS)
LOCAL_MODULE := pmu_test 
LOCAL_SRC_FILES := pmu_test.cpp
APP_STL := c++_static
APP_ABI := arm64-v8a
include $(BUILD_EXECUTABLE)