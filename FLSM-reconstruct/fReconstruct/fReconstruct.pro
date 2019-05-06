TARGET = fReconstruct
CONFIG += console c++11 thread precompile_header
CONFIG -= app_bundle qt

TEMPLATE = app

INCLUDEPATH += ../base
INCLUDEPATH += ../../external/rapidjson/include
#mingw
#INCLUDEPATH+=E:/cv2/include
#LIBS += -L"E:/cv2/lib/" -lopencv_calib3d2410 -lopencv_contrib2410 -lopencv_core2410 -lopencv_features2d2410 -lopencv_flann2410 -lopencv_gpu2410 -lopencv_highgui2410 -lopencv_imgproc2410 -lopencv_legacy2410 -lopencv_ml2410 -lopencv_nonfree2410 -lopencv_objdetect2410 -lopencv_ocl2410 -lopencv_photo2410 -lopencv_stitching2410 -lopencv_superres2410 -lopencv_ts2410 -lopencv_video2410 -lopencv_videostab2410

win32 {
    QMAKE_CXXFLAGS += -openmp
    INCLUDEPATH += ../../lib/opencv/build/include
    INCLUDEPATH += ../../external/dirent/include
    LIBS += -L"../../lib/opencv/build/x64/vc14/lib" -lopencv_world320
}

unix {
    QMAKE_CXXFLAGS += -fopenmp
    QMAKE_LFLAGS += -fopenmp
    LIBS += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_features2d -lopencv_flann -lopencv_videoio -lopencv_photo
}

PRECOMPILED_HEADER += headers.h

SOURCES += main.cpp \
    ftask.cpp \
    genvoxeltask.cpp \
    cellcountingtask.cpp \
    ../base/fstack.cpp \
    ../base/structure.cpp \
    ../base/segment.cpp \
    ../base/fimage.cpp \
    ../base/fslice.cpp \
    ../base/fbrain.cpp \
    ../base/misc.cpp \
    ../base/mark.cpp \
    ../base/pipeline.cpp \
    compresstask.cpp

HEADERS += headers.h\
    ftask.h \
    genvoxeltask.h \
    cellcountingtask.h \
    ../base/fstack.h \
    ../base/structure.h \
    ../base/segment.h \
    ../base/fimage.h \
    ../base/fslice.h \
    ../base/fbrain.h \
    ../base/misc.h \
    ../base/mark.h \
    ../base/pipeline.h \
    compresstask.h
