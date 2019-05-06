#-------------------------------------------------
#
# Project created by QtCreator 2016-10-26T19:07:09
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Viewer
TEMPLATE = app

CONFIG += c++11 precompile_header

PRECOMPILED_HEADER += headers.h

INCLUDEPATH += ../base
INCLUDEPATH += ../../external/rapidjson/include

SOURCES += main.cpp\
        widget.cpp \
    ../base/fbrain.cpp \
    ../base/fimage.cpp \
    ../base/fslice.cpp \
    ../base/fstack.cpp \
    ../base/misc.cpp \
    ../base/segment.cpp \
    ../base/structure.cpp \
    markpainter.cpp \
    ../base/mark.cpp

HEADERS  += \
    widget.h \
    headers.h \
    ../base/fbrain.h \
    ../base/fimage.h \
    ../base/fslice.h \
    ../base/fstack.h \
    ../base/misc.h \
    ../base/segment.h \
    ../base/structure.h \
    markpainter.h \
    ../base/mark.h

FORMS    += widget.ui

DISTFILES +=

RESOURCES += \
    icons.qrc

win32 {
    INCLUDEPATH += ../../lib/opencv/build/include
    INCLUDEPATH += ../../external/dirent/include
    LIBS += -L"../../lib/opencv/build/x64/vc14/lib" -lopencv_world320
}

unix {
    LIBS += -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio
}
