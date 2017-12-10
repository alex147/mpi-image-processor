TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
INCLUDEPATH += "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\x64"
LIBS += "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib" "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpifec.lib" "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpifmc.lib"

SOURCES += main.c
