build_lib(
  LIBNAME MyModules
  SOURCE_FILES
    Model/MyTcpApp.cc
  HEADER_FILES
    Model/MyTcpApp.h
  LIBRARIES_TO_LINK
  TEST_SOURCES
)
