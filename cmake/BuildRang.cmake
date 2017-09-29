
set(BRANCH master)
set(URL https://github.com/agauniyal/rang.git)

message("Cloning Rang")

execute_process(COMMAND git clone -b ${BRANCH} ${URL} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies)

set(Rang_INCLUDE ${CMAKE_SOURCE_DIR}/dependencies/rang/include)
