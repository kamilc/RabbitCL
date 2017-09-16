set(BRANCH develop)
set(URL https://github.com/clMathLibraries/clBLAS.git)

message("Cloning clBLAS")

execute_process(COMMAND git clone -b ${BRANCH} ${URL}  WORKING_DIRECTORY ../dependencies)
execute_process(COMMAND mkdir clBLAS/src/build  WORKING_DIRECTORY ../dependencies)
execute_process(COMMAND cmake .. WORKING_DIRECTORY ../dependencies/clBLAS/src/build)
execute_process(COMMAND make WORKING_DIRECTORY ../dependencies/clBLAS/src/build)
