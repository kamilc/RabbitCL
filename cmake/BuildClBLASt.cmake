set(BRANCH master)
set(URL https://github.com/CNugteren/CLBlast.git)

message("Cloning CLBlast - Tuned OpenCL BLAS")

execute_process(COMMAND git clone -b ${BRANCH} ${URL} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies)
execute_process(COMMAND mkdir CLBlast/build  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies)
execute_process(COMMAND cmake -DTUNERS=ON -DBUILD_SHARED_LIBS=OFF .. WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND make WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND ./clblast_tuner_xgemm WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND ./clblast_tuner_xgemm_direct WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND ./clblast_tuner_transpose_fast WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND ./clblast_tuner_transpose_pad WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND python ../scripts/database/database.py . .. WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)
execute_process(COMMAND make WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build)

set(ClBlast_INCLUDE ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/include)
set(ClBlast_LIB ${CMAKE_SOURCE_DIR}/dependencies/ClBlast/build/libclblast.a)
