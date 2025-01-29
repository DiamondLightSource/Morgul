find_package(Git REQUIRED)

function(ensure_submodule dir)
    # Just run this any time, this is only at configure and we may need submodules
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY)
endfunction(ensure_submodule)