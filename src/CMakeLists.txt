if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv tnl ${CUDA_cusparse_LIBRARY} )
else()
    ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv tnl )
endif()

install( TARGETS tnl-benchmark-spmv RUNTIME DESTINATION bin )
