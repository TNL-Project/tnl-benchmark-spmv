if( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv ${CUDA_cusparse_LIBRARY} -lcudadevrt )
else()
    ADD_EXECUTABLE( tnl-benchmark-spmv tnl-benchmark-spmv.cpp )
endif()

install( TARGETS tnl-benchmark-spmv RUNTIME DESTINATION bin )
