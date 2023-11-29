// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include <TNL/Backend/Macros.h>
#include <TNL/Exceptions/BackendSupportMissing.h>
#include <TNL/Exceptions/BackendBadAlloc.h>

namespace TNL::Cuda {

template< typename ObjectType >
//[[deprecated( "Allocators::Cuda and MultiDeviceMemoryOperations should be used instead." )]]
ObjectType*
passToDevice( const ObjectType& object )
{
#ifdef __CUDACC__
   ObjectType* deviceObject;
   if( cudaMalloc( (void**) &deviceObject, (size_t) sizeof( ObjectType ) ) != cudaSuccess )
      throw Exceptions::BackendBadAlloc();
   if( cudaMemcpy( (void*) deviceObject, (void*) &object, sizeof( ObjectType ), cudaMemcpyHostToDevice ) != cudaSuccess ) {
      TNL_CHECK_CUDA_DEVICE;
      cudaFree( (void*) deviceObject );
      TNL_CHECK_CUDA_DEVICE;
      return 0;
   }
   return deviceObject;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

template< typename ObjectType >
//[[deprecated( "Allocators::Cuda should be used instead." )]]
void
freeFromDevice( ObjectType* deviceObject )
{
#ifdef __CUDACC__
   cudaFree( (void*) deviceObject );
   TNL_CHECK_CUDA_DEVICE;
#else
   throw Exceptions::BackendSupportMissing();
#endif
}

}  // namespace TNL::Cuda
