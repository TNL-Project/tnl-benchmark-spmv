/***************************************************************************
                          SpmvBenchmarkResult.h  -  description
                             -------------------
    begin                : Mar 5, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "../Benchmarks.h"

namespace TNL {
namespace Benchmarks {

template< typename Real = double,
          typename Index = int >
struct SpmvBenchmarkResult
: public BenchmarkResult
{
   using RealType = Real;
   using IndexType = Index;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, Index >;

   SpmvBenchmarkResult( CudaVector& cudaResult, HostVector& hostResult, CudaVector& cusparseResult )
   : hostResult( hostResult ), cudaResult( cudaResult), cusparseResult( cusparseResult ){};

   virtual HeaderElements getTableHeader() const override
   {
      return HeaderElements({"time", "stddev", "stddev/time", "speedup", "Host.Diff.Max", "Host.Diff.L2", "Cusparse.Diff.Max", "Cusparse.Diff.L2"});
   }

   virtual RowElements getRowElements() const override
   {
      HostVector cudaCopy, cusparseCopy, a, b;
      cudaCopy = cudaResult;
      cusparseCopy = cusparseResult;
      a = cudaCopy - hostResult;
      b = cudaCopy - cusparseCopy;
      return RowElements( { 
         convertToString( time ),
         convertToString( stddev ),
         convertToString( stddev/time ),
         speedup != 0 ? convertToString( speedup ) : "N/A",
         convertToString( max( abs( a ) ) ),
         convertToString( lpNorm( a, 2.0 ) ),
         convertToString( max( abs( b ) ) ),
         convertToString( lpNorm( b, 2.0 ) ) } );
   }

   HostVector &hostResult;

   CudaVector &cudaResult, &cusparseResult;
};
   
} //namespace Benchmarks
} //namespace TNL
