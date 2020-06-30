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

template< typename Real,
          typename Device,
          typename Index >
struct SpmvBenchmarkResult
: public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using BenchmarkVector = Containers::Vector< Real, Device, Index >;

   SpmvBenchmarkResult( const HostVector& csrResult,
                        const BenchmarkVector& benchmarkResult,
                        const IndexType nonzeros )
   : csrResult( csrResult ), benchmarkResult( benchmarkResult ), nonzeros( nonzeros ){};

   virtual HeaderElements getTableHeader() const override
   {
      return HeaderElements( {"non-zeros", "time", "stddev", "stddev/time", "bandwidth", "speedup", "CSR Diff.Max", "CSR Diff.L2"} );
   }

   virtual RowElements getRowElements() const override
   {
      HostVector benchmarkResultCopy;
      benchmarkResultCopy = benchmarkResult;
      auto diff = csrResult - benchmarkResultCopy;
      RowElements elements;
      elements << nonzeros << time << stddev << stddev/time << bandwidth;
      if( speedup != 0.0 )
         elements << speedup;
      else elements << "N/A";
      elements << max( abs( diff ) ) << lpNorm( diff, 2.0 );
      return elements;
   }

   const HostVector& csrResult;
   const BenchmarkVector& benchmarkResult;
   const IndexType nonzeros;
};
   
} //namespace Benchmarks
} //namespace TNL
