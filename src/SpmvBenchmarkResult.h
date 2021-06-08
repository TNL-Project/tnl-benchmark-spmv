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
          typename Index,
          typename Logger = JsonLogging >
struct SpmvBenchmarkResult
: public BenchmarkResult< Logger >
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using BenchmarkVector = Containers::Vector< Real, Device, Index >;

   using typename BenchmarkResult< Logger >::HeaderElements;
   using typename BenchmarkResult< Logger >::RowElements;
   using BenchmarkResult< Logger >::stddev;
   using BenchmarkResult< Logger >::bandwidth;
   using BenchmarkResult< Logger >::speedup;
   using BenchmarkResult< Logger >::time;


   SpmvBenchmarkResult( const String& format,
                        const HostVector& csrResult,
                        const BenchmarkVector& benchmarkResult,
                        const IndexType nonzeros )
   : format( format ), csrResult( csrResult ), benchmarkResult( benchmarkResult ), nonzeros( nonzeros ){};

   virtual HeaderElements getTableHeader() const override
   {
      return HeaderElements( {
         std::pair< String, int >( "format", 30 ),
         std::pair< String, int >( "device", 12 ),
         std::pair< String, int >( "non-zeros", 12 ),
         std::pair< String, int >( "time", 12 ),
         std::pair< String, int >( "stddev", 12 ),
         std::pair< String, int >( "stddev/time", 14 ),
         std::pair< String, int >( "bandwidth", 12 ),
         std::pair< String, int >( "speedup", 12 ),
         std::pair< String, int >( "CSR Diff.Max", 14 ),
         std::pair< String, int >( "CSR Diff.L2", 14 ) } );
   }

   void setFormat( const String& format ) { this->format = format; };

   virtual RowElements getRowElements() const override
   {
      HostVector benchmarkResultCopy;
      benchmarkResultCopy = benchmarkResult;
      auto diff = csrResult - benchmarkResultCopy;
      RowElements elements;
      elements << format
               << ( std::is_same< Device, Devices::Host >::value ? "CPU" : "GPU" )
               << nonzeros << time << stddev << stddev/time << bandwidth;
      if( speedup != 0.0 )
         elements << speedup;
      else elements << "N/A";
      elements << max( abs( diff ) ) << lpNorm( diff, 2.0 );
      return elements;
   }

   String format;
   const HostVector& csrResult;
   const BenchmarkVector& benchmarkResult;
   const IndexType nonzeros;
};

} //namespace Benchmarks
} //namespace TNL
