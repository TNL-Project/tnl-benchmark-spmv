#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

namespace TNL::Benchmarks {

template< typename Real, typename Device, typename Index, typename ResultReal = Real, typename Logger = JsonLogging >
struct SpmvBenchmarkResult : public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using BenchmarkVector = Containers::Vector< ResultReal, Device, Index >;

   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time_mean;
   using BenchmarkResult::time_median;
   using BenchmarkResult::time_stddev;
   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;

   SpmvBenchmarkResult( const HostVector& csrResult, const BenchmarkVector& benchmarkResult )
   : csrResult( csrResult ),
     benchmarkResult( benchmarkResult )
   {}

   virtual HeaderElements
   getTableHeader() const override
   {
      return HeaderElements(
         { "time median", "speedup", "bandwidth", "CSR Diff.Max", "CSR Diff.L2", "time mean", "time stddev", "loops" } );
   }

   virtual std::vector< int >
   getColumnWidthHints() const override
   {
      return std::vector< int >( { 16,      // time_median
                                   16,      // speedup
                                   16,      // bandwidth
                                   16,      // CSR Diff.Max
                                   16,      // CSR Diff. L2,
                                   16,      // time mean
                                   16,      // time stddev
                                   10 } );  // loops
   }

   virtual RowElements
   getRowElements() const override
   {
      HostVector benchmarkResultCopy;
      benchmarkResultCopy = benchmarkResult;
      auto diff = csrResult - benchmarkResultCopy;
      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific << time_median;
      if( speedup != 0.0 )
         elements << speedup;
      else
         elements << "N/A";
      elements << bandwidth << max( abs( diff ) ) << lpNorm( diff, 2.0 ) << time_mean << time_stddev << loops;
      return elements;
   }

   const HostVector& csrResult;
   const BenchmarkVector& benchmarkResult;
};

}  // namespace TNL::Benchmarks
