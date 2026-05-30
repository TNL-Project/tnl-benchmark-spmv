#pragma once

#include <TNL/Benchmarks/Benchmark.h>

namespace TNL::Benchmarks {

template< typename Real, typename Device, typename Index, typename ResultReal = Real >
struct SpmvBenchmarkResult : public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using BenchmarkVector = Containers::Vector< ResultReal, Device, Index >;

   SpmvBenchmarkResult( const HostVector& csrResult, const BenchmarkVector& benchmarkResult )
   : csrResult( csrResult ),
     benchmarkResult( benchmarkResult )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements elements = BenchmarkResult::getTableHeader();
      elements.emplace_back( "CSR Diff.Max" );
      elements.emplace_back( "CSR Diff.L2" );
      return elements;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      HostVector benchmarkResultCopy;
      benchmarkResultCopy = benchmarkResult;
      auto diff = csrResult - benchmarkResultCopy;
      RowElements elements = BenchmarkResult::getRowElements();
      elements << max( abs( diff ) ) << lpNorm( diff, 2.0 );
      return elements;
   }

   const HostVector& csrResult;
   const BenchmarkVector& benchmarkResult;
};

}  // namespace TNL::Benchmarks
