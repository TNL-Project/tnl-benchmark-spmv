#pragma once

#include <cstdint>

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Benchmarks/JsonLogging.h>
#include "SpmvBenchmarkResult.h"

#include "ReferenceFormats/Legacy/MatrixInfo.h"
#include "ReferenceFormats/Legacy/CSR.h"
#include "ReferenceFormats/Legacy/Ellpack.h"
#include "ReferenceFormats/Legacy/SlicedEllpack.h"
#include "ReferenceFormats/Legacy/ChunkedEllpack.h"
#include "ReferenceFormats/Legacy/AdEllpack.h"
#include "ReferenceFormats/Legacy/BiEllpack.h"
#include "ReferenceFormats/Legacy/LegacyMatrixReader.h"

#include <TNL/Algorithms/sort.h>

using namespace TNL::Matrices;

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark< JsonLogging >;

/////
// Legacy formats
//
template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Scalar = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Vector = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light2 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight2 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light3 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight3 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light4 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight4 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light5 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight5 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light6 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight6 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Adaptive = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRAdaptive >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_MultiVector = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRMultiVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_LightWithoutAtomic = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLightWithoutAtomic >;

template< typename Real, typename Device, typename Index >
using SlicedEllpackAlias = Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index >;

template< typename Real,
          template< typename, typename, typename > class Matrix >
void
benchmarkSpMVLegacy( BenchmarkType& benchmark,
                     const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                     const String& inputFileName,
                     const Config::ParameterContainer& parameters,
                     bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() });

   HostMatrix hostMatrix;
   CudaMatrix cudaMatrix;

   try
   {
      SpMV::ReferenceFormats::Legacy::LegacyMatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix, verboseMR );
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to read the matrix:" + String(e.what()) );
      return;
   }

   const int nonzeros = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   if( allCpuTests )
   {
      HostVector hostInVector( hostMatrix.getColumns() );
      HostVector hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef __CUDACC__
   try
   {
      cudaMatrix = hostMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;
   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real >
void
dispatchLegacy( BenchmarkType& benchmark,
                const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                const String& inputFileName,
                const Config::ParameterContainer& parameters,
                bool verboseMR )
{
   using namespace Benchmarks::SpMV::ReferenceFormats;
   bool withEllpack = parameters.getParameter< bool >( "with-ellpack-formats" );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Scalar             >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Vector             >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light              >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light2             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light3             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light4             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light5             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light6             >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Adaptive           >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_MultiVector        >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_LightWithoutAtomic >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
   if( withEllpack )
   {
      benchmarkSpMVLegacy< Real, Legacy::Ellpack                           >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMVLegacy< Real, SlicedEllpackAlias                        >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::ChunkedEllpack                    >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::BiEllpack                         >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
      // AdEllpack is broken
      //benchmarkSpMV< Real, Matrices::AdEllpack              >( benchmark, hostOutVector, inputFileName, verboseMR );
   }
}

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = SparseMatrixLegacy_CSR_Scalar< Real, TNL::Devices::Host, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   CSRHostMatrix csrHostMatrix;

   ////
   // Set-up benchmark datasize
   //
   SpMV::ReferenceFormats::Legacy::LegacyMatrixReader< CSRHostMatrix >::readMtxFile( inputFileName, csrHostMatrix, verboseMR );
   const int nonzeros = csrHostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   ////
   // Nonzero elements per row statistics
   //
   TNL::Containers::Vector< int > nonzerosPerRow( csrHostMatrix.getRows() );
   TNL::Containers::Vector< double > aux;
   csrHostMatrix.getCompressedRowLengths( nonzerosPerRow );
   double average = sum( nonzerosPerRow ) / nonzerosPerRow.getSize();
   aux = nonzerosPerRow - average;
   double std_dev = lpNorm( aux, 2.0 ) / nonzerosPerRow.getSize();
   TNL::Algorithms::ascendingSort( nonzerosPerRow );
   double percentile_25 = nonzerosPerRow[ nonzerosPerRow.getSize() * 0.25 ];
   double percentile_50 = nonzerosPerRow[ nonzerosPerRow.getSize() * 0.5 ];
   double percentile_75 = nonzerosPerRow[ nonzerosPerRow.getSize() * 0.75 ];


   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   benchmark.setMetadataColumns({
      { "matrix name", inputFileName },
      { "precision", getType< Real >() },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "nonzeros", convertToString( nonzeros ) },
      { "nonzeros per row std_dev", convertToString( std_dev ) },
      { "nonzeros per row percentile 25", convertToString( percentile_25 ) },
      { "nonzeros per row percentile 50", convertToString( percentile_50 ) },
      { "nonzeros per row percentile 75", convertToString( percentile_75 ) }
      // NOTE: 'nonzeros per row average' can be easily calculated with Pandas based on the other metadata
   });
   benchmark.setMetadataWidths({
      { "matrix name", 32 },
      { "format", 46 },
      { "threads", 5 },
   });

   HostVector hostInVector( csrHostMatrix.getColumns() );
   HostVector hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvCSRHost = [&]() {
       csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > csrBenchmarkResults( hostOutVector, hostOutVector );
   const int maxThreadsCount = Devices::Host::getMaxThreadsCount();
   int threads = 1;
   while( true ) {
      benchmark.setMetadataElement({ "format", "CSR" });
      benchmark.setMetadataElement({ "threads", convertToString( threads ).getString() });
      Devices::Host::setMaxThreadsCount( threads );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvCSRHost, csrBenchmarkResults );
      if( threads == maxThreadsCount )
         break;
      threads = min( 2 * threads, maxThreadsCount );
   }

   csrHostMatrix.reset();

   /////
   // Benchmarking of TNL legacy formats
   //
   dispatchLegacy< Real >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
}

} // namespace TNL::Benchmarks::SpMV
