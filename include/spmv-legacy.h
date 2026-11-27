#pragma once

#include <cstdint>

#include <TNL/Benchmarks/Benchmark.h>
#include "SpmvBenchmarkResult.h"
#include "CpuBenchmarking.h"
#include "benchmark-progress.h"

#include "Legacy/MatrixInfo.h"
#include "Legacy/CSR.h"
#include "Legacy/Ellpack.h"
#include "Legacy/SlicedEllpack.h"
#include "Legacy/ChunkedEllpack.h"
#include "Legacy/AdEllpack.h"
#include "Legacy/BiEllpack.h"
#include "Legacy/LegacyMatrixReader.h"

using namespace TNL::Matrices;

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark;

/////
// Legacy formats
//
template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Scalar = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Vector = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light =
   Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light2 = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight2 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light3 = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight3 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light4 = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight4 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light5 = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight5 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light6 = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight6 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Adaptive = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRAdaptive >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_MultiVector = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRMultiVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_LightWithoutAtomic = Benchmarks::SpMV::ReferenceFormats::Legacy::
   CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLightWithoutAtomic >;

template< typename Real, typename Device, typename Index >
using SlicedEllpackAlias = Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index >;

template< typename Real, template< typename, typename, typename > class Matrix >
void
benchmarkSpMVLegacy( BenchmarkType& benchmark,
                     BenchmarkProgress& progress,
                     const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                     const String& inputFileName,
                     const Config::ParameterContainer& parameters,
                     bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   const bool cpuTests =
      parameters.getParameter< bool >( "with-cpu-tests" ) && parameters.getParameter< bool >( "with-all-cpu-tests" );
   const std::string format = MatrixInfo< HostMatrix >::getFormat();
   benchmark.setMetadataElement( { "format", format } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );

   HostMatrix hostMatrix;
   CudaMatrix cudaMatrix;

   try {
      SpMV::ReferenceFormats::Legacy::LegacyMatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix, verboseMR );
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( "Unable to read the matrix:" + String( e.what() ) );
      return;
   }

   const int nonzeros = hostMatrix.getNonzeroElementsCount();
   const std::size_t datasetSize = static_cast< std::size_t >( nonzeros ) * ( 2 * sizeof( Real ) + sizeof( int ) );  // in bytes
   benchmark.setDatasetSize( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   if( cpuTests ) {
      HostVector hostInVector( hostMatrix.getColumns() );
      HostVector hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [ & ]()
      {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [ & ]()
      {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );
      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      forEachCpuThreadsCount( false,
                              [ & ]( const std::string& launchConfig )
                              {
                                 if( progress.isDone( getType< Real >(), format, "CPU", launchConfig ) )
                                    return;
                                 benchmark.setMetadataElement( { "launch cfg.", launchConfig } );
                                 benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
                                 progress.markDone( getType< Real >(), format, "CPU", launchConfig );
                              } );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef __CUDACC__
   try {
      cudaMatrix = hostMatrix;
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String( e.what() ) );
      return;
   }

   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;
   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [ & ]()
   {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [ & ]()
   {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   if( ! progress.isDone( getType< Real >(), format, "GPU", "Default" ) ) {
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
      progress.markDone( getType< Real >(), format, "GPU", "Default" );
   }
#endif
}

template< typename Real >
void
dispatchLegacy( BenchmarkType& benchmark,
                BenchmarkProgress& progress,
                const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                const String& inputFileName,
                const Config::ParameterContainer& parameters,
                bool verboseMR )
{
   using namespace Benchmarks::SpMV::ReferenceFormats;
   bool withEllpack = parameters.getParameter< bool >( "with-ellpack-formats" );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Scalar >(
      benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Vector >(
      benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light              >( benchmark, hostOutVector, inputFileName, verboseMR
   //); benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light2             >( benchmark, hostOutVector, inputFileName,
   //verboseMR ); benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light3             >( benchmark, hostOutVector,
   //inputFileName, verboseMR ); benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light4             >( benchmark,
   //hostOutVector, inputFileName, verboseMR ); benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light5             >(
   //benchmark, hostOutVector, inputFileName, verboseMR ); benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light6 >(
   //benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Adaptive >(
      benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_MultiVector >(
      benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_LightWithoutAtomic >(
      benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
   if( withEllpack ) {
      benchmarkSpMVLegacy< Real, Legacy::Ellpack >( benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMVLegacy< Real, SlicedEllpackAlias >(
         benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::ChunkedEllpack >(
         benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::BiEllpack >(
         benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
      // AdEllpack is broken
      //benchmarkSpMV< Real, Matrices::AdEllpack              >( benchmark, hostOutVector, inputFileName, verboseMR );
   }
}

template< typename Real = double, typename Index = int >
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
   const std::size_t datasetSize = static_cast< std::size_t >( nonzeros ) * ( 2 * sizeof( Real ) + sizeof( int ) );  // in bytes
   benchmark.setDatasetSize( datasetSize );

   String matrixName = parameters.getParameter< String >( "matrix-name" );
   if( matrixName.empty() )
      matrixName = inputFileName;

   const std::string progressDir = parameters.getParameter< String >( "progress-dir" ).getString();
   BenchmarkProgress progress( progressDir, matrixName, false );

   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   // Matrix dimensions and row-length statistics are not logged here anymore - they are written
   // once per matrix by tnl-benchmark-spmv-matrix-info and joined in by "matrix name" when the logs are parsed,
   // instead of being repeated on every record here.
   benchmark.setMetadataColumns( {
      { "matrix name", matrixName },
      { "precision", getType< Real >() },
      { "build", getBuildName() },
   } );

   HostVector hostInVector( csrHostMatrix.getColumns() );
   HostVector hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [ & ]()
   {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvCSRHost = [ & ]()
   {
      csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   // The result other formats are compared with is established once, unconditionally, as a
   // separate copy - decoupled from the timed/logged sweep below, so that skipping an
   // already-done combination on resume never leaves it uninitialized.
   resetHostVectors();
   spmvCSRHost();
   const HostVector csrResultVector( hostOutVector );

   if( parameters.getParameter< bool >( "with-cpu-tests" ) ) {
      SpmvBenchmarkResult< Real, Devices::Host, int > csrBenchmarkResults( csrResultVector, hostOutVector );
      const std::string format = "CSR Legacy";
      benchmark.setMetadataElement( { "format", format } );
      forEachCpuThreadsCount( true,
                              [ & ]( const std::string& launchConfig )
                              {
                                 if( progress.isDone( getType< Real >(), format, "CPU", launchConfig ) )
                                    return;
                                 benchmark.setMetadataElement( { "launch cfg.", launchConfig } );
                                 benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvCSRHost, csrBenchmarkResults );
                                 progress.markDone( getType< Real >(), format, "CPU", launchConfig );
                              } );
   }

   csrHostMatrix.reset();

   /////
   // Benchmarking of TNL legacy formats
   //
   dispatchLegacy< Real >( benchmark, progress, hostOutVector, inputFileName, parameters, verboseMR );
}

}  // namespace TNL::Benchmarks::SpMV
