// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <cstdint>

#include "SpmvBenchmarkResult.h"
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Benchmarks/JsonLogging.h>

#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Matrices/MatrixInfo.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Matrices/SparseMatrix.h>

using namespace TNL::Matrices;

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark< JsonLogging >;

/////
// Segments aliases
//
template< typename Device, typename Index, typename IndexAllocator >
using CSRSegments = Algorithms::Segments::CSR< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using AdaptiveCSRSegments = Algorithms::Segments::AdaptiveCSR< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

/////
// Main benchmarking
//
template< typename Real,
          typename Device,
          typename Index,
          typename InputMatrix,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class SegmentsType,
          typename TestValue,
          typename MatrixType = TNL::Matrices::GeneralMatrix >
void
benchmarkSpMVWithDevice( BenchmarkType& benchmark,
                         const InputMatrix& inputMatrix,
                         const TNL::Containers::Vector< Real, Devices::Host, Index >& csrResultVector,
                         const String& inputFileName,
                         const Config::ParameterContainer& parameters,
                         bool verboseMR )
{
   using DeviceMatrix = Matrices::SparseMatrix< TestValue, Device, Index, MatrixType, SegmentsType, Real >;
   using DeviceVector = Containers::Vector< Real, Device, Index >;

   DeviceMatrix deviceMatrix;
   try {
      deviceMatrix = inputMatrix;
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String( e.what() ) );
      return;
   }
   DeviceVector deviceInVector( deviceMatrix.getColumns() ), deviceOutVector( deviceMatrix.getRows() );

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( deviceMatrix.getSegments() ) ) {
      benchmark.setMetadataElement( { "format", MatrixInfo< DeviceMatrix >::getFormat() } );
      benchmark.setMetadataElement( { "launch cfg.", tag } );

      auto resetDeviceVectors = [ & ]()
      {
         deviceInVector = 1.0;
         deviceOutVector = 0.0;
      };

      auto launch_config_ = launch_config;  // Just to avoid warning 'captured structured bindings are a C++20 extension'
      auto spmvDevice = [ & ]()
      {
         deviceMatrix.vectorProduct( deviceInVector, deviceOutVector, launch_config_ );
      };
      SpmvBenchmarkResult< Real, Device, Index > deviceBenchmarkResults( csrResultVector, deviceOutVector );
      benchmark.time< Device >( resetDeviceVectors, "GPU", spmvDevice, deviceBenchmarkResults );
   }
}

template< typename Real,
          typename Index,
          typename InputMatrix,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class SegmentsType,
          typename TestValue,
          typename MatrixType = TNL::Matrices::GeneralMatrix >
void
benchmarkSpMV( BenchmarkType& benchmark,
               const InputMatrix& inputMatrix,
               const TNL::Containers::Vector< Real, Devices::Host, Index >& csrResultVector,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   // TODO: Host should be benchmarked as other devices, i.e. with
   //       launch configurations when it allows changing the number of OMP
   //       threads.
   using TestMatrix = Matrices::SparseMatrix< TestValue, TNL::Devices::Host, Index, MatrixType, SegmentsType, Real >;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   benchmark.setMetadataElement( { "format", MatrixInfo< TestMatrix >::getFormat() } );
   benchmark.setMetadataElement( { "launch cfg.", "" } );

   TestMatrix hostMatrix;
   try {
      hostMatrix = inputMatrix;
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String( e.what() ) );
      return;
   }

   const Index nonzeros = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( Index ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   // Benchmark SpMV on host
   if( allCpuTests || TestMatrix::isBinary() || TestMatrix::isSymmetric() ) {
      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [ & ]()
      {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [ & ]()
      {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );
      };
      SpmvBenchmarkResult< Real, Devices::Host, Index > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

#ifdef __CUDACC__
   // Benchmark SpMV on CUDA
   benchmarkSpMVWithDevice< Real, Devices::Cuda, Index, InputMatrix, SegmentsType, TestValue, MatrixType >(
      benchmark, inputMatrix, csrResultVector, inputFileName, parameters, verboseMR );
#endif

#ifdef __HIP__
   // Benchmark SpMV on HIP
   benchmarkSpMVWithDevice< Real, Devices::Hip, Index, InputMatrix, SegmentsType, TestValue, MatrixType >(
      benchmark, inputMatrix, csrResultVector, inputFileName, parameters, verboseMR );
#endif
}

template< typename Real, typename Index, typename TestValue, typename MatrixType, typename InputMatrix >
void
dispatchSpMV( BenchmarkType& benchmark,
              const InputMatrix& hostMatrix,
              const TNL::Containers::Vector< Real, Devices::Host, Index >& hostOutVector,
              const String& inputFileName,
              const Config::ParameterContainer& parameters,
              bool verboseMR )
{
   benchmarkSpMV< Real, Index, InputMatrix, CSRSegments, TestValue, MatrixType >(
      benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMV< Real, Index, InputMatrix, AdaptiveCSRSegments, TestValue, MatrixType >(
      benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   if( parameters.getParameter< bool >( "with-ellpack-formats" ) ) {
      benchmarkSpMV< Real, Index, InputMatrix, EllpackSegments, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, SlicedEllpackSegments, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ChunkedEllpackSegments, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, BiEllpackSegments, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   }
}

template< typename Real, typename Index, typename InputMatrix >
void
dispatchGeneral( BenchmarkType& benchmark,
                 const InputMatrix& hostMatrix,
                 const TNL::Containers::Vector< Real, Devices::Host, Index >& hostOutVector,
                 const String& inputFileName,
                 const Config::ParameterContainer& parameters,
                 bool verboseMR )
{
   dispatchSpMV< Real, Index, Real, TNL::Matrices::GeneralMatrix >(
      benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );

   // Dispatch binary SpMV for general matrices
   dispatchSpMV< Real, Index, bool, TNL::Matrices::GeneralMatrix >(
      benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
}

template< typename Real, typename Index >
void
dispatchSymmetric( BenchmarkType& benchmark,
                   const TNL::Containers::Vector< Real, Devices::Host, Index >& hostOutVector,
                   const String& inputFileName,
                   const Config::ParameterContainer& parameters,
                   bool verboseMR )
{
   using SymmetricInputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, Index, TNL::Matrices::SymmetricMatrix >;
   SymmetricInputMatrix symmetricHostMatrix;
   try {
      TNL::Matrices::MatrixReader< SymmetricInputMatrix >::readMtx( inputFileName, symmetricHostMatrix, verboseMR );
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( "Unable to read the symmetric matrix: " + String( e.what() ) );
      return;
   }
   compressSparseMatrix( symmetricHostMatrix );
   dispatchSpMV< Real, Index, Real, TNL::Matrices::SymmetricMatrix >(
      benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );

   // Dispatch binary SpMV for general matrices
   dispatchSpMV< Real, Index, bool, TNL::Matrices::SymmetricMatrix >(
      benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
}

template< typename Real = double, typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   using CSRHostMatrix = Matrices::SparseMatrix< Real, TNL::Devices::Host, Index >;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;

   ////
   // Set-up benchmark datasize
   //
   CSRHostMatrix csrHostMatrix;
   MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
   const Index uncompressedSize = csrHostMatrix.getValues().getSize();
   TNL::Matrices::compressSparseMatrix( csrHostMatrix );
   const Index compressedSize = csrHostMatrix.getValues().getSize();
   std::cout << "Compression ratio: " << (double) uncompressedSize / compressedSize << std::endl;
   const Index nonzeros = csrHostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( Index ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   ////
   // Nonzero elements per row statistics
   //
   TNL::Containers::Vector< Index > nonzerosPerRow( csrHostMatrix.getRows() );
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
   benchmark.setMetadataColumns( {
      { "matrix name", inputFileName },
      { "precision", getType< Real >() },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "nonzeros", convertToString( nonzeros ) },
      { "nonzeros per row std_dev", convertToString( std_dev ) },
      { "nonzeros per row percentile 25", convertToString( percentile_25 ) },
      { "nonzeros per row percentile 50", convertToString( percentile_50 ) },
      { "nonzeros per row percentile 75", convertToString( percentile_75 ) },
      { "format", "" },
      { "launch cfg.", "" },
      { "threads", "1" }  // NOTE: 'nonzeros per row average' can be easily
                          // calculated with Pandas based on the other metadata
   } );
   benchmark.setMetadataWidths( {
      { "matrix name", 32 },
      { "format", 32 },
      { "launch cfg.", 40 },
      { "threads", 5 },
   } );

   HostVector hostInVector( csrHostMatrix.getColumns() ), hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [ & ]()
   {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvCSRHost = [ & ]()
   {
      csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   // TODO: Move the following benchmarking to benchmarkSpMV function
   // when we can set the number of threads for the host in the
   // launch configuration.
   SpmvBenchmarkResult< Real, Devices::Host, Index > csrBenchmarkResults( hostOutVector, hostOutVector );
#ifdef HAVE_OPENMP
   const int maxThreadsCount = Devices::Host::getMaxThreadsCount();
#else
   const int maxThreadsCount = 1;
#endif
   int threads = 1;
   while( true ) {
      std::cout << "Benchmarking with " << threads << " threads." << std::endl;
      benchmark.setMetadataElement( { "format", "CSR" } );
      benchmark.setMetadataElement( { "threads", convertToString( threads ).getString() } );
      Devices::Host::setMaxThreadsCount( threads );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvCSRHost, csrBenchmarkResults );
      if( threads == maxThreadsCount )
         break;
      threads = min( 2 * threads, maxThreadsCount );
   }

   dispatchGeneral< Real, Index >( benchmark, csrHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );

   if( parameters.getParameter< bool >( "with-symmetric-matrices" ) )
      dispatchSymmetric< Real, Index >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
}

}  // namespace TNL::Benchmarks::SpMV
