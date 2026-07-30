// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <cstdint>

#include "SpmvBenchmarkResult.h"
#include <TNL/Benchmarks/Benchmark.h>

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

#include "MatrixStatistics.h"

using namespace TNL::Matrices;

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark;

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
using RowMajorSlicedEllpackSegments_SliceSize_2 =
   Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 2 >;

template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpackSegments_SliceSize_4 =
   Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 4 >;

template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpackSegments_SliceSize_8 =
   Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 8 >;

template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpackSegments_SliceSize_16 =
   Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 16 >;

template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpackSegments_SliceSize_32 =
   Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 32 >;

template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpackSegments_SliceSize_2 =
   Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 2 >;

template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpackSegments_SliceSize_4 =
   Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 4 >;

template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpackSegments_SliceSize_8 =
   Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 8 >;

template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpackSegments_SliceSize_16 =
   Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 16 >;

template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpackSegments_SliceSize_32 =
   Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 32 >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorSlicedEllpackSegments_SliceSize_2 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 2 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorSlicedEllpackSegments_SliceSize_4 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 4 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorSlicedEllpackSegments_SliceSize_8 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 8 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorSlicedEllpackSegments_SliceSize_16 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 16 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedRowMajorSlicedEllpackSegments_SliceSize_32 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::RowMajorSlicedEllpack< Device, Index, IndexAllocator, 32 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorSlicedEllpackSegments_SliceSize_2 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 2 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorSlicedEllpackSegments_SliceSize_4 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 4 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorSlicedEllpackSegments_SliceSize_8 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 8 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorSlicedEllpackSegments_SliceSize_16 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 16 > >;

template< typename Device, typename Index, typename IndexAllocator >
using SortedColumnMajorSlicedEllpackSegments_SliceSize_32 =
   Algorithms::Segments::SortedSegments< Algorithms::Segments::ColumnMajorSlicedEllpack< Device, Index, IndexAllocator, 32 > >;

/////
// SigmaSparseMatrix
//
// TNL::Algorithms::Segments::SortedSegments sorts segments in blocks of `sigma` elements
// (sigma == -1 sorts all segments together). `sigma` is a runtime property (setSigma()), not
// a template parameter, and it must be set on the matrix's segments *before* setRowCapacities()
// triggers the sort. SparseMatrix::segments is protected and its operator= does not propagate
// sigma from the source object, so the only way to configure it from outside SparseMatrix is a
// subclass that sets it right after construction, before the matrix gets assigned/populated.
//
// NOTE: this is a workaround for a TNL API design gap, not our preferred design -- there is
// currently no way to configure sigma through SparseMatrix's public API at all. In particular,
// matrix.getSegments().setSigma(...) looks like it should work but silently doesn't: getSegments()
// returns a *view*, and SortedSegments::getView() constructs that view without ever passing sigma
// to it, so the view's sigma is always its own independent default (-1), completely disconnected
// from the owning segments' sigma that setSegmentsSizes() actually reads during the sort.
// Tracked upstream: https://gitlab.com/tnl-project/tnl/-/work_items/233
//
template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments,
          typename ComputeReal = std::conditional_t< std::is_same_v< Real, bool >, Index, Real >,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class SigmaSparseMatrix
: public Matrices::SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >
{
public:
   using Base =
      Matrices::SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >;
   using Base::Base;
   using Base::operator=;

   // Only meaningful (and only compiles) when Segments is based on TNL::Algorithms::Segments::SortedSegments.
   void
   setSigma( Index sigma )
   {
      this->segments.setSigma( sigma );
   }
};

// Renames the "Sorted" prefix produced by SortedSegments::getSegmentsType() to make the sigma
// value used for a particular benchmark run visible in the log, e.g. "Sorted CSR" with sigma=256
// becomes "Sigma-256-Sorted CSR". No-op when sigma is not a positive block size (i.e. sigma == -1,
// meaning all segments were sorted together, keeps the plain "Sorted" name).
template< typename Index >
std::string
applySigmaToFormatName( std::string format, Index sigma )
{
   if( sigma <= 0 )
      return format;
   const std::string needle = "Sorted";
   const auto pos = format.find( needle );
   if( pos != std::string::npos )
      format.replace( pos, needle.size(), "Sigma-" + std::to_string( sigma ) + "-Sorted" );
   return format;
}

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
                         bool verboseMR,
                         Index sigma = -1 )
{
   using DeviceMatrix = SigmaSparseMatrix< TestValue, Device, Index, MatrixType, SegmentsType, Real >;
   using DeviceVector = Containers::Vector< Real, Device, Index >;

#ifdef __CUDACC__
   const char* deviceName = "CUDA";
#elif defined( __HIP__ )
   const char* deviceName = "HIP";
#else
   const char* deviceName = "Unknown device";
#endif

   DeviceMatrix deviceMatrix;
   if constexpr( Algorithms::Segments::isSortedSegments_v< typename DeviceMatrix::SegmentsType > )
      if( sigma > 0 )
         deviceMatrix.setSigma( sigma );
   try {
      deviceMatrix = inputMatrix;
   }
   catch( const std::exception& e ) {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String( e.what() ) );
      return;
   }
   DeviceVector deviceInVector( deviceMatrix.getColumns() ), deviceOutVector( deviceMatrix.getRows() );

   for( auto [ launch_config, tag ] : TNL::Algorithms::Segments::reductionLaunchConfigurations( deviceMatrix.getSegments() ) ) {
      benchmark.setMetadataElement(
         { "format", applySigmaToFormatName( MatrixInfo< typename DeviceMatrix::Base >::getFormat(), sigma ) } );
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
      benchmark.time< Device >( resetDeviceVectors, deviceName, spmvDevice, deviceBenchmarkResults );
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
               bool verboseMR,
               Index sigma = -1 )
{
   // TODO: Host should be benchmarked as other devices, i.e. with
   //       launch configurations when it allows changing the number of OMP
   //       threads.
   using TestMatrix = SigmaSparseMatrix< TestValue, TNL::Devices::Host, Index, MatrixType, SegmentsType, Real >;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   benchmark.setMetadataElement(
      { "format", applySigmaToFormatName( MatrixInfo< typename TestMatrix::Base >::getFormat(), sigma ) } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );

   TestMatrix hostMatrix;
   if constexpr( Algorithms::Segments::isSortedSegments_v< typename TestMatrix::SegmentsType > )
      if( sigma > 0 )
         hostMatrix.setSigma( sigma );
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
      benchmark, inputMatrix, csrResultVector, inputFileName, parameters, verboseMR, sigma );
#endif

#ifdef __HIP__
   // Benchmark SpMV on HIP
   benchmarkSpMVWithDevice< Real, Devices::Hip, Index, InputMatrix, SegmentsType, TestValue, MatrixType >(
      benchmark, inputMatrix, csrResultVector, inputFileName, parameters, verboseMR, sigma );
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
      benchmarkSpMV< Real, Index, InputMatrix, RowMajorSlicedEllpackSegments_SliceSize_2, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, RowMajorSlicedEllpackSegments_SliceSize_4, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, RowMajorSlicedEllpackSegments_SliceSize_8, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, RowMajorSlicedEllpackSegments_SliceSize_16, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, RowMajorSlicedEllpackSegments_SliceSize_32, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ColumnMajorSlicedEllpackSegments_SliceSize_2, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ColumnMajorSlicedEllpackSegments_SliceSize_4, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ColumnMajorSlicedEllpackSegments_SliceSize_8, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ColumnMajorSlicedEllpackSegments_SliceSize_16, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ColumnMajorSlicedEllpackSegments_SliceSize_32, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, ChunkedEllpackSegments, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, Index, InputMatrix, BiEllpackSegments, TestValue, MatrixType >(
         benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   }
   if( parameters.getParameter< bool >( "with-sorted-segments" ) ) {
      // sigma == -1 sorts all segments together (the historical "Sorted" behavior); sigma == 256
      // sorts within blocks of 256 segments instead, see with-sigma-256-sorted-segments.
      auto benchmarkSortedFormats = [ & ]( Index sigma )
      {
         benchmarkSpMV< Real, Index, InputMatrix, Algorithms::Segments::SortedCSR, TestValue, MatrixType >(
            benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
         benchmarkSpMV< Real, Index, InputMatrix, Algorithms::Segments::SortedAdaptiveCSR, TestValue, MatrixType >(
            benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
         if( parameters.getParameter< bool >( "with-ellpack-formats" ) ) {
            benchmarkSpMV< Real, Index, InputMatrix, Algorithms::Segments::SortedEllpack, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, SortedRowMajorSlicedEllpackSegments_SliceSize_2, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, SortedRowMajorSlicedEllpackSegments_SliceSize_4, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, SortedRowMajorSlicedEllpackSegments_SliceSize_8, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, SortedRowMajorSlicedEllpackSegments_SliceSize_16, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, SortedRowMajorSlicedEllpackSegments_SliceSize_32, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real,
                           Index,
                           InputMatrix,
                           SortedColumnMajorSlicedEllpackSegments_SliceSize_2,
                           TestValue,
                           MatrixType >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real,
                           Index,
                           InputMatrix,
                           SortedColumnMajorSlicedEllpackSegments_SliceSize_4,
                           TestValue,
                           MatrixType >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real,
                           Index,
                           InputMatrix,
                           SortedColumnMajorSlicedEllpackSegments_SliceSize_8,
                           TestValue,
                           MatrixType >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real,
                           Index,
                           InputMatrix,
                           SortedColumnMajorSlicedEllpackSegments_SliceSize_16,
                           TestValue,
                           MatrixType >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real,
                           Index,
                           InputMatrix,
                           SortedColumnMajorSlicedEllpackSegments_SliceSize_32,
                           TestValue,
                           MatrixType >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, Algorithms::Segments::SortedChunkedEllpack, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
            benchmarkSpMV< Real, Index, InputMatrix, Algorithms::Segments::SortedBiEllpack, TestValue, MatrixType >(
               benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR, sigma );
         }
      };
      benchmarkSortedFormats( -1 );
      if( parameters.getParameter< bool >( "with-sigma-256-sorted-segments" ) )
         benchmarkSortedFormats( 256 );
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
   const double datasetSize = (double) ( nonzeros * ( 2 * sizeof( Real ) + sizeof( Index ) )
                                         + ( csrHostMatrix.getRows() + csrHostMatrix.getColumns() ) * sizeof( Index ) )
                            / oneGB;
   benchmark.setDatasetSize( datasetSize );

   ////
   // Nonzero elements per row statistics
   //
   TNL::Containers::Vector< Index > nonzerosPerRow( csrHostMatrix.getRows() );
   TNL::Containers::Vector< double > aux;
   csrHostMatrix.getCompressedRowLengths( nonzerosPerRow );
   MatrixStatistics< CSRHostMatrix > matrixStats( csrHostMatrix );

   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   benchmark.setMetadataColumns( { { "matrix name", inputFileName },
                                   { "precision", getType< Real >() },
                                   { "rows", convertToString( csrHostMatrix.getRows() ) },
                                   { "columns", convertToString( csrHostMatrix.getColumns() ) },
                                   { "nonzeros", convertToString( nonzeros ) },
                                   { "nonzeros/row average", convertToString( matrixStats.getAverage() ) },
                                   { "nonzeros/row std_dev", convertToString( matrixStats.getStddev() ) },
                                   { "nonzeros/row P50", convertToString( matrixStats.getP50() ) },
                                   { "nonzeros/row P95", convertToString( matrixStats.getP95() ) },
                                   { "nonzeros/row P99", convertToString( matrixStats.getP99() ) },
                                   { "nonzeros/row R50", convertToString( matrixStats.getR50() ) },
                                   { "nonzeros/row R95", convertToString( matrixStats.getR95() ) },
                                   { "nonzeros/row R99", convertToString( matrixStats.getR99() ) },
                                   { "warps/row average", convertToString( matrixStats.getAverageWarpsPerRow() ) },
                                   { "rho ELL", convertToString( matrixStats.getRhoELL() ) },
                                   { "rho SELL 2", convertToString( matrixStats.getRhoSELL()[ 0 ] ) },
                                   { "rho SELL 4", convertToString( matrixStats.getRhoSELL()[ 1 ] ) },
                                   { "rho SELL 8", convertToString( matrixStats.getRhoSELL()[ 2 ] ) },
                                   { "rho SELL 16", convertToString( matrixStats.getRhoSELL()[ 3 ] ) },
                                   { "rho SELL 32", convertToString( matrixStats.getRhoSELL()[ 4 ] ) },
                                   { "eta SELL 2", convertToString( matrixStats.getEtaSELL()[ 0 ] ) },
                                   { "eta SELL 4", convertToString( matrixStats.getEtaSELL()[ 1 ] ) },
                                   { "eta SELL 8", convertToString( matrixStats.getEtaSELL()[ 2 ] ) },
                                   { "eta SELL 16", convertToString( matrixStats.getEtaSELL()[ 3 ] ) },
                                   { "eta SELL 32", convertToString( matrixStats.getEtaSELL()[ 4 ] ) },
                                   { "entropy", convertToString( matrixStats.getEntropy() ) },
                                   { "Gini", convertToString( matrixStats.getGini() ) },
                                   { "format", "" },
                                   { "launch cfg.", "" },
                                   { "threads", "1" } } );

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
      benchmark.setMetadataElement( { "format", "CSR" } );
      auto launch_config = convertToString( threads ) + " threads";
      if( threads == 1 )
         launch_config = "1 thread";
      benchmark.setMetadataElement( { "launch cfg.", launch_config.getString() } );
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
