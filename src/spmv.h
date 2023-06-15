// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <cstdint>

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Benchmarks/JsonLogging.h>
#include "SpmvBenchmarkResult.h"

#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixInfo.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRScalarKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRVectorKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRHybridKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRLightKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRAdaptiveKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/EllpackKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/SlicedEllpackKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/ChunkedEllpackKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/BiEllpackKernel.h>
#include <TNL/Algorithms/sort.h>

// Uncomment the following line to enable benchmarking the sandbox sparse matrix.
//#define WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#endif

using namespace TNL::Matrices;

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark< JsonLogging >;

/////
// Segments aliases
//
template< typename Device, typename Index, typename IndexAllocator >
using CSRSegments = Algorithms::Segments::CSR< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

/////
// General sparse matrix aliases
//
template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, CSRSegments >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, EllpackSegments >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, SlicedEllpackSegments >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_ChunkedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, ChunkedEllpackSegments >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_BiEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, BiEllpackSegments >;

/////
// Symmetric sparse matrix aliases
//
template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, CSRSegments >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, EllpackSegments >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, SlicedEllpackSegments >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_ChunkedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, ChunkedEllpackSegments >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_BiEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, BiEllpackSegments >;

#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
template< typename Real, typename Device, typename Index >
using SparseSandboxMatrix = Matrices::Sandbox::SparseSandboxMatrix< Real, Device, Index, Matrices::GeneralMatrix >;
#endif

/////
// SpMV kernels
template< typename Device, typename Index >
using CSRScalarKernel = Algorithms::SegmentsReductionKernels::CSRScalarKernel< Index, Device >;

template< typename Device, typename Index >
using CSRVectorKernel = Algorithms::SegmentsReductionKernels::CSRVectorKernel< Index, Device >;

template< typename Device, typename Index >
using CSRHybridKernel = Algorithms::SegmentsReductionKernels::CSRHybridKernel< Index, Device >;

template< typename Device, typename Index >
using CSRLightKernel = Algorithms::SegmentsReductionKernels::CSRLightKernel< Index, Device >;

template< typename Device, typename Index >
using CSRAdaptiveKernel = Algorithms::SegmentsReductionKernels::CSRAdaptiveKernel< Index, Device >;

template< typename Device, typename Index >
using EllpackKernel = Algorithms::SegmentsReductionKernels::EllpackKernel< Index, Device >;

template< typename Device, typename Index >
using SlicedEllpackKernel = Algorithms::SegmentsReductionKernels::SlicedEllpackKernel< Index, Device >;

template< typename Device, typename Index >
using ChunkedEllpackKernel = Algorithms::SegmentsReductionKernels::ChunkedEllpackKernel< Index, Device >;

template< typename Device, typename Index >
using BiEllpackKernel = Algorithms::SegmentsReductionKernels::BiEllpackKernel< Index, Device >;

template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix,
          template< typename, typename > class Kernel >
void
benchmarkSpMV( BenchmarkType& benchmark,
               const InputMatrix& inputMatrix,
               const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using HostKernel = Kernel< TNL::Devices::Host, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() + " " + HostKernel::getKernelType() });

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String(e.what()) );
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
      HostKernel kernel;
      kernel.init( hostMatrix.getSegments() );

      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector, kernel );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef __CUDACC__
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using CudaKernel = Kernel< TNL::Devices::Cuda, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   CudaMatrix cudaMatrix;
   try
   {
      cudaMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaKernel kernel;
   kernel.init( cudaMatrix.getSegments() );

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector, kernel );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix,
          template< typename, typename > class Kernel,
          typename TestReal = Real >
void
benchmarkSpMVCSRLight( BenchmarkType& benchmark,
                       const InputMatrix& inputMatrix,
                       const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                       const String& inputFileName,
                       const Config::ParameterContainer& parameters,
                       bool verboseMR )
{
   using HostMatrix = Matrix< TestReal, TNL::Devices::Host, int >;
   using HostKernel = Kernel< TNL::Devices::Host, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() + " " + HostKernel::getKernelType() });

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String(e.what()) );
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
      HostKernel kernel;
      kernel.init( hostMatrix.getSegments() );

      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector, kernel );
      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef __CUDACC__
   using CudaMatrix = Matrix< TestReal, TNL::Devices::Cuda, int >;
   using CudaKernel = Kernel< TNL::Devices::Cuda, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   CudaMatrix cudaMatrix;
   try
   {
      cudaMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaKernel kernel;
   kernel.init( cudaMatrix.getSegments() );

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector, kernel );
   };

   {
      kernel.setThreadsMapping( Algorithms::SegmentsReductionKernels::CSRLightAutomaticThreads );
      String format = MatrixInfo< HostMatrix >::getFormat() + " " + HostKernel::getKernelType() + " Automatic";
      benchmark.setMetadataElement({ "format", format });

      SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
   };

   {
      kernel.setThreadsMapping( Algorithms::SegmentsReductionKernels::CSRLightAutomaticThreadsLightSpMV );
      String format = MatrixInfo< HostMatrix >::getFormat() + " " + HostKernel::getKernelType() + " Automatic Light";
      benchmark.setMetadataElement({ "format", format });

      SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
   };

   for( auto threadsPerRow : std::vector< int >{ 1, 2, 4, 8, 16, 32, 64, 128 } )
   {
      kernel.setThreadsPerSegment( threadsPerRow );
      String format = MatrixInfo< HostMatrix >::getFormat() + " " + HostKernel::getKernelType() + " " + std::to_string( threadsPerRow );
      benchmark.setMetadataElement({ "format", format });

      SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
   }
 #endif
}


template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix,
          template< typename, typename > class Kernel >
void
benchmarkBinarySpMV( BenchmarkType& benchmark,
                     const InputMatrix& inputMatrix,
                     const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                     const String& inputFileName,
                     const Config::ParameterContainer& parameters,
                     bool verboseMR )
{
   using HostMatrix = Matrix< bool, TNL::Devices::Host, int >;
   using HostKernel = Kernel< TNL::Devices::Host, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() + " " + HostKernel::getKernelType() });

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String(e.what()) );
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
      HostKernel kernel;
      kernel.init( hostMatrix.getSegments() );

      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector, kernel );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef __CUDACC__
   using CudaMatrix = Matrix< bool, TNL::Devices::Cuda, int >;
   using CudaKernel = Kernel< TNL::Devices::Cuda, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   CudaMatrix cudaMatrix;
   try
   {
      cudaMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaKernel kernel;
   kernel.init( cudaMatrix.getSegments() );

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector, kernel );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real, typename HostMatrix >
void
dispatchBinary( BenchmarkType& benchmark,
                const HostMatrix& hostMatrix,
                const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                const String& inputFileName,
                const Config::ParameterContainer& parameters,
                bool verboseMR )
{
   bool withEllpack = parameters.getParameter< bool >( "with-ellpack-formats" );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_CSR, CSRScalarKernel              >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_CSR, CSRVectorKernel              >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVCSRLight< Real, HostMatrix, SparseMatrix_CSR, CSRLightKernel, bool       >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_CSR, CSRAdaptiveKernel            >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   if( withEllpack )
   {
      benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_Ellpack, EllpackKernel               >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_SlicedEllpack, SlicedEllpackKernel   >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_ChunkedEllpack, ChunkedEllpackKernel >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_BiEllpack, BiEllpackKernel           >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   }
}

template< typename Real >
void
dispatchSpMV( BenchmarkType& benchmark,
              const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
              const String& inputFileName,
              const Config::ParameterContainer& parameters,
              bool verboseMR )
{
   using HostMatrixType = SparseMatrix_CSR< Real, TNL::Devices::Host, int >;
   bool withEllpack = parameters.getParameter< bool >( "with-ellpack-formats" );
   HostMatrixType hostMatrix;
   TNL::Matrices::MatrixReader< HostMatrixType >::readMtx( inputFileName, hostMatrix, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR, CSRScalarKernel                   >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR, CSRVectorKernel                   >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   //benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR, CSRHybridKernel                   >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVCSRLight< Real, HostMatrixType, SparseMatrix_CSR, CSRLightKernel            >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR, CSRAdaptiveKernel                 >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   if( withEllpack )
   {
      benchmarkSpMV< Real, HostMatrixType, SparseMatrix_Ellpack, EllpackKernel               >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, HostMatrixType, SparseMatrix_SlicedEllpack, SlicedEllpackKernel   >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, HostMatrixType, SparseMatrix_ChunkedEllpack, ChunkedEllpackKernel >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, HostMatrixType, SparseMatrix_BiEllpack, BiEllpackKernel           >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   }
   dispatchBinary< Real >( benchmark, hostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
   benchmarkSpMV< Real, HostMatrixType, SparseSandboxMatrix                       >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
#endif
}

template< typename Real, typename SymmetricInputMatrix >
void
dispatchSymmetricBinary( BenchmarkType& benchmark,
                         const SymmetricInputMatrix& symmetricHostMatrix,
                         const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                         const String& inputFileName,
                         const Config::ParameterContainer& parameters,
                         bool verboseMR )
{
   bool withEllpack = parameters.getParameter< bool >( "with-ellpack-formats" );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRScalarKernel              >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRVectorKernel              >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   //benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRHybridKernel            >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVCSRLight< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRLightKernel, bool       >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRAdaptiveKernel            >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   if( withEllpack )
   {
      benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_Ellpack, EllpackKernel               >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_SlicedEllpack, SlicedEllpackKernel   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_ChunkedEllpack, ChunkedEllpackKernel >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_BiEllpack, BiEllpackKernel           >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   }
}

template< typename Real >
void
dispatchSymmetric( BenchmarkType& benchmark,
                   const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                   const String& inputFileName,
                   const Config::ParameterContainer& parameters,
                   bool verboseMR )
{
   using SymmetricInputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >;
   //using InputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;
   //bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   bool withEllpack = parameters.getParameter< bool >( "with-ellpack-formats" );
   SymmetricInputMatrix symmetricHostMatrix;
   try
   {
      TNL::Matrices::MatrixReader< SymmetricInputMatrix >::readMtx( inputFileName, symmetricHostMatrix, verboseMR );
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to read the symmetric matrix: " + String(e.what()) );
      return;
   }
   //InputMatrix hostMatrix;
   //TNL::Matrices::MatrixReader< InputMatrix >::readMtx( inputFileName, hostMatrix, verboseMR );
   // TODO: Comparison of symmetric and general matrix does not work yet.
   //if( hostMatrix != symmetricHostMatrix )
   //{
   //   std::cerr << "ERROR: Symmetric matrices do not match !!!" << std::endl;
   //}
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRScalarKernel                    >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRVectorKernel                    >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   //benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRHybridKernel                   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMVCSRLight< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRLightKernel             >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR, CSRAdaptiveKernel                  >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   if( withEllpack )
   {
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_Ellpack, EllpackKernel               >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_SlicedEllpack, SlicedEllpackKernel   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_ChunkedEllpack, ChunkedEllpackKernel >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_BiEllpack, BiEllpackKernel           >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
   }
   dispatchSymmetricBinary< Real >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, parameters, verboseMR );
}

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   using CSRHostMatrix = SparseMatrix_CSR< Real, TNL::Devices::Host, Index >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   CSRHostMatrix csrHostMatrix;

   ////
   // Set-up benchmark datasize
   //
   MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
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

   HostVector hostInVector( csrHostMatrix.getColumns() ), hostOutVector( csrHostMatrix.getRows() );

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
   // Benchmarking TNL formats
   //
#if ! defined ( __CUDACC__ )
   if(  parameters.getParameter< bool >( "with-all-cpu-tests" ) )
      dispatchSpMV< Real >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
#else
   dispatchSpMV< Real >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
#endif

   /////
   // Benchmarking symmetric sparse matrices
   //
   if( parameters.getParameter< bool >("with-symmetric-matrices") )
      dispatchSymmetric< Real >( benchmark, hostOutVector, inputFileName, parameters, verboseMR );
}

} // namespace TNL::Benchmarks::SpMV
