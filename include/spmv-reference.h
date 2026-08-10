#pragma once

#include <cstdint>

#include <TNL/Benchmarks/Benchmark.h>
#include "SpmvBenchmarkResult.h"

#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixInfo.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRScalarKernel.h>
#include <TNL/Algorithms/sort.h>

#ifdef HAVE_PETSC
   #include <petscmat.h>
#endif

#ifdef HAVE_GINKGO
   #include <TNL/Containers/GinkgoVector.h>
   #include <TNL/Matrices/GinkgoOperator.h>
#endif

#ifdef HAVE_HYPRE
   #include <TNL/Hypre.h>
   #include <TNL/Matrices/HypreCSRMatrix.h>
#endif

#include "cusparseCSRMatrix.h"
#include "cusparseSlicedEllMatrix.h"
#include "hipsparseCSRMatrix.h"
#include "LightSpMVBenchmark.h"
#include "CSR5Benchmark.h"

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark;

#ifdef __CUDACC__
// cuSPARSE's Sliced ELLPACK matches TNL's column-major SlicedEllpack segments,
// see the comment in cusparseSlicedEllMatrix.h. Alias templates must live at
// namespace scope, so this cannot be declared inside benchmarkSpmv().
template< typename Device_, typename Index_, typename IndexAllocator_ >
using SlicedEllpackSegments = TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< Device_, Index_, IndexAllocator_ >;
#endif

// Runs the full reference-library benchmark suite (PETSc/HYPRE/Ginkgo/cuSPARSE/
// CSR5/LightSpMV/hipsparse, whichever are compiled in) for one already loaded
// matrix.
template< typename Real = double, typename Index = int >
void
runSpmvBenchmarksForMatrix( BenchmarkType& benchmark,
                            // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
                            TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >& csrHostMatrix,
                            const String& inputFileName,
                            const Config::ParameterContainer& parameters,
                            bool verboseMR,
                            bool transposed )
{
#ifdef __CUDACC__
   using CSRCudaMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Cuda, int >;
   using CusparseMatrix = TNL::CusparseCSR< Real >;
   using SlicedEllCudaMatrix =
      TNL::Matrices::SparseMatrix< Real, TNL::Devices::Cuda, int, Matrices::GeneralMatrix, SlicedEllpackSegments >;
   using CusparseSlicedEllMatrix = TNL::CusparseSlicedEll< Real >;
#endif
#ifdef __HIP__
   using CSRHipMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Hip, int >;
   using HipsparseMatrix = TNL::HipsparseCSR< Real >;
#endif

   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   ////
   // Set-up benchmark datasize
   //
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
   benchmark.setMetadataColumns( {
      { "matrix name", inputFileName },
      { "transposed", transposed ? "true" : "false" },
      { "precision", getType< Real >() },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "nonzeros", convertToString( nonzeros ) },
      { "nonzeros per row std_dev", convertToString( std_dev ) },
      { "nonzeros per row percentile 25", convertToString( percentile_25 ) },
      { "nonzeros per row percentile 50", convertToString( percentile_50 ) },
      { "nonzeros per row percentile 75", convertToString( percentile_75 ) }
      // NOTE: 'nonzeros per row average' can be easily calculated with Pandas based on the other metadata
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

   // no benchmark, just initialize hostOutVector
   resetHostVectors();
   spmvCSRHost();

#ifdef HAVE_PETSC
   Mat petscMatrix;
   Containers::Vector< PetscInt, Devices::Host, PetscInt > petscRowPointers( csrHostMatrix.getRowPointers() );
   Containers::Vector< PetscInt, Devices::Host, PetscInt > petscColumns( csrHostMatrix.getColumnIndexes() );
   Containers::Vector< PetscScalar, Devices::Host, PetscInt > petscValues( csrHostMatrix.getValues() );
   MatCreateSeqAIJWithArrays( PETSC_COMM_WORLD,  //PETSC_COMM_SELF,
                              csrHostMatrix.getRows(),
                              csrHostMatrix.getColumns(),
                              petscRowPointers.getData(),
                              petscColumns.getData(),
                              petscValues.getData(),
                              &petscMatrix );
   Vec inVector, outVector;
   VecCreateSeq( PETSC_COMM_WORLD, csrHostMatrix.getColumns(), &inVector );
   VecCreateSeq( PETSC_COMM_WORLD, csrHostMatrix.getRows(), &outVector );

   auto resetPetscVectors = [ & ]()
   {
      VecSet( inVector, 1.0 );
      VecSet( outVector, 0.0 );
   };

   auto petscSpmvCSRHost = [ & ]()
   {
      MatMult( petscMatrix, inVector, outVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > petscBenchmarkResults( hostOutVector, outVector );
   benchmark.setMetadataElement( { "format", "Petsc" } );
   benchmark.time< Devices::Host >( resetPetscVectors, "CPU", petscSpmvCSRHost, petscBenchmarkResults );
#endif

#if defined( HAVE_HYPRE ) && ! defined( HYPRE_USING_CUDA )
   // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
   if constexpr( std::is_same< HYPRE_Real, Real >::value && std::is_same< HYPRE_Int, int >::value ) {
      TNL::Hypre hypre;
      using HypreCSR = TNL::Matrices::HypreCSRMatrix;
      HypreCSR hypreCSRMatrix( csrHostMatrix.getRows(),
                               csrHostMatrix.getColumns(),
                               csrHostMatrix.getValues().getView(),
                               csrHostMatrix.getColumnIndexes().getView(),
                               csrHostMatrix.getSegments().getOffsets().getView() );
      auto hostInVectorView = hostInVector.getView();
      auto hostOutVectorView = hostOutVector.getView();

      auto spmvHypreCSRHost = [ & ]()
      {
         hypreCSRMatrix.vectorProduct( hostInVectorView, hostOutVectorView );
      };

      SpmvBenchmarkResult< Real, Devices::Host, int > hypreBenchmarkResults( hostOutVector, hostOutVector );
      const int maxThreadsCount =
         max( 1, Devices::Host::getMaxThreadsCount() );  // TODO: This si workaround for getMaxThreadsCoutn returning 0 if
                                                         // OpenMP is disabled
      int threads = 1;
      while( true ) {
         benchmark.setMetadataElement( { "format", "Hypre" } );
         auto launch_config = convertToString( threads ) + " threads";
         if( threads == 1 )
            launch_config = "1 thread";
         benchmark.setMetadataElement( { "launch cfg.", launch_config.getString() } );
         Devices::Host::setMaxThreadsCount( threads );
         benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHypreCSRHost, hypreBenchmarkResults );
         if( threads == maxThreadsCount )
            break;
         threads = min( 2 * threads, maxThreadsCount );
      }
   }
   else {
      std::cerr << "Current Real or Index type does not agree with HYPRE_Real or HYPRE_Index." << std::endl;
   }
#endif

#ifdef HAVE_GINKGO
   // Create a Ginkgo Csr view
   auto gko_host_exec = gko::OmpExecutor::create();
   auto gko_host_A = gko::share( TNL::Matrices::getGinkgoMatrixCsrView( gko_host_exec, csrHostMatrix ) );

   // Wrap the vectors
   auto gko_host_b = Containers::GinkgoVector< Real, Devices::Host >::create( gko_host_exec, hostOutVector.getView() );
   auto gko_host_x = Containers::GinkgoVector< Real, Devices::Host >::create( gko_host_exec, hostInVector.getView() );

   auto spmvGinkgoCSRHost = [ & ]()
   {
      gko_host_A->apply( gko_host_b.get(), gko_host_x.get() );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > ginkgoHostBenchmarkResults( hostOutVector, hostOutVector );
   const int maxThreadsCount =
      max( 1, Devices::Host::getMaxThreadsCount() );  // TODO: This si workaround for getMaxThreadsCoutn returning 0 if OpenMP
                                                      // is disabled
   int threads = 1;
   while( true ) {
      benchmark.setMetadataElement( { "format", "Ginkgo" } );
      auto launch_config = convertToString( threads ) + " threads";
      if( threads == 1 )
         launch_config = "1 thread";
      benchmark.setMetadataElement( { "launch cfg.", launch_config.getString() } );
      Devices::Host::setMaxThreadsCount( threads );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvGinkgoCSRHost, ginkgoHostBenchmarkResults );
      if( threads == maxThreadsCount )
         break;
      threads = min( 2 * threads, maxThreadsCount );
   }
#endif

#ifdef __CUDACC__
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;
   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
   cusparseHandle_t cusparseHandle;
   cusparseCreate( &cusparseHandle );
   int cusparseVersion = 0;
   cusparseGetVersion( cusparseHandle, &cusparseVersion );
   benchmark.setMetadataElement( { "cusparse version", convertToString( cusparseVersion ) } );

   CSRCudaMatrix csrCudaMatrix;
   csrCudaMatrix = csrHostMatrix;

   CudaVector cudaInVector( csrCudaMatrix.getColumns() ), cudaOutVector( csrCudaMatrix.getRows() );

   auto resetCudaVectors = [ & ]()
   {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( hostOutVector, cudaOutVector );

   // Compare the algorithm cuSPARSE picks automatically (Default) against explicitly
   // requesting the CSR-specific algorithms, to check whether Default is actually optimal.
   struct CusparseAlgVariant
   {
      const char* name;
      cusparseSpMVAlg_t alg;
   };
   const CusparseAlgVariant cusparseAlgorithms[] = {
      { "Default", CUSPARSE_SPMV_ALG_DEFAULT },
      { "CSR ALG1", CUSPARSE_SPMV_CSR_ALG1 },
      { "CSR ALG2", CUSPARSE_SPMV_CSR_ALG2 },
   };
   for( const auto& variant : cusparseAlgorithms ) {
      CusparseMatrix cusparseMatrix;
      cusparseMatrix.init( csrCudaMatrix, cudaInVector, cudaOutVector, &cusparseHandle, variant.alg );

      auto spmvCusparse = [ & ]()
      {
         cusparseMatrix.vectorProduct( cudaInVector, cudaOutVector );
      };

      benchmark.setMetadataElement( { "format", "cusparse" } );
      benchmark.setMetadataElement( { "launch cfg.", variant.name } );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "CUDA", spmvCusparse, cudaBenchmarkResults );
   }

   ////
   // Perform benchmark on CUDA device with cuSPARSE's Sliced ELLPACK format
   //
   SlicedEllCudaMatrix slicedEllCudaMatrix;
   slicedEllCudaMatrix = csrHostMatrix;

   const CusparseAlgVariant cusparseSlicedEllAlgorithms[] = {
      { "Default", CUSPARSE_SPMV_ALG_DEFAULT },
      { "SELL ALG1", CUSPARSE_SPMV_SELL_ALG1 },
   };
   for( const auto& variant : cusparseSlicedEllAlgorithms ) {
      CusparseSlicedEllMatrix cusparseSlicedEllMatrix;
      cusparseSlicedEllMatrix.init( slicedEllCudaMatrix, cudaInVector, cudaOutVector, &cusparseHandle, variant.alg );

      auto spmvCusparseSlicedEll = [ & ]()
      {
         cusparseSlicedEllMatrix.vectorProduct( cudaInVector, cudaOutVector );
      };

      benchmark.setMetadataElement( { "format", "cusparse SlicedEll" } );
      benchmark.setMetadataElement( { "launch cfg.", variant.name } );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "CUDA", spmvCusparseSlicedEll, cudaBenchmarkResults );
   }

   #if defined( HAVE_HYPRE ) && defined( HYPRE_USING_CUDA )
   // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
   if constexpr( std::is_same< HYPRE_Real, Real >::value && std::is_same< HYPRE_Int, int >::value ) {
      TNL::Hypre hypre;
      using HypreCSR = TNL::Matrices::HypreCSRMatrix;
      HypreCSR hypreCSRMatrix( csrCudaMatrix.getRows(),
                               csrCudaMatrix.getColumns(),
                               csrCudaMatrix.getValues().getView(),
                               csrCudaMatrix.getColumnIndexes().getView(),
                               csrCudaMatrix.getSegments().getOffsets().getView() );
      auto cudaInVectorView = cudaInVector.getView();
      auto cudaOutVectorView = cudaOutVector.getView();

      auto spmvHypreCSRCuda = [ & ]()
      {
         hypreCSRMatrix.vectorProduct( cudaInVectorView, cudaOutVectorView );
      };

      SpmvBenchmarkResult< Real, Devices::Cuda, int > hypreCudaBenchmarkResults( hostOutVector, cudaOutVector );
      benchmark.setMetadataElement( { "format", "Hypre" } );
      benchmark.setMetadataElement( { "launch cfg.", "Default" } );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "CUDA", spmvHypreCSRCuda, hypreCudaBenchmarkResults );
   }
   else {
      std::cerr << "Current Real or Index type does not agree with HYPRE_Real or HYPRE_Index." << std::endl;
   }
   #endif

   #ifdef HAVE_GINKGO
   // Create a Ginkgo Csr view
   auto gko_cuda_exec = gko::CudaExecutor::create( 0, gko_host_exec );
   auto gko_cuda_A = gko::share( TNL::Matrices::getGinkgoMatrixCsrView( gko_cuda_exec, csrCudaMatrix ) );

   // Wrap the vectors
   auto gko_cuda_b = Containers::GinkgoVector< Real, Devices::Cuda >::create( gko_cuda_exec, cudaOutVector.getView() );
   auto gko_cuda_x = Containers::GinkgoVector< Real, Devices::Cuda >::create( gko_cuda_exec, cudaInVector.getView() );

   auto spmvGinkgoCSRCuda = [ & ]()
   {
      gko_cuda_A->apply( gko_cuda_b.get(), gko_cuda_x.get() );
   };

   SpmvBenchmarkResult< Real, Devices::Cuda, int > ginkgoCudaBenchmarkResults( hostOutVector, cudaOutVector );
   benchmark.setMetadataElement( { "format", "Ginkgo" } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "CUDA", spmvGinkgoCSRCuda, ginkgoCudaBenchmarkResults );
   #endif

   #ifdef HAVE_CSR5
   ////
   // Perform benchmark on CUDA device with CSR5 as a reference GPU format
   //
   CudaVector cudaOutVector2( cudaOutVector );
   CSR5Benchmark::CSR5Benchmark< CSRCudaMatrix > csr5Benchmark( csrCudaMatrix, cudaInVector, cudaOutVector );

   auto csr5SpMV = [ & ]()
   {
      csr5Benchmark.vectorProduct();
   };

   benchmark.setMetadataElement( { "format", "CSR5" } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );
   benchmark.time< Devices::Cuda >( resetCusparseVectors, "CUDA", csr5SpMV, cudaBenchmarkResults );
   std::cerr << "CSR5 error = " << max( abs( cudaOutVector - cudaOutVector2 ) ) << std::endl;
   csrCudaMatrix.reset();
   #endif

   // LightSpMV fails with CUDA 12
   #if __CUDACC_VER_MAJOR__ < 11 || ( __CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 2 )
   ////
   // Perform benchmark on CUDA device with LightSpMV as a reference GPU format
   //
   using LightSpMVCSRHostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, uint32_t >;
   LightSpMVCSRHostMatrix lightSpMVCSRHostMatrix;
   lightSpMVCSRHostMatrix = csrHostMatrix;
   LightSpMVBenchmark< Real > lightSpMVBenchmark( lightSpMVCSRHostMatrix, LightSpMVBenchmarkKernelVector );
   auto resetLightSpMVVectors = [ & ]()
   {
      lightSpMVBenchmark.resetVectors();
   };

   auto spmvLightSpMV = [ & ]()
   {
      lightSpMVBenchmark.vectorProduct();
   };
   benchmark.setMetadataElement( { "format", "LightSpMV Vector" } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "CUDA", spmvLightSpMV, cudaBenchmarkResults );

   lightSpMVBenchmark.setKernelType( LightSpMVBenchmarkKernelWarp );
   benchmark.setMetadataElement( { "format", "LightSpMV Warp" } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "CUDA", spmvLightSpMV, cudaBenchmarkResults );
   #endif

   cusparseDestroy( cusparseHandle );
#endif

#ifdef __HIP__
   using HipVector = Containers::Vector< Real, Devices::Hip, int >;
   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
   hipsparseHandle_t hipsparseHandle;
   hipsparseCreate( &hipsparseHandle );

   CSRHipMatrix csrHipMatrix;
   csrHipMatrix = csrHostMatrix;

   HipVector hipInVector( csrHipMatrix.getColumns() ), hipOutVector( csrHipMatrix.getRows() );

   HipsparseMatrix hipsparseMatrix;
   hipsparseMatrix.init( csrHipMatrix, hipInVector, hipOutVector, &hipsparseHandle );

   auto resetHipVectors = [ & ]()
   {
      hipInVector = 1.0;
      hipOutVector = 0.0;
   };

   auto spmvHipsparse = [ & ]()
   {
      hipsparseMatrix.vectorProduct( hipInVector, hipOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Hip, int > hipBenchmarkResults( hostOutVector, hipOutVector );
   benchmark.setMetadataElement( { "format", "hipsparse" } );
   benchmark.time< Devices::Hip >( resetHipVectors, "HIP", spmvHipsparse, hipBenchmarkResults );

   #ifdef HAVE_GINKGO
   // Create a Ginkgo Csr view
   auto gko_hip_exec = gko::CudaExecutor::create( 0, gko_host_exec );
   auto gko_hip_A = gko::share( TNL::Matrices::getGinkgoMatrixCsrView( gko_hip_exec, csrHipMatrix ) );

   // Wrap the vectors
   auto gko_hip_b = Containers::GinkgoVector< Real, Devices::Hip >::create( gko_hip_exec, hipOutVector.getView() );
   auto gko_hip_x = Containers::GinkgoVector< Real, Devices::Hip >::create( gko_hip_exec, hipInVector.getView() );

   auto spmvGinkgoCSRHip = [ & ]()
   {
      gko_hip_A->apply( gko_hip_b.get(), gko_hip_x.get() );
   };

   SpmvBenchmarkResult< Real, Devices::Hip, int > ginkgoHipBenchmarkResults( hostOutVector, hipOutVector );
   benchmark.setMetadataElement( { "format", "Ginkgo CSR" } );
   benchmark.time< Devices::Hip >( resetHipVectors, "HIP", spmvGinkgoCSRHip, ginkgoHipBenchmarkResults );
   #endif

#endif
   csrHostMatrix.reset();
}

template< typename Real = double, typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;

   CSRHostMatrix csrHostMatrix;
   TNL::Matrices::MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
   const Index uncompressedSize = csrHostMatrix.getValues().getSize();
   TNL::Matrices::compressSparseMatrix( csrHostMatrix );
   const Index compressedSize = csrHostMatrix.getValues().getSize();
   std::cout << "Compression ratio: " << (double) uncompressedSize / compressedSize << std::endl;

   const String& withTransposedMatrix = parameters.getParameter< String >( "with-transposed-matrix" );

   // Compute the transpose (if needed) before csrHostMatrix is possibly
   // reset by runSpmvBenchmarksForMatrix() below.
   CSRHostMatrix transposedHostMatrix;
   if( withTransposedMatrix != "false" )
      transposedHostMatrix.getTransposition( csrHostMatrix );

   if( withTransposedMatrix != "only" )
      runSpmvBenchmarksForMatrix< Real, Index >( benchmark, csrHostMatrix, inputFileName, parameters, verboseMR, false );
   else
      csrHostMatrix.reset();

   if( withTransposedMatrix != "false" )
      runSpmvBenchmarksForMatrix< Real, Index >( benchmark, transposedHostMatrix, inputFileName, parameters, verboseMR, true );
}

}  // namespace TNL::Benchmarks::SpMV
