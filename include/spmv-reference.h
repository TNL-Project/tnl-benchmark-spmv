#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <TNL/Benchmarks/Benchmark.h>
#include "SpmvBenchmarkResult.h"
#include "CpuBenchmarking.h"
#include "MatrixTransposition.h"

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

#ifdef HAVE_GINKGO
// Creates a Ginkgo Csr matrix view of a TNL CSR matrix. This is the same as
// TNL::Matrices::getGinkgoMatrixCsrView, but TNL/Matrices/GinkgoOperator.h
// cannot be included with Ginkgo >= 2.0 (gko::EnableLinOp was removed).
// The SpMV strategy of the Csr matrix is the default one of Ginkgo (automatic)
// unless another one is given.
template< typename Matrix >
auto
getGinkgoMatrixCsrView( std::shared_ptr< const gko::Executor > exec,
                        Matrix& matrix,
                        gko::matrix::csr::spmv_strategy strategy = gko::matrix::csr::spmv_strategy::automatic )
   -> std::unique_ptr< gko::matrix::Csr< typename Matrix::RealType, typename Matrix::IndexType > >
{
   return gko::matrix::Csr< typename Matrix::RealType, typename Matrix::IndexType >::create(
      exec,
      gko::dim< 2 >{ static_cast< std::size_t >( matrix.getRows() ), static_cast< std::size_t >( matrix.getColumns() ) },
      gko::make_array_view( exec, matrix.getNonzeroElementsCount(), matrix.getValues().getData() ),
      gko::make_array_view( exec, matrix.getNonzeroElementsCount(), matrix.getColumnIndexes().getData() ),
      gko::make_array_view( exec, matrix.getRows() + 1, matrix.getSegments().getOffsets().getData() ),
      strategy );
}

// A SpMV strategy of the Ginkgo Csr matrix to benchmark, with the names for
// the "format" and "launch cfg." columns.
struct GinkgoCsrStrategy
{
   std::string format;
   std::string launchConfig;
   gko::matrix::csr::spmv_strategy strategy;
};

// Benchmarks all SpMV kernels of Ginkgo on a GPU, similarly to the algorithms
// of cuSPARSE:
//
//  - the Csr matrix with its SpMV strategies. "Default" is the automatic
//    strategy, which chooses between load_balance and classical, the same
//    configuration as benchmarked before the other ones were added. The
//    sparselib strategy calls cuSPARSE or hipSPARSE, so it is logged under
//    the format "Ginkgo Sparselib" to keep it apart from the kernels of
//    Ginkgo itself.
//  - the Ell, Sellp (with the default slice size 64 and with 32), Hybrid and
//    Coo matrices converted from the Csr matrix.
//
// The matrix is given as a TNL CSR matrix on the device of the executor. A
// failure of a conversion (e.g. not enough memory for Ell) is logged as an
// error and the other kernels are benchmarked anyway.
template< typename Real,
          typename Device,
          typename CsrMatrix,
          typename InVector,
          typename OutVector,
          typename ResultVector,
          typename ResetVectors >
void
benchmarkGinkgoOnGpu( BenchmarkType& benchmark,
                      std::shared_ptr< const gko::Executor > exec,
                      CsrMatrix& csrMatrix,
                      InVector& inVector,
                      OutVector& outVector,
                      const ResultVector& csrResultVector,
                      ResetVectors& resetVectors,
                      const std::string& performer )
{
   using Index = typename CsrMatrix::IndexType;
   namespace csr = gko::matrix::csr;

   // Wrap the vectors
   // apply( b, x ) computes x = A * b
   auto gko_b = Containers::GinkgoVector< Real, Device >::create( exec, inVector.getView() );
   auto gko_x = Containers::GinkgoVector< Real, Device >::create( exec, outVector.getView() );
   SpmvBenchmarkResult< Real, Device, int > benchmarkResults( csrResultVector, outVector );

   // `create` returns the matrix to benchmark, it throws if it cannot be created
   auto benchmarkMatrix = [ & ]( const std::string& format, const std::string& launchConfig, auto&& create )
   {
      benchmark.setMetadataElement( { "format", format } );
      benchmark.setMetadataElement( { "launch cfg.", launchConfig } );
      std::shared_ptr< const gko::LinOp > gko_A;
      try {
         gko_A = create();
      }
      catch( const std::exception& e ) {
         benchmark.addErrorMessage( "Unable to create the Ginkgo matrix: " + String( e.what() ) );
         return;
      }
      auto spmvGinkgo = [ & ]()
      {
         gko_A->apply( gko_b.get(), gko_x.get() );
      };
      benchmark.time< Device >( resetVectors, performer.c_str(), spmvGinkgo, benchmarkResults );
   };

   const GinkgoCsrStrategy strategies[] = {
      { "Ginkgo", "Default", csr::spmv_strategy::automatic },
      { "Ginkgo", "Classical", csr::spmv_strategy::classical },
      { "Ginkgo", "Load balance", csr::spmv_strategy::load_balance },
      { "Ginkgo", "Merge path", csr::spmv_strategy::merge_path },
      { "Ginkgo Sparselib", "Default", csr::spmv_strategy::sparselib },
   };
   for( const auto& variant : strategies )
      benchmarkMatrix( variant.format,
                       variant.launchConfig,
                       [ & ]()
                       {
                          return gko::share( getGinkgoMatrixCsrView( exec, csrMatrix, variant.strategy ) );
                       } );

   // The other formats are converted from the Csr matrix.
   auto gko_csr = gko::share( getGinkgoMatrixCsrView( exec, csrMatrix ) );
   benchmarkMatrix( "Ginkgo Ell",
                    "Default",
                    [ & ]()
                    {
                       auto matrix = gko::share( gko::matrix::Ell< Real, Index >::create( exec ) );
                       gko_csr->convert_to( matrix );
                       return matrix;
                    } );
   for( const gko::size_type sliceSize : { gko::matrix::default_slice_size, 32 } )
      benchmarkMatrix( "Ginkgo Sellp",
                       "Slice " + std::to_string( sliceSize ),
                       [ & ]()
                       {
                          auto matrix = gko::share( gko::matrix::Sellp< Real, Index >::create(
                             exec, gko::dim< 2 >{}, sliceSize, gko::matrix::default_stride_factor, 0 ) );
                          gko_csr->convert_to( matrix );
                          return matrix;
                       } );
   benchmarkMatrix( "Ginkgo Hybrid",
                    "Default",
                    [ & ]()
                    {
                       auto matrix = gko::share( gko::matrix::Hybrid< Real, Index >::create( exec ) );
                       gko_csr->convert_to( matrix );
                       return matrix;
                    } );
   benchmarkMatrix( "Ginkgo Coo",
                    "Default",
                    [ & ]()
                    {
                       auto matrix = gko::share( gko::matrix::Coo< Real, Index >::create( exec ) );
                       gko_csr->convert_to( matrix );
                       return matrix;
                    } );
}
#endif

#if defined( HAVE_HYPRE ) && defined( HYPRE_USING_GPU )
// Benchmarks the SpMV kernels of Hypre on a GPU. Hypre has only the CSR
// format, but two implementations of the multiplication: the one of the
// vendor library (cuSPARSE, rocSPARSE), which is the default one if Hypre is
// built with the vendor library, and its own kernel. They are chosen by
// HYPRE_SetSpMVUseVendor and logged with the launch configurations "Default"
// and "Internal". Without the vendor library, Hypre has only its own kernel,
// logged as "Default".
//
// The matrix is given as a TNL CSR matrix on the GPU.
template< typename Real,
          typename Device,
          typename CsrMatrix,
          typename InVector,
          typename OutVector,
          typename ResultVector,
          typename ResetVectors >
void
benchmarkHypreOnGpu( BenchmarkType& benchmark,
                     CsrMatrix& csrMatrix,
                     InVector& inVector,
                     OutVector& outVector,
                     const ResultVector& csrResultVector,
                     ResetVectors& resetVectors,
                     const std::string& performer )
{
   if constexpr( std::is_same< HYPRE_Real, Real >::value && std::is_same< HYPRE_Int, int >::value ) {
      // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
      TNL::Hypre hypre;
      using HypreCSR = TNL::Matrices::HypreCSRMatrix;
      HypreCSR hypreCSRMatrix( csrMatrix.getRows(),
                               csrMatrix.getColumns(),
                               csrMatrix.getValues().getView(),
                               csrMatrix.getColumnIndexes().getView(),
                               csrMatrix.getSegments().getOffsets().getView() );
      auto inVectorView = inVector.getView();
      auto outVectorView = outVector.getView();

      auto spmvHypreCSR = [ & ]()
      {
         hypreCSRMatrix.vectorProduct( inVectorView, outVectorView );
      };

      SpmvBenchmarkResult< Real, Device, int > hypreBenchmarkResults( csrResultVector, outVector );
      benchmark.setMetadataElement( { "format", "Hypre" } );
   #if defined( HYPRE_USING_CUSPARSE ) || defined( HYPRE_USING_ROCSPARSE )
      // The setting is global and kept for the next matrix, so the default
      // one must be set explicitly too.
      const std::pair< std::string, HYPRE_Int > implementations[] = { { "Default", 1 }, { "Internal", 0 } };
      for( const auto& [ launchConfig, useVendor ] : implementations ) {
         HYPRE_SetSpMVUseVendor( useVendor );
         benchmark.setMetadataElement( { "launch cfg.", launchConfig } );
         benchmark.time< Device >( resetVectors, performer.c_str(), spmvHypreCSR, hypreBenchmarkResults );
      }
      HYPRE_SetSpMVUseVendor( 1 );
   #else
      benchmark.setMetadataElement( { "launch cfg.", "Default" } );
      benchmark.time< Device >( resetVectors, performer.c_str(), spmvHypreCSR, hypreBenchmarkResults );
   #endif
   }
   else {
      std::cerr << "Current Real or Index type does not agree with HYPRE_Real or HYPRE_Index." << std::endl;
   }
}
#endif

#ifdef __CUDACC__
// cuSPARSE's Sliced ELLPACK matches TNL's column-major SlicedEllpack segments,
// see the comment in cusparseSlicedEllMatrix.h. Alias templates must live at
// namespace scope, so this cannot be declared inside benchmarkSpmv().
template< typename Device_, typename Index_, typename IndexAllocator_ >
using SlicedEllpackSegments = TNL::Algorithms::Segments::ColumnMajorSlicedEllpack< Device_, Index_, IndexAllocator_ >;

// A cuSPARSE SpMV algorithm to benchmark and its name for the "launch cfg." column.
// It is declared at namespace scope, because some nvcc versions fail to compile
// a local struct used in a function template.
struct CusparseAlgVariant
{
   std::string name;
   cusparseSpMVAlg_t alg;
};
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
                            bool transposed,
                            const MatrixSymmetry& symmetry )
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

   String matrixName = parameters.getParameter< String >( "matrix-name" );
   if( matrixName.empty() )
      matrixName = inputFileName;

   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   benchmark.setMetadataColumns( {
      { "matrix name", matrixName },
      { "transposed", transposed ? "true" : "false" },
      { "symmetric", symmetry.symmetric ? "true" : "false" },
      { "structurally symmetric", symmetry.structurallySymmetric ? "true" : "false" },
      { "precision", getType< Real >() },
      { "build", getBuildName() },
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

   // no benchmark, just compute the reference result the other formats are compared with - it must be a separate
   // vector, because hostOutVector is overwritten by the benchmarks on the host
   resetHostVectors();
   spmvCSRHost();
   const HostVector csrResultVector( hostOutVector );

   const bool cpuTests = parameters.getParameter< bool >( "with-cpu-tests" );

#ifdef HAVE_PETSC
   if( cpuTests ) {
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

      SpmvBenchmarkResult< Real, Devices::Host, int > petscBenchmarkResults( csrResultVector, outVector );
      benchmark.setMetadataElement( { "format", "Petsc" } );
      // MatCreateSeqAIJWithArrays creates a sequential matrix
      benchmark.setMetadataElement( { "launch cfg.", getCpuLaunchConfig( 1 ) } );
      benchmark.time< Devices::Host >( resetPetscVectors, "CPU", petscSpmvCSRHost, petscBenchmarkResults );
   }
#endif

#if defined( HAVE_HYPRE ) && ! defined( HYPRE_USING_GPU )
   // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
   if constexpr( std::is_same< HYPRE_Real, Real >::value && std::is_same< HYPRE_Int, int >::value ) {
      if( cpuTests ) {
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

         SpmvBenchmarkResult< Real, Devices::Host, int > hypreBenchmarkResults( csrResultVector, hostOutVector );
         benchmark.setMetadataElement( { "format", "Hypre" } );
         forEachCpuThreadsCount( true,
                                 [ & ]( const std::string& launchConfig )
                                 {
                                    benchmark.setMetadataElement( { "launch cfg.", launchConfig } );
                                    benchmark.time< Devices::Host >(
                                       resetHostVectors, "CPU", spmvHypreCSRHost, hypreBenchmarkResults );
                                 } );
      }
   }
   else {
      std::cerr << "Current Real or Index type does not agree with HYPRE_Real or HYPRE_Index." << std::endl;
   }
#endif

#ifdef HAVE_GINKGO
   // The host executor is also the master executor of the GPU executors below.
   auto gko_host_exec = gko::OmpExecutor::create();
#endif

#if defined( HAVE_GINKGO ) && ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   // Ginkgo on the CPU is benchmarked only in the host binary, the GPU binaries
   // would just repeat the same measurements.
   if( cpuTests ) {
      // Create a Ginkgo Csr view
      auto gko_host_A = gko::share( getGinkgoMatrixCsrView( gko_host_exec, csrHostMatrix ) );

      // Wrap the vectors
      // apply( b, x ) computes x = A * b
      auto gko_host_b = Containers::GinkgoVector< Real, Devices::Host >::create( gko_host_exec, hostInVector.getView() );
      auto gko_host_x = Containers::GinkgoVector< Real, Devices::Host >::create( gko_host_exec, hostOutVector.getView() );

      auto spmvGinkgoCSRHost = [ & ]()
      {
         gko_host_A->apply( gko_host_b.get(), gko_host_x.get() );
      };

      SpmvBenchmarkResult< Real, Devices::Host, int > ginkgoHostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.setMetadataElement( { "format", "Ginkgo" } );
      forEachCpuThreadsCount( true,
                              [ & ]( const std::string& launchConfig )
                              {
                                 benchmark.setMetadataElement( { "launch cfg.", launchConfig } );
                                 benchmark.time< Devices::Host >(
                                    resetHostVectors, "CPU", spmvGinkgoCSRHost, ginkgoHostBenchmarkResults );
                              } );
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

   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );

   // Compare the algorithm cuSPARSE picks automatically (Default) against explicitly
   // requesting the CSR-specific algorithms, to check whether Default is actually optimal.
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
      benchmark.setMetadataElement( std::make_pair( std::string( "launch cfg." ), variant.name ) );
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
      benchmark.setMetadataElement( std::make_pair( std::string( "launch cfg." ), variant.name ) );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "CUDA", spmvCusparseSlicedEll, cudaBenchmarkResults );
   }

   #if defined( HAVE_HYPRE ) && defined( HYPRE_USING_CUDA )
   benchmarkHypreOnGpu< Real, Devices::Cuda >(
      benchmark, csrCudaMatrix, cudaInVector, cudaOutVector, csrResultVector, resetCudaVectors, "CUDA" );
   #endif

   #ifdef HAVE_GINKGO
   benchmarkGinkgoOnGpu< Real, Devices::Cuda >( benchmark,
                                                gko::CudaExecutor::create( 0, gko_host_exec ),
                                                csrCudaMatrix,
                                                cudaInVector,
                                                cudaOutVector,
                                                csrResultVector,
                                                resetCudaVectors,
                                                "CUDA" );
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
   // Perform benchmark on HIP device with hipSPARSE as a reference GPU format
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

   SpmvBenchmarkResult< Real, Devices::Hip, int > hipBenchmarkResults( csrResultVector, hipOutVector );
   benchmark.setMetadataElement( { "format", "hipsparse" } );
   benchmark.setMetadataElement( { "launch cfg.", "Default" } );
   benchmark.time< Devices::Hip >( resetHipVectors, "HIP", spmvHipsparse, hipBenchmarkResults );

   #if defined( HAVE_HYPRE ) && defined( HYPRE_USING_HIP )
   benchmarkHypreOnGpu< Real, Devices::Hip >(
      benchmark, csrHipMatrix, hipInVector, hipOutVector, csrResultVector, resetHipVectors, "HIP" );
   #endif

   #ifdef HAVE_GINKGO
   benchmarkGinkgoOnGpu< Real, Devices::Hip >( benchmark,
                                               gko::HipExecutor::create( 0, gko_host_exec ),
                                               csrHipMatrix,
                                               hipInVector,
                                               hipOutVector,
                                               csrResultVector,
                                               resetHipVectors,
                                               "HIP" );
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
   const MatrixSymmetry symmetry = getMatrixSymmetry( inputFileName, csrHostMatrix );
   const bool withTransposition = withTransposedMatrix != "false" && ! symmetry.structurallySymmetric;
   if( withTransposedMatrix != "false" && symmetry.structurallySymmetric )
      std::cout << "The matrix is structurally symmetric, its transposition is not benchmarked." << std::endl;

   // Compute the transpose (if needed) before csrHostMatrix is possibly
   // reset by runSpmvBenchmarksForMatrix() below.
   CSRHostMatrix transposedHostMatrix;
   if( withTransposition )
      getSortedTransposition( transposedHostMatrix, csrHostMatrix );

   if( withTransposedMatrix != "only" )
      runSpmvBenchmarksForMatrix< Real, Index >(
         benchmark, csrHostMatrix, inputFileName, parameters, verboseMR, false, symmetry );
   else
      csrHostMatrix.reset();

   if( withTransposition )
      runSpmvBenchmarksForMatrix< Real, Index >(
         benchmark, transposedHostMatrix, inputFileName, parameters, verboseMR, true, symmetry );
}

}  // namespace TNL::Benchmarks::SpMV
