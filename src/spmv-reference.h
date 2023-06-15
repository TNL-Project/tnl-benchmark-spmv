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
#include <TNL/Algorithms/SegmentsReductionKernels/CSRScalarKernel.h>
#include <TNL/Algorithms/sort.h>

#ifdef HAVE_PETSC
#include <petscmat.h>
#endif

#ifdef HAVE_HYPRE
#include <TNL/Hypre.h>
#include <TNL/Matrices/HypreCSRMatrix.h>
#endif

#include "ReferenceFormats/cusparseCSRMatrix.h"
#include "ReferenceFormats/LightSpMVBenchmark.h"
#include "ReferenceFormats/CSR5Benchmark.h"

namespace TNL::Benchmarks::SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark< JsonLogging >;

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;
   #ifdef __CUDACC__
   using CSRCudaMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Cuda, int >;
   using CusparseMatrix = TNL::CusparseCSR< Real >;
   #endif

   using HostVector = Containers::Vector< Real, Devices::Host, int >;

   CSRHostMatrix csrHostMatrix;

   ////
   // Set-up benchmark datasize
   //
   TNL::Matrices::MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
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

   // no benchmark, just initialize hostOutVector
   resetHostVectors();
   spmvCSRHost();

#ifdef HAVE_PETSC
   Mat petscMatrix;
   Containers::Vector< PetscInt, Devices::Host, PetscInt > petscRowPointers( csrHostMatrix.getRowPointers() );
   Containers::Vector< PetscInt, Devices::Host, PetscInt > petscColumns( csrHostMatrix.getColumnIndexes() );
   Containers::Vector< PetscScalar, Devices::Host, PetscInt > petscValues( csrHostMatrix.getValues() );
   MatCreateSeqAIJWithArrays( PETSC_COMM_WORLD, //PETSC_COMM_SELF,
                              csrHostMatrix.getRows(),
                              csrHostMatrix.getColumns(),
                              petscRowPointers.getData(),
                              petscColumns.getData(),
                              petscValues.getData(),
                              &petscMatrix );
   Vec inVector, outVector;
   VecCreateSeq( PETSC_COMM_WORLD, csrHostMatrix.getColumns(), &inVector );
   VecCreateSeq( PETSC_COMM_WORLD, csrHostMatrix.getRows(), &outVector );

   auto resetPetscVectors = [&]() {
      VecSet( inVector, 1.0 );
      VecSet( outVector, 0.0 );
   };

   auto petscSpmvCSRHost = [&]() {
      MatMult( petscMatrix, inVector, outVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > petscBenchmarkResults( hostOutVector, hostOutVector );
   benchmark.setMetadataElement({ "format", "Petsc" });
   benchmark.time< Devices::Host >( resetPetscVectors, "CPU", petscSpmvCSRHost, petscBenchmarkResults );
#endif

#if defined( HAVE_HYPRE ) && ! defined( HYPRE_USING_CUDA )
   // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
   if constexpr( std::is_same< HYPRE_Real, Real >::value &&
                 std::is_same< HYPRE_Int, int >::value ) {
      TNL::Hypre hypre;
      using HypreCSR = TNL::Matrices::HypreCSRMatrix;
      HypreCSR hypreCSRMatrix( csrHostMatrix.getRows(),
                               csrHostMatrix.getColumns(),
                               csrHostMatrix.getValues().getView(),
                               csrHostMatrix.getColumnIndexes().getView(),
                               csrHostMatrix.getSegments().getOffsets().getView());
      auto hostInVectorView = hostInVector.getView();
      auto hostOutVectorView = hostOutVector.getView();

      auto spmvHypreCSRHost = [&]() {
         hypreCSRMatrix.vectorProduct( hostInVectorView, hostOutVectorView );
      };

      SpmvBenchmarkResult< Real, Devices::Host, int > hypreBenchmarkResults( hostOutVector, hostOutVector );
      const int maxThreadsCount = Devices::Host::getMaxThreadsCount();
      int threads = 1;
      while( true ) {
         benchmark.setMetadataElement({ "format", "Hypre" });
         benchmark.setMetadataElement({ "threads", convertToString( threads ).getString() });
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


#ifdef __CUDACC__
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;
   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
   cusparseHandle_t cusparseHandle;
   cusparseCreate( &cusparseHandle );

   CSRCudaMatrix csrCudaMatrix;
   csrCudaMatrix = csrHostMatrix;

   CudaVector cudaInVector( csrCudaMatrix.getColumns() ), cudaOutVector( csrCudaMatrix.getRows() );

   CusparseMatrix cusparseMatrix;
   cusparseMatrix.init( csrCudaMatrix, cudaInVector, cudaOutVector, &cusparseHandle );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCusparse = [&]() {
       cusparseMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( hostOutVector, cudaOutVector );
   benchmark.setMetadataElement({ "format", "cusparse" });
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCusparse, cudaBenchmarkResults );

#if defined( HAVE_HYPRE ) && defined( HYPRE_USING_CUDA )
   // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
   if constexpr( std::is_same< HYPRE_Real, Real >::value &&
                 std::is_same< HYPRE_Int, int >::value ) {
      TNL::Hypre hypre;
      using HypreCSR = TNL::Matrices::HypreCSRMatrix;
      HypreCSR hypreCSRMatrix( csrCudaMatrix.getRows(),
                               csrCudaMatrix.getColumns(),
                               csrCudaMatrix.getValues().getView(),
                               csrCudaMatrix.getColumnIndexes().getView(),
                               csrCudaMatrix.getSegments().getOffsets().getView());
      auto cudaInVectorView = cudaInVector.getView();
      auto cudaOutVectorView = cudaOutVector.getView();

      auto spmvHypreCSRCuda = [&]() {
         hypreCSRMatrix.vectorProduct( cudaInVectorView, cudaOutVectorView );
      };

      SpmvBenchmarkResult< Real, Devices::Cuda, int > hypreCudaBenchmarkResults( hostOutVector, cudaOutVector );
      benchmark.setMetadataElement({ "format", "Hypre" });
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvHypreCSRCuda, hypreCudaBenchmarkResults );
   }
   else {
      std::cerr << "Current Real or Index type does not agree with HYPRE_Real or HYPRE_Index." << std::endl;
   }
#endif

#ifdef HAVE_CSR5
   ////
   // Perform benchmark on CUDA device with CSR5 as a reference GPU format
   //
   CudaVector cudaOutVector2( cudaOutVector );
   CSR5Benchmark::CSR5Benchmark< CSRCudaMatrix > csr5Benchmark( csrCudaMatrix, cudaInVector, cudaOutVector );

   auto csr5SpMV = [&]() {
       csr5Benchmark.vectorProduct();
   };

   benchmark.setMetadataElement({ "format", "CSR5" });
   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", csr5SpMV, cudaBenchmarkResults );
   std::cerr << "CSR5 error = " << max( abs( cudaOutVector - cudaOutVector2 ) ) << std::endl;
   csrCudaMatrix.reset();
#endif

// FIXME: LightSpMV fails with CUDA 12
#if __CUDACC_VER_MAJOR__ < 12
   ////
   // Perform benchmark on CUDA device with LightSpMV as a reference GPU format
   //
   using LightSpMVCSRHostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, uint32_t >;
   LightSpMVCSRHostMatrix lightSpMVCSRHostMatrix;
   lightSpMVCSRHostMatrix = csrHostMatrix;
   LightSpMVBenchmark< Real > lightSpMVBenchmark( lightSpMVCSRHostMatrix, LightSpMVBenchmarkKernelVector );
   auto resetLightSpMVVectors = [&]() {
      lightSpMVBenchmark.resetVectors();
   };

   auto spmvLightSpMV = [&]() {
       lightSpMVBenchmark.vectorProduct();
   };
   benchmark.setMetadataElement({ "format", "LightSpMV Vector" });
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "GPU", spmvLightSpMV, cudaBenchmarkResults );

   lightSpMVBenchmark.setKernelType( LightSpMVBenchmarkKernelWarp );
   benchmark.setMetadataElement({ "format", "LightSpMV Warp" });
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "GPU", spmvLightSpMV, cudaBenchmarkResults );
#endif
#endif
   csrHostMatrix.reset();
}

} // namespace TNL::Benchmarks::SpMV
