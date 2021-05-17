/***************************************************************************
                          spmv.h  -  description
                             -------------------
    begin                : Dec 30, 2018
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <cstdint>

#include "../Benchmarks.h"
#include "../JsonLogging.h"
#include "SpmvBenchmarkResult.h"

#include <TNL/Pointers/DevicePointer.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/AdEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/BiEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/LegacyMatrixReader.h>

#include <TNL/Matrices/MatrixInfo.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>

// Comment the following to turn off some groups of SpMV benchmarks and speed-up the compilation
#define WITH_TNL_BENCHMARK_SPMV_GENERAL_MATRICES
#define WITH_TNL_BENCHMARK_SPMV_SYMMETRIC_MATRICES
#define WITH_TNL_BENCHMARK_SPMV_LEGACY_FORMATS

// Uncomment the following line to enable benchmarking the sandbox sparse matrix.
//#define WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#endif

using namespace TNL::Matrices;

#include <Benchmarks/SpMV/ReferenceFormats/cusparseCSRMatrix.h>
#include <Benchmarks/SpMV/ReferenceFormats/cusparseCSRMatrixLegacy.h>
#include <Benchmarks/SpMV/ReferenceFormats/LightSpMVBenchmark.h>
#include <Benchmarks/SpMV/ReferenceFormats/CSR5Benchmark.h>

namespace TNL {
   namespace Benchmarks {
      namespace SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark< JsonLogging >;

/////
// General sparse matrix aliases
//
template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Scalar = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Vector = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Hybrid = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRHybrid >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Adaptive = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRAdaptive >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, EllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, SlicedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_ChunkedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, ChunkedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_BiEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, BiEllpackSegments >;

/////
// Symmetric sparse matrix aliases
//
template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Scalar = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Vector = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRVector >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Hybrid = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRHybrid >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Adaptive = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRAdaptive >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, EllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, SlicedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_ChunkedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, ChunkedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_BiEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, BiEllpackSegments >;

#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
template< typename Real, typename Device, typename Index >
using SparseSandboxMatrix = Matrices::Sandbox::SparseSandboxMatrix< Real, Device, Index, Matrices::GeneralMatrix >;
#endif

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

// Get the name (with extension) of input matrix file
std::string getMatrixFileName( const String& InputFileName )
{
    std::string fileName = InputFileName;

    const size_t last_slash_idx = fileName.find_last_of( "/\\" );
    if( std::string::npos != last_slash_idx )
        fileName.erase( 0, last_slash_idx + 1 );

    return fileName;
}

// Get only the name of the format from getType()
template< typename Matrix >
std::string getMatrixFormat( const Matrix& matrix )
{
    std::string mtrxFullType = getType( matrix );
    std::string mtrxType = mtrxFullType.substr( 0, mtrxFullType.find( "<" ) );
    std::string format = mtrxType.substr( mtrxType.find( ':' ) + 2 );

    return format;
}

template< typename Matrix >
std::string getFormatShort( const Matrix& matrix )
{
    std::string mtrxFullType = getType( matrix );
    std::string mtrxType = mtrxFullType.substr( 0, mtrxFullType.find( "<" ) );
    std::string format = mtrxType.substr( mtrxType.find( ':' ) + 2 );
    format = format.substr( format.find(':') + 2);
    format = format.substr( 0, 3 );

    return format;
}

// Print information about the matrix.
/*template< typename Matrix >
void printMatrixInfo( const Matrix& matrix,
                      std::ostream& str )
{
    str << "\n Format: " << getMatrixFormat( matrix ) << std::endl;
    str << " Rows: " << matrix.getRows() << std::endl;
    str << " Cols: " << matrix.getColumns() << std::endl;
    str << " Nonzero Elements: " << matrix.getNumberOfNonzeroMatrixElements() << std::endl;
}*/

template< typename Real,
          template< typename, typename, typename > class Matrix,
          template< typename, typename, typename, typename > class Vector = Containers::Vector >
void
benchmarkSpMVLegacy( BenchmarkType& benchmark,
                     const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                     const String& inputFileName,
                     bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   HostMatrix hostMatrix;
   CudaMatrix cudaMatrix;

   SpMV::ReferenceFormats::Legacy::LegacyMatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix, verboseMR );

   /*benchmark.setMetadataColumns( BenchmarkType::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         { "rows", convertToString( hostMatrix.getRows() ) },
         { "columns", convertToString( hostMatrix.getColumns() ) },
         { "matrix format", MatrixInfo< HostMatrix >::getFormat() }
      } ));*/
   const int elements = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setOperation( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( hostInVector, hostOutVector );

   };
   SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( MatrixInfo< HostMatrix >::getFormat(), csrResultVector, hostOutVector, hostMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef HAVE_CUDA
   cudaMatrix = hostMatrix;
   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( MatrixInfo< HostMatrix >::getFormat(), csrResultVector, cudaOutVector, cudaMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
   // std::cout << std::endl;
}

template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix,
          template< typename, typename, typename, typename > class Vector = Containers::Vector >
void
benchmarkSpMV( BenchmarkType& benchmark,
               const InputMatrix& inputMatrix,
               const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
               const String& inputFileName,
               bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      std::cerr << "Unable to convert the matrix to the target format." << std::endl;
      return;
   }

   /*benchmark.setMetadataColumns( BenchmarkType::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         { "rows", convertToString( hostMatrix.getRows() ) },
         { "columns", convertToString( hostMatrix.getColumns() ) },
         { "matrix format", MatrixInfo< HostMatrix >::getFormat() }
      } ));*/
   const int elements = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setOperation( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( hostInVector, hostOutVector );

   };
   SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( MatrixInfo< HostMatrix >::getFormat(), csrResultVector, hostOutVector, hostMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef HAVE_CUDA
   CudaMatrix cudaMatrix;
   cudaMatrix = inputMatrix;
   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( MatrixInfo< HostMatrix >::getFormat(), csrResultVector, cudaOutVector, cudaMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
   // std::cout << std::endl;
}

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   // The following is another workaround because of a bug in nvcc versions 10 and 11.
   // If we use the current matrix formats, not the legacy ones, we get
   // ' error: redefinition of ‘void TNL::Algorithms::__wrapper__device_stub_CudaReductionKernel...'
   // It seems that there is a problem with lambda functions identification when we create
   // two instances of TNL::Matrices::SparseMatrix. The second one comes from calling of
   // `benchmarkSpMV< Real, SparseMatrix_CSR_Scalar >( benchmark, hostOutVector, inputFileName, verboseMR );`
   // and simillar later in this function.
#define USE_LEGACY_FORMATS
#ifdef USE_LEGACY_FORMATS
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Host, int >;
   using CSRCudaMatrix = SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Cuda, int >;
   using CusparseMatrix = TNL::CusparseCSRLegacy< Real >;
   using LightSpMVCSRHostMatrix = SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Host, uint32_t >;
#else
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;
   using CSRCudaMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Cuda, int >;
   using CusparseMatrix = TNL::CusparseCSR< Real >;
#endif


   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   CSRHostMatrix csrHostMatrix;

   ////
   // Set-up benchmark datasize
   //
   MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
   const int elements = csrHostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setOperation( datasetSize );

   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   benchmark.addCommonLogs( BenchmarkType::CommonLogs( {
      { "matrix name", convertToString( inputFileName ) },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) } } ) );

   /*benchmark.setMetadataColumns( BenchmarkType::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         { "rows", convertToString( csrHostMatrix.getRows() ) },
         { "columns", convertToString( csrHostMatrix.getColumns() ) },
         { "matrix format", String( "CSR" ) }
      } ));*/

   HostVector hostInVector( csrHostMatrix.getRows() ), hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvCSRHost = [&]() {
       csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > csrBenchmarkResults( String( "CSR" ), hostOutVector, hostOutVector, csrHostMatrix.getNonzeroElementsCount() );
   benchmark.addLogsMetadata( csrBenchmarkResults.getTableHeader() );
   benchmark.writeHeader();
   benchmark.time< Devices::Host >( resetHostVectors, "", spmvCSRHost, csrBenchmarkResults );

#ifdef HAVE_CUDA
   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
   /*benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         { "rows", convertToString( csrHostMatrix.getRows() ) },
         { "columns", convertToString( csrHostMatrix.getColumns() ) },
         { "matrix format", String( "cuSparse" ) }
      } ));*/

   cusparseHandle_t cusparseHandle;
   cusparseCreate( &cusparseHandle );

   CSRCudaMatrix csrCudaMatrix;
   csrCudaMatrix = csrHostMatrix;

   CusparseMatrix cusparseMatrix;
   cusparseMatrix.init( csrCudaMatrix, &cusparseHandle );

   CudaVector cudaInVector( csrCudaMatrix.getColumns() ), cudaOutVector( csrCudaMatrix.getRows() );

   auto resetCusparseVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCusparse = [&]() {
       cusparseMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( String( "cusparse" ), hostOutVector, cudaOutVector, csrHostMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", spmvCusparse, cudaBenchmarkResults );

#ifdef HAVE_CSR5
   ////
   // Perform benchmark on CUDA device with CSR5 as a reference GPU format
   //
   cudaBenchmarkResults.setFormat( String( "CSR5" ) );
   /*benchmark.setMetadataColumns( Benchmark::MetadataColumns({
      { "matrix name", convertToString( inputFileName ) },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "matrix format", String( "CSR5" ) }
   } ));*/

   CudaVector cudaOutVector2( cudaOutVector );
   CSR5Benchmark::CSR5Benchmark< CSRCudaMatrix > csr5Benchmark( csrCudaMatrix, cudaInVector, cudaOutVector );

   auto csr5SpMV = [&]() {
       csr5Benchmark.vectorProduct();
   };

   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", csr5SpMV, cudaBenchmarkResults );
   std::cerr << "CSR5 error = " << max( abs( cudaOutVector - cudaOutVector2 ) ) << std::endl;
   csrCudaMatrix.reset();
#endif

   ////
   // Perform benchmark on CUDA device with LightSpMV as a reference GPU format
   //
   cudaBenchmarkResults.setFormat( String( "LightSpMV Vector" ) );
   /*benchmark.setMetadataColumns( Benchmark::MetadataColumns({
      { "matrix name", convertToString( inputFileName ) },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "matrix format", String( "LightSpMV Vector" ) }
   } ));*/

   LightSpMVCSRHostMatrix lightSpMVCSRHostMatrix;
   lightSpMVCSRHostMatrix = csrHostMatrix;
   LightSpMVBenchmark< Real > lightSpMVBenchmark( lightSpMVCSRHostMatrix, LightSpMVBenchmarkKernelVector );
   auto resetLightSpMVVectors = [&]() {
      lightSpMVBenchmark.resetVectors();
   };

   auto spmvLightSpMV = [&]() {
       lightSpMVBenchmark.vectorProduct();
   };
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "GPU", spmvLightSpMV, cudaBenchmarkResults );

   cudaBenchmarkResults.setFormat( String( "LightSpMV Warp" ) );
   /*benchmark.setMetadataColumns( Benchmark::MetadataColumns({
      { "matrix name", convertToString( inputFileName ) },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "matrix format", String( "LightSpMV Warp" ) }
   } ));*/
   lightSpMVBenchmark.setKernelType( LightSpMVBenchmarkKernelWarp );
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "GPU", spmvLightSpMV, cudaBenchmarkResults );
#endif
   csrHostMatrix.reset();

#ifdef WITH_TNL_BENCHMARK_SPMV_LEGACY_FORMATS
   /////
   // Benchmarking of TNL legacy formats
   //
   if( parameters.getParameter< bool >("with-legacy-matrices") )
   {
      using namespace Benchmarks::SpMV::ReferenceFormats;
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Scalar             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Vector             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light              >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light2             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light3             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light4             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light5             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light6             >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Adaptive           >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_MultiVector        >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_LightWithoutAtomic >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::Ellpack                           >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, SlicedEllpackAlias                        >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::ChunkedEllpack                    >( benchmark, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMVLegacy< Real, Legacy::BiEllpack                         >( benchmark, hostOutVector, inputFileName, verboseMR );
   }
   // AdEllpack is broken
   //benchmarkSpMV< Real, Matrices::AdEllpack              >( benchmark, hostOutVector, inputFileName, verboseMR );
#endif

#ifdef WITH_TNL_BENCHMARK_SPMV_GENERAL_MATRICES
   /////
   // Benchmarking TNL formats
   //
   using HostMatrixType = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host >;
   HostMatrixType hostMatrix;
   TNL::Matrices::MatrixReader< HostMatrixType >::readMtx( inputFileName, hostMatrix, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Scalar                   >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Vector                   >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Hybrid                   >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Adaptive                 >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_Ellpack                      >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_SlicedEllpack                >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_ChunkedEllpack               >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_BiEllpack                    >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
   benchmarkSpMV< Real, HostMatrixType, SparseSandboxMatrix                       >( benchmark, hostMatrix, hostOutVector, inputFileName, verboseMR );
#endif
   hostMatrix.reset();
#endif

#ifdef WITH_TNL_BENCHMARK_SPMV_SYMMETRIC_MATRICES
   /////
   // Benchmarking symmetric sparse matrices
   //
   if( parameters.getParameter< bool >("with-symmetric-matrices") )
   {
      using SymmetricInputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >;
      using InputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;
      SymmetricInputMatrix symmetricHostMatrix;
      try
      {
         TNL::Matrices::MatrixReader< SymmetricInputMatrix >::readMtx( inputFileName, symmetricHostMatrix, verboseMR );
      }
      catch(const std::exception& e)
      {
         std::cerr << e.what() << " ... SKIPPING " << std::endl;
         return;
      }
      InputMatrix hostMatrix;
      TNL::Matrices::MatrixReader< InputMatrix >::readMtx( inputFileName, hostMatrix, verboseMR );
      if( hostMatrix != symmetricHostMatrix )
      {
         std::cerr << "ERROR: Symmetric matrices do not match !!!" << std::endl;
      }
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Scalar                   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Vector                   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Hybrid                   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Adaptive                 >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_Ellpack                      >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_SlicedEllpack                >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_ChunkedEllpack               >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
      benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_BiEllpack                    >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, verboseMR );
   }
#endif
}

} // namespace SpMVLegacy
} // namespace Benchmarks
} // namespace TNL
