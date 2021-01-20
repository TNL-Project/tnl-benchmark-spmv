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

#include "../Benchmarks.h"
#include "SpmvBenchmarkResult.h"

#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Matrices/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>
#include <TNL/Matrices/Legacy/AdEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/BiEllpack.h>

#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixInfo.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
using namespace TNL::Matrices;

#include <Benchmarks/SpMV/ReferenceFormats/cusparseCSRMatrix.h>

namespace TNL {
   namespace Benchmarks {
      namespace SpMVLegacy {

// Alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpackAlias = Matrices::Legacy::SlicedEllpack< Real, Device, Index >;

// Segments based sparse matrix aliases
template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRDefault >;

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

// Legacy formats
template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Scalar = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Vector = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLight >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light2 = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLight2 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light3 = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLight3 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light4 = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLight4 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light5 = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLight5 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light6 = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLight6 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Adaptive = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRAdaptive >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_MultiVector = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRMultiVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_LightWithoutAtomic = Matrices::Legacy::CSR< Real, Device, Index, Matrices::Legacy::CSRLightWithoutAtomic >;

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
template< typename Matrix >
void printMatrixInfo( const Matrix& matrix,
                      std::ostream& str )
{
    str << "\n Format: " << getMatrixFormat( matrix ) << std::endl;
    str << " Rows: " << matrix.getRows() << std::endl;
    str << " Cols: " << matrix.getColumns() << std::endl;
    str << " Nonzero Elements: " << matrix.getNumberOfNonzeroMatrixElements() << std::endl;
}

template< typename Real,
          template< typename, typename, typename > class Matrix,
          template< typename, typename, typename, typename > class Vector = Containers::Vector >
void
benchmarkSpMV( Benchmark& benchmark,
               const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
               const String& inputFileName,
               bool verboseMR )
{
   using HostMatrix = Matrix< Real, Devices::Host, int >;
   using CudaMatrix = Matrix< Real, Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   HostMatrix hostMatrix;
   CudaMatrix cudaMatrix;

   MatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix, verboseMR );

   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         //{ "non-zeros", convertToString( hostMatrix.getNonzeroElementsCount() ) },
         { "rows", convertToString( hostMatrix.getRows() ) },
         { "columns", convertToString( hostMatrix.getColumns() ) },
         { "matrix format", MatrixInfo< HostMatrix >::getFormat() }
      } ));
   const int elements = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setOperation( datasetSize );

   /***
    * Benchmark SpMV on host
    */
   HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( hostInVector, hostOutVector );

   };
   SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector, hostMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );

   /***
    * Benchmark SpMV on CUDA
    */
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
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector, cudaMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
    std::cout << std::endl;
}

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmvSynthetic( Benchmark& benchmark,
                        const String& inputFileName,
                        bool verboseMR )
{
   using CSRHostMatrix = Matrices::Legacy::CSR< Real, Devices::Host, int >;
   using CSRCudaMatrix = Matrices::Legacy::CSR< Real, Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   CSRHostMatrix csrHostMatrix;
   CSRCudaMatrix csrCudaMatrix;

   ////
   // Set-up benchmark datasize
   //
   MatrixReader< CSRHostMatrix >::readMtxFile( inputFileName, csrHostMatrix, verboseMR );
   const int elements = csrHostMatrix.getNumberOfNonzeroMatrixElements();
   const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setOperation( datasetSize );

   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         //{ "non-zeros", convertToString( csrHostMatrix.getNumberOfNonzeroMatrixElements() ) },
         { "rows", convertToString( csrHostMatrix.getRows() ) },
         { "columns", convertToString( csrHostMatrix.getColumns() ) },
         { "matrix format", String( "CSR" ) }
      } ));

   HostVector hostInVector( csrHostMatrix.getRows() ), hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvCSRHost = [&]() {
       csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > csrBenchmarkResults( hostOutVector, hostOutVector, csrHostMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Cuda >( resetHostVectors, "CPU", spmvCSRHost, csrBenchmarkResults );

   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
#ifdef HAVE_CUDA
   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "matrix name", convertToString( inputFileName ) },
         //{ "non-zeros", convertToString( csrHostMatrix.getNumberOfNonzeroMatrixElements() ) },
         { "rows", convertToString( csrHostMatrix.getRows() ) },
         { "columns", convertToString( csrHostMatrix.getColumns() ) },
         { "matrix format", String( "cuSparse" ) }
      } ));

   cusparseHandle_t cusparseHandle;
   cusparseCreate( &cusparseHandle );

   csrCudaMatrix = csrHostMatrix;

   // Delete the CSRhostMatrix, so it doesn't take up unnecessary space
   csrHostMatrix.reset();

   TNL::CusparseCSR< Real > cusparseMatrix;
   cusparseMatrix.init( csrCudaMatrix, &cusparseHandle );

   CudaVector cusparseInVector( csrCudaMatrix.getColumns() ), cusparseOutVector( csrCudaMatrix.getRows() );

   auto resetCusparseVectors = [&]() {
      cusparseInVector = 1.0;
      cusparseOutVector = 0.0;
   };

   auto spmvCusparse = [&]() {
       cusparseMatrix.vectorProduct( cusparseInVector, cusparseOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > cusparseBenchmarkResults( hostOutVector, hostOutVector, csrHostMatrix.getNonzeroElementsCount() );
   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", spmvCusparse, cusparseBenchmarkResults );
#endif

   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Scalar    >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Vector    >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Light     >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Light2     >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Light3     >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Light4     >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Light5     >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Light6     >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_Adaptive  >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_MultiVector>( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrixLegacy_CSR_LightWithoutAtomic>( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_CSR                 >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, Matrices::Legacy::Ellpack        >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_Ellpack             >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SlicedEllpackAlias               >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_SlicedEllpack       >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, Matrices::Legacy::ChunkedEllpack >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_ChunkedEllpack      >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, Matrices::Legacy::BiEllpack      >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_BiEllpack           >( benchmark, hostOutVector, inputFileName, verboseMR );
   /* AdEllpack is broken
   benchmarkSpMV< Real, Matrices::AdEllpack              >( benchmark, hostOutVector, inputFileName, verboseMR );
    */
}

} // namespace SpMVLegacy
} // namespace Benchmarks
} // namespace TNL
