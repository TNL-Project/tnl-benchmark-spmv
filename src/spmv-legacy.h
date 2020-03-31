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
#include <TNL/Matrices/Legacy/Ellpack.h>
#include <TNL/Matrices/Legacy/SlicedEllpack.h>
#include <TNL/Matrices/Legacy/ChunkedEllpack.h>
#include <TNL/Matrices/Legacy/AdEllpack.h>
#include <TNL/Matrices/Legacy/BiEllpack.h>

#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/MatrixInfo.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Containers/Segments/CSR.h>
#include <TNL/Containers/Segments/Ellpack.h>
#include <TNL/Containers/Segments/SlicedEllpack.h>
using namespace TNL::Matrices;

#include "cusparseCSRMatrix.h"

namespace TNL {
   namespace Benchmarks {
      namespace SpMVLegacy {

// Alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpackAlias = Matrices::Legacy::SlicedEllpack< Real, Device, Index >;

// Segments based sparse matrix aliases
template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Containers::Segments::CSR >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Containers::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, EllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, SlicedEllpackSegments >;

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
         { "non-zeros", convertToString( hostMatrix.getNumberOfNonzeroMatrixElements() ) },
         { "rows", convertToString( hostMatrix.getRows() ) },
         { "columns", convertToString( hostMatrix.getColumns() ) },
         { "matrix format", MatrixInfo< HostMatrix >::getFormat() }
      } ));
   const int elements = hostMatrix.getNumberOfNonzeroMatrixElements();
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
   SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
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
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
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
         { "matrix name", convertToString( getMatrixFileName( inputFileName ) ) },
         { "non-zeros", convertToString( csrHostMatrix.getNumberOfNonzeroMatrixElements() ) },
         { "rows", convertToString( csrHostMatrix.getRows() ) },
         { "columns", convertToString( csrHostMatrix.getColumns() ) },
         { "matrix format", String( "CSR" ) }
      } ));

   HostVector hostInVector( csrHostMatrix.getRows() ), hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector == 0.0;
   };

   auto spmvCSRHost = [&]() {
       csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   benchmark.time< Devices::Cuda >( resetHostVectors, "CPU", spmvCSRHost );

   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
#ifdef HAVE_CUDA
   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "matrix name", convertToString( getMatrixFileName( inputFileName ) ) },
         { "non-zeros", convertToString( csrHostMatrix.getNumberOfNonzeroMatrixElements() ) },
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
      cusparseOutVector == 0.0;
   };

   auto spmvCusparse = [&]() {
       cusparseMatrix.vectorProduct( cusparseInVector, cusparseOutVector );
   };

   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", spmvCusparse );
#endif

   benchmarkSpMV< Real, Matrices::Legacy::CSR            >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_CSR                 >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, Matrices::Legacy::Ellpack        >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_Ellpack             >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SlicedEllpackAlias               >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, SparseMatrix_SlicedEllpack       >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, Matrices::Legacy::ChunkedEllpack >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMV< Real, Matrices::Legacy::BiEllpack      >( benchmark, hostOutVector, inputFileName, verboseMR );
   /* AdEllpack is broken
   benchmarkSpMV< Real, Matrices::AdEllpack              >( benchmark, hostOutVector, inputFileName, verboseMR );
    */
}

} // namespace SpMVLegacy
} // namespace Benchmarks
} // namespace TNL
