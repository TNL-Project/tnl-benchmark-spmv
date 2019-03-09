/***************************************************************************
                          spmv.h  -  description
                             -------------------
    begin                : Dec 30, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "../Benchmarks.h"

#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>
#include <TNL/Matrices/ChunkedEllpack.h>

#include <TNL/Matrices/MatrixReader.h>
using namespace TNL::Matrices;

namespace TNL {
namespace Benchmarks {

// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpack = Matrices::SlicedEllpack< Real, Device, Index >;

// Get only the name of the format from getType().
template< typename Matrix >
std::string getMatrixFormat( const Matrix& matrix )
{
    std::string mtrxFullType = matrix.getType();
    std::string mtrxType = mtrxFullType.substr(0, mtrxFullType.find("<"));
    std::string format = mtrxType.substr(mtrxType.find(':') + 2);
    
    return format;
}

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
          template< typename, typename, typename > class Vector = Containers::Vector >
bool
benchmarkSpMV( Benchmark & benchmark,
               const String & inputFileName )
{
    typedef Matrix< Real, Devices::Host, int > HostMatrix;
    typedef Matrix< Real, Devices::Cuda, int > DeviceMatrix;
    typedef Containers::Vector< Real, Devices::Host, int > HostVector;
    typedef Containers::Vector< Real, Devices::Cuda, int > CudaVector;
    
    HostMatrix hostMatrix;
    DeviceMatrix deviceMatrix;
    HostVector hostVector, hostVector2;
    CudaVector deviceVector, deviceVector2;
    
    try
      {
         if( ! MatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix ) )
         {
            std::cerr << "Failed to read the matrix file " << inputFileName << "." << std::endl;
            
            std::string matrixFormat = getMatrixFormat( hostMatrix );
            
            std::string stringErrorMsg = "Failed to read the matrix file " + 
                                         ( std::string )inputFileName + ".\n" + 
                                         "matrix format: " + matrixFormat + 
                                         "\nBenchmark failed: Unable to read the matrix.";
            
            char *errorMsg = &stringErrorMsg[0u];
            
            benchmark.addErrorMessage( errorMsg, 3 );
            return false;
         }
      }
      catch( std::bad_alloc )
      {
         std::cerr << "Failed to allocate memory to read the matrix file " << inputFileName << "." << std::endl;
         
         std::string matrixFormat = getMatrixFormat( hostMatrix );
         
         std::string stringErrorMsg = "Failed to allocate memory to read the matrix file " +
                                      ( std::string )inputFileName + ".\n" +
                                      "matrix format: " + matrixFormat + 
                                      "\nBenchmark failed: Not enough memory.";
         
         char *errorMsg = &stringErrorMsg[0u];
         
         benchmark.addErrorMessage( errorMsg, 3 );
         return false;
      }
    // printMatrixInfo is redundant, because all the information is in the Benchmark's MetadataColumns.
//    printMatrixInfo( hostMatrix, std::cout );
#ifdef HAVE_CUDA
    // FIXME: This doesn't work for ChunkedEllpack, because
    //        its cross-device assignment is not implemented yet.
    deviceMatrix = hostMatrix;
#endif

    benchmark.setMetadataColumns( Benchmark::MetadataColumns({
          { "matrix format", convertToString( getMatrixFormat( hostMatrix ) ) },
          { "non-zeros", convertToString( hostMatrix.getNumberOfNonzeroMatrixElements() ) },
          { "rows", convertToString( hostMatrix.getRows() ) },
          { "columns", convertToString( hostMatrix.getColumns() ) }
       } ));

    hostVector.setSize( hostMatrix.getColumns() );
    hostVector2.setSize( hostMatrix.getRows() );

#ifdef HAVE_CUDA
    deviceVector.setSize( hostMatrix.getColumns() );
    deviceVector2.setSize( hostMatrix.getRows() );
#endif

    // reset function
    auto reset = [&]() {
       hostVector.setValue( 1.0 );
       hostVector2.setValue( 0.0 );
 #ifdef HAVE_CUDA
       deviceVector.setValue( 1.0 );
       deviceVector2.setValue( 0.0 );
 #endif
    };

    const int elements = hostMatrix.getNumberOfNonzeroMatrixElements();

    const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;

    // compute functions
    auto spmvHost = [&]() {
       hostMatrix.vectorProduct( hostVector, hostVector2 );
    };
    auto spmvCuda = [&]() {
       deviceMatrix.vectorProduct( deviceVector, deviceVector2 );
    };

    benchmark.setOperation( datasetSize );
    benchmark.time< Devices::Host >( reset, "CPU", spmvHost );
 #ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU", spmvCuda );
 #endif
    std::cout << std::endl;
    return true;
}

template< typename Real = double,
          typename Index = int >
bool
benchmarkSpmvSynthetic( Benchmark & benchmark,
                        const String& inputFileName )
{
   bool result = true;
   // TODO: benchmark all formats from tnl-benchmark-spmv (different parameters of the base formats)
   result |= benchmarkSpMV< Real, Matrices::CSR >( benchmark, inputFileName );
   result |= benchmarkSpMV< Real, Matrices::Ellpack >( benchmark, inputFileName );
   result |= benchmarkSpMV< Real, SlicedEllpack >( benchmark, inputFileName );
//   result |= benchmarkSpMV< Real, Matrices::ChunkedEllpack >( benchmark, inputFileName );
   return result;
}

} // namespace Benchmarks
} // namespace TNL
