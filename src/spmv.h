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

#include <cusparse.h>
#include "cusparseCSRMatrix.h"
using namespace TNL;

namespace TNL {
namespace Benchmarks {

// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpack = Matrices::SlicedEllpack< Real, Device, Index >;

// Get only the name of the format from getType()
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
         // Start a buffer to capture the output of MatrixReader
         std::stringstream buffer;
         std::streambuf * old = std::cerr.rdbuf( buffer.rdbuf() );
         
         if( ! MatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix ) )
         {
            // Capture the original output of MatrixReader, so it isn't printed by console.
            std::string errorMsgBuffer = buffer.str();
            // Reset the buffer
            std::cerr.rdbuf( old );
            
            // WHY DID I CAPTURE THE ERROR MESSAGE ONLY TO RUN MatrixReader again? Use the above capture to print into log and console?
            
             
            std::string matrixFormat = getMatrixFormat( hostMatrix );
            
            //https://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string
            std::stringstream buffer;
            std::streambuf * old = std::cerr.rdbuf( buffer.rdbuf() );

            MatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix );

            errorMsgBuffer = buffer.str();
            
            // Reset the buffer
            std::cerr.rdbuf( old );
            
            std::string stringErrorMsg = "Benchmark failed: Unable to read the matrix.\n"
                                         "matrix format: " + matrixFormat +
                                         "\nFailed to read the matrix file " + 
                                         ( std::string )inputFileName + ".\n" + 
                                         errorMsgBuffer;
            
            //https://stackoverflow.com/questions/1488775/c-remove-new-line-from-multiline-string
            if ( ! stringErrorMsg.empty() && stringErrorMsg[ stringErrorMsg.length() - 1 ] == '\n' )
                stringErrorMsg.erase( stringErrorMsg.length() - 1 );
            
            // https://stackoverflow.com/questions/7352099/stdstring-to-char
            char* errorMsg = &stringErrorMsg[ 0u ];
            
            
            // FIXME: Every other benchmark, the errorMsg doesn't have a "!" as 
            //        a prefix in the log file. 
            //        (Try adding more benchmarks in benchmarkSpmvSynthetic(...) 
            //         and you'll see)
            benchmark.addErrorMessage( errorMsg, 1 );
            
            std::cout << std::endl;
            
            return false;
         }
         std::cerr.rdbuf( old );
      }
      catch( std::bad_alloc )
      {
         std::string matrixFormat = getMatrixFormat( hostMatrix );
         
         //https://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string
         std::stringstream buffer;
         std::streambuf * old = std::cerr.rdbuf(buffer.rdbuf());

         MatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix );

         std::string errorMsgBuffer = buffer.str();
         
         // Reset the buffer
         std::cerr.rdbuf( old );
          
         std::string stringErrorMsg = "Benchmark failed: Not enough memory.\n"
                                      "matrix format: " + matrixFormat + 
                                      "\nFailed to allocate memory to read the matrix file " +
                                      ( std::string )inputFileName + ".\n" + 
                                      errorMsgBuffer;
         
         //https://stackoverflow.com/questions/1488775/c-remove-new-line-from-multiline-string
         if ( ! stringErrorMsg.empty() && stringErrorMsg[ stringErrorMsg.length() - 1 ] == '\n' )
                stringErrorMsg.erase( stringErrorMsg.length() - 1 );
         
         // https://stackoverflow.com/questions/7352099/stdstring-to-char
         char *errorMsg = &stringErrorMsg[ 0u ];
         
         // FIXME: Every other benchmark, the errorMsg doesn't have a "!" as 
         //        a prefix in the log file. 
         //        (Try adding more benchmarks in benchmarkSpmvSynthetic(...) 
         //         and you'll see)
         benchmark.addErrorMessage( errorMsg, 1 );
         
         std::cout << std::endl;
         
         return false;
      }
    // printMatrixInfo is redundant, because all the information is in the Benchmark's MetadataColumns
//    printMatrixInfo( hostMatrix, std::cout );
#ifdef HAVE_CUDA
    // FIXME: This doesn't work for ChunkedEllpack, because
    //        its cross-device assignment is not implemented yet
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
    
    // Initialize the host vector to be compared. (The values in hostVector2 will be reset when spmvCuda starts)
    HostVector resultHostVector2;
    resultHostVector2.setSize( hostVector2.getSize() );
    resultHostVector2.setValue( 0.0 );
    // Copy the values
    for( int i = 0; i < hostVector2.getSize(); i++ )
        resultHostVector2.setElement( i, hostVector2.getElement( i ) );
    
 #ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU", spmvCuda );
 #endif

    // Setup the device vector to be compared
    HostVector resultDeviceVector2;
    resultDeviceVector2.setSize( hostVector2.getSize() );
    resultDeviceVector2.setValue( 0.0 );
    
//    resultDeviceVector2 += deviceVector2; // Throws a segfault.
    
    // Copy the values
    for( int i = 0; i < deviceVector2.getSize(); i++ )
        resultDeviceVector2.setElement( i, deviceVector2.getElement( i ) );
    
    Real differenceAbsMax = resultHostVector2.differenceAbsMax( resultDeviceVector2 );
    Real differenceLpNorm = resultHostVector2.differenceLpNorm( resultDeviceVector2, 1 );
    
    std::string resultDifferenceAbsMax = "differenceAbsMax = " + std::to_string( differenceAbsMax );
    std::string resultDifferenceLpNorm = "differenceLpNorm = " + std::to_string( differenceLpNorm );
    
    char *absMax = &resultDifferenceAbsMax[ 0u ];
    char *lpNorm = &resultDifferenceLpNorm[ 0u ];
    
    // FIXME: THIS ISN'T AN ELEGANT SOLUTION, IT MAKES THE LOG FILE VERY LONG
//    benchmark.addErrorMessage( absMax, 1 );
//    benchmark.addErrorMessage( lpNorm, 1 );
    
    std::cout << std::endl;
    return true;
}

// Compares only CSR on GPU and Cusparse on GPU.
template< typename Real,
          template< typename, typename, typename > class Vector = Containers::Vector >
bool
benchmarkCusparseSpMV( Benchmark & benchmark,
               const String & inputFileName )
{    
    typedef Matrices::CSR< Real, Devices::Host, int > CSR_HostMatrix;
    typedef Matrices::CSR< Real, Devices::Cuda, int > CSR_DeviceMatrix;
    typedef Containers::Vector< Real, Devices::Host, int > HostVector;
    typedef Containers::Vector< Real, Devices::Cuda, int > CudaVector;
    
    CSR_HostMatrix CSRhostMatrix;
    CSR_DeviceMatrix CSRdeviceMatrix;
    CudaVector deviceVector, deviceVector2;
    
    try
      {
         // Start a buffer to capture the output of MatrixReader
         std::stringstream buffer;
         std::streambuf * old = std::cerr.rdbuf( buffer.rdbuf() );
         
         if( ! MatrixReader< CSR_HostMatrix >::readMtxFile( inputFileName, CSRhostMatrix ) )
         {
            // Capture the original output of MatrixReader, so it isn't printed by console.
            std::string errorMsgBuffer = buffer.str();
            // Reset the buffer
            std::cerr.rdbuf( old );
            
             
            std::string matrixFormat = getMatrixFormat( CSRhostMatrix );
            
            //https://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string
            std::stringstream buffer;
            std::streambuf * old = std::cerr.rdbuf( buffer.rdbuf() );

            MatrixReader< CSR_HostMatrix >::readMtxFile( inputFileName, CSRhostMatrix );

            errorMsgBuffer = buffer.str();
            
            // Reset the buffer
            std::cerr.rdbuf( old );
            
            std::string stringErrorMsg = "Benchmark failed: Unable to read the matrix.\n"
                                         "matrix format: " + matrixFormat +
                                         "\nFailed to read the matrix file " + 
                                         ( std::string )inputFileName + ".\n" + 
                                         errorMsgBuffer;
            
            //https://stackoverflow.com/questions/1488775/c-remove-new-line-from-multiline-string
            if ( ! stringErrorMsg.empty() && stringErrorMsg[ stringErrorMsg.length() - 1 ] == '\n' )
                stringErrorMsg.erase( stringErrorMsg.length() - 1 );
            
            // https://stackoverflow.com/questions/7352099/stdstring-to-char
            char* errorMsg = &stringErrorMsg[ 0u ];
            
            
            // FIXME: Every other benchmark, the errorMsg doesn't have a "!" as 
            //        a prefix in the log file. 
            //        (Try adding more benchmarks in benchmarkSpmvSynthetic(...) 
            //         and you'll see)
            benchmark.addErrorMessage( errorMsg, 1 );
            
            std::cout << std::endl;
            
            return false;
         }
         std::cerr.rdbuf( old );
      }
      catch( std::bad_alloc )
      {
         std::string matrixFormat = getMatrixFormat( CSRhostMatrix );
         
         //https://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string
         std::stringstream buffer;
         std::streambuf * old = std::cerr.rdbuf(buffer.rdbuf());

         MatrixReader< CSR_HostMatrix >::readMtxFile( inputFileName, CSRhostMatrix );

         std::string errorMsgBuffer = buffer.str();
         
         // Reset the buffer
         std::cerr.rdbuf( old );
          
         std::string stringErrorMsg = "Benchmark failed: Not enough memory.\n"
                                      "matrix format: " + matrixFormat + 
                                      "\nFailed to allocate memory to read the matrix file " +
                                      ( std::string )inputFileName + ".\n" + 
                                      errorMsgBuffer;
         
         //https://stackoverflow.com/questions/1488775/c-remove-new-line-from-multiline-string
         if ( ! stringErrorMsg.empty() && stringErrorMsg[ stringErrorMsg.length() - 1 ] == '\n' )
                stringErrorMsg.erase( stringErrorMsg.length() - 1 );
         
         // https://stackoverflow.com/questions/7352099/stdstring-to-char
         char *errorMsg = &stringErrorMsg[ 0u ];
         
         // FIXME: Every other benchmark, the errorMsg doesn't have a "!" as 
         //        a prefix in the log file. 
         //        (Try adding more benchmarks in benchmarkSpmvSynthetic(...) 
         //         and you'll see)
         benchmark.addErrorMessage( errorMsg, 1 );
         
         std::cout << std::endl;
         
         return false;
      }
    
    benchmark.setMetadataColumns( Benchmark::MetadataColumns({
          { "matrix format", convertToString( getMatrixFormat( CSRhostMatrix ) ) },
          { "non-zeros", convertToString( CSRhostMatrix.getNumberOfNonzeroMatrixElements() ) },
          { "rows", convertToString( CSRhostMatrix.getRows() ) },
          { "columns", convertToString( CSRhostMatrix.getColumns() ) }
       } ));
    
    cusparseHandle_t cusparseHandle;
    cusparseCreate( &cusparseHandle );
    
#ifdef HAVE_CUDA
    // FIXME: This doesn't work for ChunkedEllpack, because
    //        its cross-device assignment is not implemented yet
    CSRdeviceMatrix = CSRhostMatrix;
    
    TNL::CusparseCSR< Real > cusparseCSR;
    cusparseCSR.init( CSRdeviceMatrix, &cusparseHandle );
#endif

#ifdef HAVE_CUDA
    deviceVector.setSize( CSRhostMatrix.getColumns() );
    deviceVector2.setSize( CSRhostMatrix.getRows() );
#endif

    // reset function
    auto reset = [&]() {
 #ifdef HAVE_CUDA
       deviceVector.setValue( 1.0 );
       deviceVector2.setValue( 0.0 );
 #endif
    };

    const int elements = CSRhostMatrix.getNumberOfNonzeroMatrixElements();

    const double datasetSize = (double) elements * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;

    // compute functions
    auto spmvCuda = [&]() {
       CSRdeviceMatrix.vectorProduct( deviceVector, deviceVector2 );
    };
    auto spmvCusparse = [&]() {
        cusparseCSR.vectorProduct( deviceVector, deviceVector2 );
    };

    benchmark.setOperation( datasetSize );
    
 #ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU", spmvCuda );
    
    // Initialize the cuda vector to be compared. (The values in hostVector2 will be reset when spmvCuda starts)
    HostVector resultCusparseVector2;
    resultCusparseVector2.setSize( deviceVector2.getSize() );
    resultCusparseVector2.setValue( 0.0 );
    // Copy the values
    for( int i = 0; i < deviceVector2.getSize(); i++ )
        resultCusparseVector2.setElement( i, deviceVector2.getElement( i ) );
    
    benchmark.time< Devices::Cuda >( reset, "GPU-Cusparse", spmvCusparse );
 #endif

    // Setup the device vector to be compared
    HostVector resultDeviceVector2;
    resultDeviceVector2.setSize( resultCusparseVector2.getSize() );
    resultDeviceVector2.setValue( 0.0 );
    
    // Copy the values
    for( int i = 0; i < deviceVector2.getSize(); i++ )
        resultDeviceVector2.setElement( i, deviceVector2.getElement( i ) );
    
    Real differenceAbsMax = resultCusparseVector2.differenceAbsMax( resultDeviceVector2 );
    Real differenceLpNorm = resultCusparseVector2.differenceLpNorm( resultDeviceVector2, 1 );
    
    std::string resultDifferenceAbsMax = "differenceAbsMax = " + std::to_string( differenceAbsMax );
    std::string resultDifferenceLpNorm = "differenceLpNorm = " + std::to_string( differenceLpNorm );
    
    char *absMax = &resultDifferenceAbsMax[ 0u ];
    char *lpNorm = &resultDifferenceLpNorm[ 0u ];
    
    // FIXME: THIS ISN'T AN ELEGANT SOLUTION, IT MAKES THE LOG FILE VERY LONG
//    benchmark.addErrorMessage( absMax, 1 );
//    benchmark.addErrorMessage( lpNorm, 1 );
    
    std::cout << std::endl;
    cusparseDestroy( cusparseHandle );
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
   
   // This doesn't have a titles (matrix format, rows, cols, etc.) in the output, because the header is the same as before (CSR).
   result |= benchmarkCusparseSpMV< Real, Matrices::CSR >( benchmark, inputFileName );
   
   result |= benchmarkSpMV< Real, Matrices::Ellpack >( benchmark, inputFileName );
   result |= benchmarkSpMV< Real, SlicedEllpack >( benchmark, inputFileName );
//   result |= benchmarkSpMV< Real, Matrices::ChunkedEllpack >( benchmark, inputFileName );
   return result;
}

} // namespace Benchmarks
} // namespace TNL
