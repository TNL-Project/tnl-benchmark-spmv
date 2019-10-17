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

#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>
#include <TNL/Matrices/ChunkedEllpack.h>
#include <TNL/Matrices/AdEllpack.h>
#include <TNL/Matrices/BiEllpack.h>

#include <TNL/Matrices/MatrixReader.h>
using namespace TNL::Matrices;

#include <TNL/Exceptions/HostBadAlloc.h>

#include "cusparseCSRMatrix.h"

namespace TNL {
namespace Benchmarks {

// silly alias to match the number of template parameters with other formats
template< typename Real, typename Device, typename Index >
using SlicedEllpack = Matrices::SlicedEllpack< Real, Device, Index >;

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
    std::string mtrxFullType = matrix.getType();
    std::string mtrxType = mtrxFullType.substr( 0, mtrxFullType.find( "<" ) );
    std::string format = mtrxType.substr( mtrxType.find( ':' ) + 2 );
    
    return format;
}

// This function is not used currently (as of 17.03.19),
//  as the log takes care of printing and saving this information into the log file.
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
          template< typename, typename, typename > class Vector = Containers::Vector >
bool
benchmarkSpMV( Benchmark & benchmark,
               const String & inputFileName,
               bool verboseMR )
{
    // Setup CSR for cuSPARSE. It will compared to the format given as a template parameter to this function
    typedef Matrices::CSR< Real, Devices::Host, int > CSR_HostMatrix;
    typedef Matrices::CSR< Real, Devices::Cuda, int > CSR_DeviceMatrix;
    
    CSR_HostMatrix CSRhostMatrix;
    CSR_DeviceMatrix CSRdeviceMatrix;
    
    // Read the matrix for CSR, to set up cuSPARSE
    try
      {         
         if( ! MatrixReader< CSR_HostMatrix >::readMtxFile( inputFileName, CSRhostMatrix, verboseMR ) )
         { 
             throw Exceptions::HostBadAlloc();
             return false;
         }
      }
      catch( Exceptions::HostBadAlloc e )
      {
          e.what();
          return false;
      }
    
    // cuSPARSE handle setup
    cusparseHandle_t cusparseHandle;
    cusparseCreate( &cusparseHandle );
    
#ifdef HAVE_CUDA
    // cuSPARSE (in TNL's CSR) only works for device, copy the matrix from host to device
    CSRdeviceMatrix = CSRhostMatrix;
    
    // Delete the CSRhostMatrix, so it doesn't take up unnecessary space
    CSRhostMatrix.reset();
    
    // Initialize the cusparseCSR matrix.
    TNL::CusparseCSR< Real > cusparseCSR;
    cusparseCSR.init( CSRdeviceMatrix, &cusparseHandle );
#endif
    
    // Setup the format which is given as a template parameter to this function
    typedef Matrix< Real, Devices::Host, int > HostMatrix;
    typedef Matrix< Real, Devices::Cuda, int > DeviceMatrix;
    typedef Containers::Vector< Real, Devices::Host, int > HostVector;
    typedef Containers::Vector< Real, Devices::Cuda, int > CudaVector;
    
    HostMatrix hostMatrix;
    DeviceMatrix deviceMatrix;
    HostVector hostVector, hostVector2;
    CudaVector deviceVector, deviceVector2;
    
    // Load the format
    try
      {         
         if( ! MatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix, verboseMR ) )
         {
             throw Exceptions::HostBadAlloc();
             return false;
         }
      }
      catch( Exceptions::HostBadAlloc e )
      {
          e.what();
          return false;
      }
    
    hostMatrix.print( std::cout );
    std::cout << "\n\n\n\n===============VALUES:\n\n" << std::endl;
    
    hostMatrix.printValues();
    
#ifdef COMMENT
#ifdef HAVE_CUDA
    // FIXME: This doesn't work for Ad/BiEllpack, because
    //        their cross-device assignment is not implemented yet
    deviceMatrix = hostMatrix;
#endif

    // Setup MetaData here (not in tnl-benchmark-spmv.h, as done in Benchmarks/BLAS),
    //  because we need the matrix loaded first to get the rows and columns
    benchmark.setMetadataColumns( Benchmark::MetadataColumns({
          { "matrix name", convertToString( getMatrixFileName( inputFileName ) ) },
          { "non-zeros", convertToString( hostMatrix.getNumberOfNonzeroMatrixElements() ) },
          { "rows", convertToString( hostMatrix.getRows() ) },
          { "columns", convertToString( hostMatrix.getColumns() ) },
          { "matrix format", convertToString( getMatrixFormat( hostMatrix ) ) }
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
    auto spmvCusparse = [&]() {
        cusparseCSR.vectorProduct( deviceVector, deviceVector2 );
    };

    benchmark.setOperation( datasetSize );
    benchmark.time< Devices::Host >( reset, "CPU", spmvHost );
    
    // Initialize the host vector to be compared.
    //  (The values in hostVector2 will be reset when spmvCuda starts)
    HostVector resultHostVector2;
    resultHostVector2.setSize( hostVector2.getSize() );
    resultHostVector2.setValue( 0.0 );
    
    // Copy the values
    resultHostVector2 = hostVector2;
    
#ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU", spmvCuda );

    // Initialize the device vector to be compared.
    //  (The values in deviceVector2 will be reset when spmvCusparse starts)
    HostVector resultDeviceVector2;
    resultDeviceVector2.setSize( deviceVector2.getSize() );
    resultDeviceVector2.setValue( 0.0 );
    
    resultDeviceVector2 = deviceVector2;
#endif
    
    // Setup cuSPARSE MetaData, since it has the same header as CSR, 
    //  and therefore will not get its own headers (rows, cols, speedup etc.) in log.
    //      * Not setting this up causes (among other undiscovered errors) the speedup from CPU to GPU on the input format to be overwritten.
    
    // FIXME: How to include benchmark with different name under the same header as the current format being benchmarked???
    // FIXME: Does it matter that speedup show difference only between current test and first test?
    //          Speedup shows difference between CPU and GPU-cuSPARSE, because in Benchmarks.h:
    //              * If there is no baseTime, the resulting test time is set to baseTime.
    //              * However, if there is a baseTime (from the CPU compared to GPU test),
    //                  baseTime isn't changed. If we change it in Benchmarks.h to compare 
    //                  the speedup from the last test, it will mess up BLAS benchmarks etc.
    benchmark.setMetadataColumns( Benchmark::MetadataColumns({
          { "matrix name", convertToString( getMatrixFileName( inputFileName ) ) },
          { "non-zeros", convertToString( hostMatrix.getNumberOfNonzeroMatrixElements() ) },
          { "rows", convertToString( hostMatrix.getRows() ) },
          { "columns", convertToString( hostMatrix.getColumns() ) },
          { "matrix format", convertToString( "CSR-cuSPARSE" ) }
       } ));
   
#ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU", spmvCusparse );
    
    HostVector resultcuSPARSEDeviceVector2;
    resultcuSPARSEDeviceVector2.setSize( deviceVector2.getSize() );
    resultcuSPARSEDeviceVector2.setValue( 0.0 );
    
    resultcuSPARSEDeviceVector2 = deviceVector2;
 #endif
    
//#ifdef COMPARE_RESULTS
    // Difference between GPU (curent format) and GPU-cuSPARSE results
    Real cuSparseDifferenceAbsMax = resultDeviceVector2.differenceAbsMax( resultcuSPARSEDeviceVector2 );
    Real cuSparseDifferenceLpNorm = resultDeviceVector2.differenceLpNorm( resultcuSPARSEDeviceVector2, 1 );
    
    std::string GPUxGPUcuSparse_resultDifferenceAbsMax = "GPUxGPUcuSPARSE differenceAbsMax = " + std::to_string( cuSparseDifferenceAbsMax );
    std::string GPUxGPUcuSparse_resultDifferenceLpNorm = "GPUxGPUcuSPARSE differenceLpNorm = " + std::to_string( cuSparseDifferenceLpNorm );
    
    char *GPUcuSparse_absMax = &GPUxGPUcuSparse_resultDifferenceAbsMax[ 0u ];
    char *GPUcuSparse_lpNorm = &GPUxGPUcuSparse_resultDifferenceLpNorm[ 0u ];
    
    
    // Difference between CPU and GPU results for the current format
    Real differenceAbsMax = resultHostVector2.differenceAbsMax( resultDeviceVector2 );
    Real differenceLpNorm = resultHostVector2.differenceLpNorm( resultDeviceVector2, 1 );
    
    std::string CPUxGPU_resultDifferenceAbsMax = "CPUxGPU differenceAbsMax = " + std::to_string( differenceAbsMax );
    std::string CPUxGPU_resultDifferenceLpNorm = "CPUxGPU differenceLpNorm = " + std::to_string( differenceLpNorm );
    
    char *CPUxGPU_absMax = &CPUxGPU_resultDifferenceAbsMax[ 0u ];
    char *CPUxGPU_lpNorm = &CPUxGPU_resultDifferenceLpNorm[ 0u ];
    
    // Print result differences of CPU and GPU of current format
    std::cout << CPUxGPU_absMax << std::endl;
    std::cout << CPUxGPU_lpNorm << std::endl;
    
    // Print result differences of GPU of current format and GPU with cuSPARSE.
    std::cout << GPUcuSparse_absMax << std::endl;
    std::cout << GPUcuSparse_lpNorm << std::endl;
    
    // FIXME: This isn't an elegant solution, it makes the log file very long.
//    benchmark.addErrorMessage( GPUcuSparse_absMax, 1 );
//    benchmark.addErrorMessage( GPUcuSparse_lpNorm, 1 );
    
//    benchmark.addErrorMessage( CPUxGPU_absMax, 1 );
//    benchmark.addErrorMessage( CPUxGPU_lpNorm, 1 );
    
//#endif
    
#endif
    std::cout << std::endl;
    return true;
}

template< typename Real = double,
          typename Index = int >
bool
benchmarkSpmvSynthetic( Benchmark & benchmark,
                        const String& inputFileName,
                        bool verboseMR )
{
   bool result = true;
   // TODO: benchmark all formats from tnl-benchmark-spmv (different parameters of the base formats)
//   result |= benchmarkSpMV< Real, Matrices::CSR >( benchmark, inputFileName, verboseMR );   
//   result |= benchmarkSpMV< Real, Matrices::Ellpack >( benchmark, inputFileName, verboseMR );
//   result |= benchmarkSpMV< Real, SlicedEllpack >( benchmark, inputFileName, verboseMR );
//   result |= benchmarkSpMV< Real, Matrices::ChunkedEllpack >( benchmark, inputFileName, verboseMR );
   
   // AdEllpack/BiEllpack doesn't have cross-device assignment ('= operator') implemented yet
//   result |= benchmarkSpMV< Real, Matrices::AdEllpack >( benchmark, inputFileName, verboseMR );
   result |= benchmarkSpMV< Real, Matrices::BiEllpack >( benchmark, inputFileName, verboseMR );
   return result;
}

} // namespace Benchmarks
} // namespace TNL
