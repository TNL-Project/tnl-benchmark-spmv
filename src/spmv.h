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

std::string getMatrixName( const String& InputFileName )
{
    std::string fileName = InputFileName;
    
    // Remove directory if present.
    // Do this before extension removal incase directory has a period character.
    // https://stackoverflow.com/questions/8520560/get-a-file-name-from-a-path
    // http://www.cplusplus.com/reference/string/string/find_last_of/
    const size_t last_slash_idx = fileName.find_last_of("/\\");
    if (std::string::npos != last_slash_idx)
    {
        fileName.erase(0, last_slash_idx + 1);
    }
    
    return fileName;
}

// Get only the name of the format from getType()
template< typename Matrix >
std::string getMatrixFormat( const Matrix& matrix )
{
    std::string mtrxFullType = matrix.getType();
    std::string mtrxType = mtrxFullType.substr( 0, mtrxFullType.find( "<" )) ;
    std::string format = mtrxType.substr( mtrxType.find( ':' ) + 2 );
    
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
    // Setup CSR for cuSPARSE
    typedef Matrices::CSR< Real, Devices::Host, int > CSR_HostMatrix;
    typedef Matrices::CSR< Real, Devices::Cuda, int > CSR_DeviceMatrix;
    
    CSR_HostMatrix CSRhostMatrix;
    CSR_DeviceMatrix CSRdeviceMatrix;
    
    // Read the matrix for CSR, to setup cuSPARSE
    try
      {         
         if( ! MatrixReader< CSR_HostMatrix >::readMtxFile( inputFileName, CSRhostMatrix ) )
         {
            benchmark.addErrorMessage( "Failed to read matrix!", 1 );            
            return false;
         }
      }
      catch( std::bad_alloc )
      {
         benchmark.addErrorMessage( "Failed to allocate memory for matrix!", 1 );
         return false;
      }
    
    // cuSPARSE handle setup
    cusparseHandle_t cusparseHandle;
    cusparseCreate( &cusparseHandle );
    
#ifdef HAVE_CUDA
    // FIXME: This doesn't work for ChunkedEllpack, because
    //        its cross-device assignment is not implemented yet
    CSRdeviceMatrix = CSRhostMatrix;
    
    // Delete the CSRhostMatrix, so it doesn't take up unnecessary space
    CSRhostMatrix.reset();
    
    TNL::CusparseCSR< Real > cusparseCSR;
    cusparseCSR.init( CSRdeviceMatrix, &cusparseHandle );
#endif
    
    // Other formats setup
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
            benchmark.addErrorMessage( "Failed to read matrix!", 1 );            
            return false;
         }
      }
      catch( std::bad_alloc )
      {
         benchmark.addErrorMessage( "Failed to allocate memory for matrix!", 1 );
         return false;
      }
    
#ifdef HAVE_CUDA
    // FIXME: This doesn't work for ChunkedEllpack, because
    //        its cross-device assignment is not implemented yet
    deviceMatrix = hostMatrix;
#endif

    benchmark.setMetadataColumns( Benchmark::MetadataColumns({
          { "matrix format", convertToString( getMatrixFormat( hostMatrix ) ) },
          { "matrix name", convertToString( getMatrixName( inputFileName ) ) },
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
    auto spmvCusparse = [&]() {
        cusparseCSR.vectorProduct( deviceVector, deviceVector2 );
    };

    benchmark.setOperation( datasetSize );
    benchmark.time< Devices::Host >( reset, "CPU", spmvHost );
    
    // Initialize the host vector to be compared. (The values in hostVector2 will be reset when spmvCuda starts)
    HostVector resultHostVector2;
    resultHostVector2.setSize( hostVector2.getSize() );
    resultHostVector2.setValue( 0.0 );
    
    // Copy the values
    resultHostVector2 = hostVector2;
    
 #ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU", spmvCuda );

    // Setup the device vector to be compared
    HostVector resultDeviceVector2;
    resultDeviceVector2.setSize( deviceVector2.getSize() );
    resultDeviceVector2.setValue( 0.0 );
    
    resultDeviceVector2 = deviceVector2;
#endif
    
    // FIXME: How to include benchmark with different name under the same header as the current format being benchmarked???
    // FIXME: Does it matter that speedup show difference only between current test and first test?
    //          Speedup shows difference between CPU and GPU-cuSPARSE, because in Benchmarks.h:
    //              * If there is no baseTime, the resulting test time is set to baseTime.
    //              * However, if there is a baseTime (from the CPU compared to GPU test),
    //                  baseTime isn't changed. If we change it in Benchmarks.h to compare 
    //                  the speedup from the last test, it will mess up BLAS benchmarks etc.
    benchmark.setMetadataColumns( Benchmark::MetadataColumns({
          { "matrix format", convertToString( "CSR-cuSPARSE" ) },
          { "matrix name", convertToString( getMatrixName( inputFileName ) ) },
          { "non-zeros", convertToString( hostMatrix.getNumberOfNonzeroMatrixElements() ) },
          { "rows", convertToString( hostMatrix.getRows() ) },
          { "columns", convertToString( hostMatrix.getColumns() ) }
       } ));
    
#ifdef HAVE_CUDA
    benchmark.time< Devices::Cuda >( reset, "GPU-Cusparse", spmvCusparse );
    
    HostVector resultcuSPARSEDeviceVector2;
    resultcuSPARSEDeviceVector2.setSize( deviceVector2.getSize() );
    resultcuSPARSEDeviceVector2.setValue( 0.0 );
    
    resultcuSPARSEDeviceVector2 = deviceVector2;
 #endif
    
#ifdef RESULTS
    // Difference between GPU (curent format) and GPU-cuSPARSE results
    Real cuSPARSEdifferenceAbsMax = resultDeviceVector2.differenceAbsMax( resultcuSPARSEDeviceVector2 );
    Real cuSPARSEdifferenceLpNorm = resultDeviceVector2.differenceLpNorm( resultcuSPARSEDeviceVector2, 1 );
    
    std::string GPUxGPUcuSPARSE_resultDifferenceAbsMax = "GPUxGPUcuSPARSE differenceAbsMax = " + std::to_string( cuSPARSEdifferenceAbsMax );
    std::string GPUxGPUcuSPARSE_resultDifferenceLpNorm = "GPUxGPUcuSPARSE differenceLpNorm = " + std::to_string( cuSPARSEdifferenceLpNorm );
    
    char *GPUcuSPARSE_absMax = &GPUxGPUcuSPARSE_resultDifferenceAbsMax[ 0u ];
    char *GPUcuSPARSE_lpNorm = &GPUxGPUcuSPARSE_resultDifferenceLpNorm[ 0u ];
    
    
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
    std::cout << GPUcuSPARSE_absMax << std::endl;
    std::cout << GPUcuSPARSE_lpNorm << std::endl;
    
    // FIXME: THIS ISN'T AN ELEGANT SOLUTION, IT MAKES THE LOG FILE VERY LONG
//    benchmark.addErrorMessage( absMax, 1 );
//    benchmark.addErrorMessage( lpNorm, 1 );
    
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
