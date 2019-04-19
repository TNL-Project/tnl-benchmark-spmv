/***************************************************************************
                          tnl-benchmark-spmv.h  -  description
                             -------------------
    begin                : Jun 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <iomanip>
#include <unistd.h>
#ifdef HAVE_CUDA
#include <cusparse.h>
#endif

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/AdEllpack.h>
#include <TNL/Matrices/BiEllpack.h>
#include <TNL/Matrices/BiEllpackSymmetric.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/EllpackSymmetric.h>
#include <TNL/Matrices/EllpackSymmetricGraph.h>
#include <TNL/Matrices/SlicedEllpack.h>
#include <TNL/Matrices/SlicedEllpackSymmetric.h>
#include <TNL/Matrices/SlicedEllpackSymmetricGraph.h>
#include <TNL/Matrices/ChunkedEllpack.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Timer.h>
#include "tnlCusparseCSRMatrix.h"

using namespace std;
using namespace TNL;
using namespace TNL::Matrices;

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter                            ( "General settings:" );
   config.addRequiredEntry< String >( "test" , "Test to be performed." );
      config.addEntryEnum< String >( "mtx" );
      config.addEntryEnum< String >( "tnl" );
   config.addRequiredEntry< String >( "input-file" , "Input file name." );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-spmv.log");
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntry< double >( "stop-time", "Seconds to iterate the SpMV operation.", 1.0 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
}

bool initLogFile( std::fstream& logFile, const String& fileName )
{
   if( access( fileName.getString(), F_OK ) == -1 )
   {
      logFile.open( fileName.getString(), std::ios::out );
      if( ! logFile )
         return false;
      const String fillingColoring = " : COLORING 0 #FFF8DC 20 #FFFF00 40 #FFD700 60 #FF8C0 80 #FF0000 100";
      const String speedupColoring = " : COLORING #0099FF 1 #FFFFFF 2 #00FF99 4 #33FF99 8 #33FF22 16 #FF9900";
      const String paddingColoring = " : COLORING #FFFFFF 1 #FFFFCC 10 #FFFF99 100 #FFFF66 1000 #FFFF33 10000 #FFFF00";
      logFile << "#Matrix file " << std::endl;
      logFile << "#Rows" << std::endl;
      logFile << "#Columns" << std::endl;
      logFile << "#Non-zero elements" << std::endl;
      logFile << "#Filling (in %)" << fillingColoring << std::endl;
      logFile << "#CSR Format" << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << std::endl;
#ifdef HAVE_CUDA
      logFile << "# Cusparse CSR" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - cusparse-csr-speedup.txt" << std::endl;
      logFile << "# CUDA" << std::endl;
      logFile << "#  Scalar" << std::endl;
      logFile << "#   Gflops" << std::endl;
      logFile << "#   Throughput" << std::endl;
      logFile << "#   Speedup" << speedupColoring << " SORT - csr-scalar-cuda-speedup.txt" << std::endl;
      logFile << "#  Vector" << std::endl;
      logFile << "#   Warp Size 1" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-1-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 2" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-2-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 4" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-4-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 8" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-8-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 16" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-16-cuda-speedup.txt" << std::endl;
      logFile << "#   Warp Size 32" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-vector-32-cuda-speedup.txt" << std::endl;
      logFile << "#  Hybrid" << std::endl;
      logFile << "#   Split 2" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-2-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 4" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-4-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 8" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-8-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 16" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-16-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 32" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-32-cuda-speedup.txt" << std::endl;
      logFile << "#   Split 64" << std::endl;
      logFile << "#    Gflops" << std::endl;
      logFile << "#    Throughput" << std::endl;
      logFile << "#    Speedup" << speedupColoring << " SORT - csr-hybrid-64-cuda-speedup.txt" << std::endl;
#endif
      logFile << "#Ellpack Format" << std::endl;
      logFile << "# Padding (in %)" << paddingColoring << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - ellpack-host-speedup.txt" << std::endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - ellpack-cuda-speedup.txt" << std::endl;
#endif
      logFile << "#SlicedEllpack Format" << std::endl;
      logFile << "# Padding (in %)" << paddingColoring << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - sliced-ellpack-host-speedup.txt" << std::endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - sliced-ellpack-cuda-speedup.txt" << std::endl;
#endif
      logFile << "#ChunkedEllpack Format" << std::endl;
      logFile << "# Padding (in %)" << paddingColoring << std::endl;
      logFile << "# CPU" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - chunked-ellpack-host-speedup.txt" << std::endl;
#ifdef HAVE_CUDA
      logFile << "# CUDA" << std::endl;
      logFile << "#  Gflops" << std::endl;
      logFile << "#  Throughput" << std::endl;
      logFile << "#  Speedup" << speedupColoring << " SORT - chunked-ellpack-cuda-speedup.txt" << std::endl;
#endif
      return true;
   }
   logFile.open( fileName.getString(), std::ios::out | std::ios::app );
   //logFile << std::setprecision( 2 );
   if( ! logFile )
      return false;
   return true;
}

template< typename Matrix >
void printMatrixInfo( const String& inputFileName,
                      const Matrix& matrix,
                      std::ostream& str )
{
   str << " Rows: " << std::setw( 8 ) << matrix.getRows();
   str << " Columns: " << std::setw( 8 ) << matrix.getColumns();
   str << " Nonzero Elements: " << std::setw( 10 ) << matrix.getNumberOfNonzeroMatrixElements();
   const double fillingRatio = ( double ) matrix.getNumberOfNonzeroMatrixElements() / ( double ) matrix.getNumberOfMatrixElements();
   str << " Filling: " << std::setw( 5 ) << 100.0 * fillingRatio << "%" << std::endl;
   str << std::setw( 25 ) << "Format"
       << std::setw( 15 ) << "Padding"
       << std::setw( 15 ) << "Time"
       << std::setw( 15 ) << "GFLOPS"
       << std::setw( 15 ) << "Throughput"
       << std::setw( 15 ) << "Speedup" << std::endl;
}

template< typename Matrix >
bool writeMatrixInfo( const String& inputFileName,
                      const Matrix& matrix,
                      std::ostream& logFile )
{
   logFile << std::endl;
   logFile << inputFileName << std::endl;
   logFile << " " << matrix.getRows() << std::endl;
   logFile << " " << matrix.getColumns() << std::endl;
   logFile << " " << matrix.getNumberOfNonzeroMatrixElements() << std::endl;
   const double fillingRatio = ( double ) matrix.getNumberOfNonzeroMatrixElements() / ( double ) matrix.getNumberOfMatrixElements();
   logFile << " " << 100.0 * fillingRatio << std::endl;
   logFile << std::flush;
   if( ! logFile.good() )
      return false;
   return true;
}

double computeGflops( const long int nonzeroElements,
                      const int iterations,
                      const double& time )
{
   return ( double ) ( 2 * iterations * nonzeroElements ) / time * 1.0e-9;
}

template< typename Real >
double computeThroughput( const long int nonzeroElements,
                          const int iterations,
                          const int rows,
                          const double& time )
{
   return ( double ) ( ( 2 * nonzeroElements + rows ) * iterations ) * sizeof( Real ) / time * 1.0e-9;
}

template< typename Matrix,
          typename Vector >
double benchmarkMatrix( const Matrix& matrix,
                        const Vector& x,
                        Vector& b,
                        const long int nonzeroElements,
                        const char* format,
                        const double& stopTime,
                        const double& baseline,
                        int verbose,
                        std::fstream& logFile )
{
   Timer timer;
   timer.start();
   double time( 0.0 );
   int iterations( 0 );
   while( time < stopTime )
   {
      matrix.vectorProduct( x, b );
#ifdef HAVE_CUDA
      if( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value )
         cudaDeviceSynchronize();
#endif
      time = timer.getRealTime();
      iterations++;
   }
   const double gflops = computeGflops( nonzeroElements, iterations, time );
   const double throughput = computeThroughput< typename Matrix::RealType >( nonzeroElements, iterations, matrix.getRows(), time );
   const long int allocatedElements = matrix.getNumberOfMatrixElements();
   const double padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
   if( verbose )
   {
     std::cout << std::setw( 25 ) << format
           << std::setw( 15 ) << padding
           << std::setw( 15 ) << time
           << std::setw( 15 ) << gflops
           << std::setw( 15 ) << throughput;
      if( baseline )
        std::cout << std::setw( 15 ) << gflops / baseline << std::endl;
      else
        std::cout << std::setw( 15 ) << "N/A" << std::endl;
   }
   logFile << "  " << gflops << std::endl;
   logFile << "  " << throughput << std::endl;
   if( baseline )
      logFile << gflops / baseline << std::endl;
   else
      logFile << "N/A" << std::endl;
   return gflops;
}

void writeTestFailed( std::fstream& logFile,
                      int repeat )
{
   for( int i = 0; i < repeat; i++ )
      logFile << "N/A" << std::endl;
}

template< typename Real >
bool setupBenchmark( const Config::ParameterContainer& parameters )
{
   const String& test = parameters.getParameter< String >( "test" );
   const String& inputFileName = parameters.getParameter< String >( "input-file" );
   const String& logFileName = parameters.getParameter< String >( "log-file" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   const double stopTime = parameters.getParameter< double >( "stop-time" );
   std::fstream logFile;
   if( ! initLogFile( logFile, logFileName ) )
   {
      std::cerr << "I am not able to open the file " << logFileName << "." << std::endl;
      return false;
   }
   if( test == "mtx" )
   {
      typedef Matrices::CSR< Real, Devices::Host, int > CSRType;
      CSRType csrMatrix;
      try
      {
         if( ! MatrixReader< CSRType >::readMtxFile( inputFileName, csrMatrix ) )
         {
            std::cerr << "I am not able to read the matrix file " << inputFileName << "." << std::endl;
            logFile << std::endl;
            logFile << inputFileName << std::endl;
            logFile << "Benchmark failed: Unable to read the matrix." << std::endl;
            return false;
         }
      }
      catch( const std::bad_alloc& )
      {
         std::cerr << "Not enough memory to read the matrix." << std::endl;
         logFile << std::endl;
         logFile << inputFileName << std::endl;
         logFile << "Benchmark failed: Not enough memory." << std::endl;
         return false;
      }
      if( verbose )
         printMatrixInfo( inputFileName, csrMatrix,std::cout );
      if( ! writeMatrixInfo( inputFileName, csrMatrix, logFile ) )
      {
         std::cerr << "I am not able to write new matrix to the log file." << std::endl;
         return false;
      }
      const int rows = csrMatrix.getRows();
      const long int nonzeroElements = csrMatrix.getNumberOfMatrixElements();
      Containers::Vector< int, Devices::Host, int > rowLengthsHost;
      rowLengthsHost.setSize( rows );
      for( int row = 0; row < rows; row++ )
         rowLengthsHost[ row ] = csrMatrix.getRowLength( row );

      typedef Containers::Vector< Real, Devices::Host, int > HostVector;
      HostVector hostX, hostB;
      hostX.setSize( csrMatrix.getColumns() );
      hostX.setValue( 1.0 );
      hostB.setSize( csrMatrix.getRows() );
#ifdef HAVE_CUDA
      typedef Containers::Vector< Real, Devices::Cuda, int > CudaVector;
      CudaVector cudaX, cudaB;
      Containers::Vector< int, Devices::Cuda, int > rowLengthsCuda;
      cudaX.setSize( csrMatrix.getColumns() );
      cudaX.setValue( 1.0 );
      cudaB.setSize( csrMatrix.getRows() );
      rowLengthsCuda.setSize( csrMatrix.getRows() );
      rowLengthsCuda = rowLengthsHost;
      cusparseHandle_t cusparseHandle;
      cusparseCreate( &cusparseHandle );
#endif
      const double baseline = benchmarkMatrix( csrMatrix,
                                               hostX,
                                               hostB,
                                               nonzeroElements,
                                               "CSR Host",
                                               stopTime,
                                               0.0,
                                               verbose,
                                               logFile );
#ifdef HAVE_CUDA
      typedef CSR< Real, Devices::Cuda, int > CSRCudaType;
      CSRCudaType cudaCSR;
      //cout << "Copying matrix to GPU... ";
      cudaCSR = csrMatrix;
      TNL::CusparseCSR< Real > cusparseCSR;
      cusparseCSR.init( cudaCSR, &cusparseHandle );
      benchmarkMatrix( cusparseCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "Cusparse CSR",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cusparseDestroy( cusparseHandle );

      std::cout << " done.   \r";
      /*cudaCSR.setCudaKernelType( CSRCudaType::scalar );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Scalar",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaKernelType( CSRCudaType::vector );
      cudaCSR.setCudaWarpSize( 1 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Vector 1",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaWarpSize( 2 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Vector 2",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaWarpSize( 4 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Vector 4",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaWarpSize( 8 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Vector 8",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaWarpSize( 16 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Vector 16",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaWarpSize( 32 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Vector 32",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setCudaKernelType( CSRCudaType::hybrid );
      cudaCSR.setHybridModeSplit( 2 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Hyrbid 2",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setHybridModeSplit( 4 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Hyrbid 4",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setHybridModeSplit( 8 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Hyrbid 8",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setHybridModeSplit( 16 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Hyrbid 16",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setHybridModeSplit( 32 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Hyrbid 32",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaCSR.setHybridModeSplit( 64 );
      benchmarkMatrix( cudaCSR,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "CSR Cuda Hyrbid 64",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );*/
      cudaCSR.reset();
#endif

      long int allocatedElements;
      double padding;
      typedef Ellpack< Real, Devices::Host, int > EllpackType;
      EllpackType ellpackMatrix;
      Matrices::copySparseMatrix( ellpackMatrix, csrMatrix );
      allocatedElements = ellpackMatrix.getNumberOfMatrixElements();
      padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
      logFile << "    " << padding << std::endl;
      benchmarkMatrix( ellpackMatrix,
                       hostX,
                       hostB,
                       nonzeroElements,
                       "Ellpack Host",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
#ifdef HAVE_CUDA
      typedef Ellpack< Real, Devices::Cuda, int > EllpackCudaType;
      EllpackCudaType cudaEllpack;
      std::cout << "Copying matrix to GPU... ";
      cudaEllpack = ellpackMatrix;
      std::cout << " done.   \r";
      benchmarkMatrix( cudaEllpack,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "Ellpack Cuda",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaEllpack.reset();
#endif
      ellpackMatrix.reset();

      typedef Matrices::EllpackSymmetric< Real, Devices::Host, int > EllpackSymmetricType;
      EllpackSymmetricType EllpackSymmetric;
      if( ! MatrixReader< EllpackSymmetricType >::readMtxFile( inputFileName, EllpackSymmetric, verbose, true ) )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = EllpackSymmetric.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding <<std::endl;
         benchmarkMatrix( EllpackSymmetric,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "EllpackSym Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         EllpackSymmetric.reset();
#ifdef HAVE_CUDA
         typedef Matrices::EllpackSymmetric< Real, Devices::Cuda, int > EllpackSymmetricCudaType;
         EllpackSymmetricCudaType cudaEllpackSymmetric;
        std::cout << "Copying matrix to GPU... ";
         for( int i = 0; i < rowLengthsHost.getSize(); i++ )
             rowLengthsHost[ i ] = EllpackSymmetric.getRowLength( i );
         rowLengthsCuda = rowLengthsHost;

         // TODO: fix this
         //if( ! cudaEllpackSymmetric.copyFrom( EllpackSymmetric, rowLengthsCuda ) )
         {
           std::cerr << "I am not able to transfer the matrix on GPU." <<std::endl;
            writeTestFailed( logFile, 3 );
         }
         //else
         {
           std::cout << " done.   \r";
            benchmarkMatrix( cudaEllpackSymmetric,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "EllpackSym Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaEllpackSymmetric.reset();
#endif
      }

      typedef Matrices::SlicedEllpack< Real, Devices::Host, int > SlicedEllpackMatrixType;
      SlicedEllpackMatrixType slicedEllpackMatrix;
      if( ! Matrices::MatrixReader< SlicedEllpackMatrixType >::readMtxFile( inputFileName, slicedEllpackMatrix, verbose ) )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = slicedEllpackMatrix.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100;
         logFile << "    " << padding <<std::endl;
         benchmarkMatrix( slicedEllpackMatrix,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "SlicedEllpack Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
#ifdef HAVE_CUDA
         typedef Matrices::SlicedEllpack< Real, Devices::Cuda, int > SlicedEllpackMatrixCudaType;
         SlicedEllpackMatrixCudaType cudaSlicedEllpackMatrix;
         for( int i = 0; i < rowLengthsHost.getSize(); i++ )
              rowLengthsHost[ i ] = slicedEllpackMatrix.getRowLength( i );
         rowLengthsCuda = rowLengthsHost;
         // TODO: fix
         //if( ! cudaSlicedEllpackMatrix.copyFrom( slicedEllpackMatrix, rowLengthsCuda ) )
         {
            std::cerr << "Nejde zkopirovat" <<std::endl;
             writeTestFailed( logFile, 3 );
         }
         //else
         {
           std::cout << " done.    \r";
            benchmarkMatrix( cudaSlicedEllpackMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "SlicedEllpack Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaSlicedEllpackMatrix.reset();        
#endif         
      }

      typedef Matrices::ChunkedEllpack< Real, Devices::Host, int > ChunkedEllpackType;
      ChunkedEllpackType chunkedEllpack;
      Matrices::copySparseMatrix( chunkedEllpack, csrMatrix );
      allocatedElements = chunkedEllpack.getNumberOfMatrixElements();
      padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
      logFile << "    " << padding << std::endl;
      benchmarkMatrix( chunkedEllpack,
                       hostX,
                       hostB,
                       nonzeroElements,
                       "ChunkedEllpack Host",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
         
#ifdef HAVE_CUDA
      typedef Matrices::ChunkedEllpack< Real, Devices::Cuda, int > ChunkedEllpackCudaType;
      ChunkedEllpackCudaType cudaChunkedEllpack;
      std::cout << "Copying matrix to GPU... ";
      cudaChunkedEllpack = chunkedEllpack;
      std::cout << " done.    \r";
      benchmarkMatrix( cudaChunkedEllpack,
                       cudaX,
                       cudaB,
                       nonzeroElements,
                       "ChunkedEllpack Cuda",
                       stopTime,
                       baseline,
                       verbose,
                       logFile );
      cudaChunkedEllpack.reset();
#endif

      typedef Matrices::BiEllpack< Real, Devices::Host, int > BiEllpackMatrixType;
      BiEllpackMatrixType biEllpackMatrix;
      // TODO: I did not check this during git merging, but I hope its gonna work
      //   Tomas Oberhuber
      //    copySparseMatrix( biEllpackMatrix, csrMatrix ); // TODO:Fix the getRow method to be compatible with othr formats
      /*if( ! biEllpackMatrix.copyFrom( csrMatrix, rowLengthsHost ) )
         writeTestFailed( logFile, 7 );
      else*/
      {
         allocatedElements = biEllpackMatrix.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding <<std::endl;
         benchmarkMatrix( biEllpackMatrix,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "BiEllpack Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         biEllpackMatrix.reset();

#ifdef HAVE_CUDA
         typedef Matrices::BiEllpack< Real, Devices::Cuda, int > BiEllpackMatrixCudaType;
         BiEllpackMatrixCudaType cudaBiEllpackMatrix;
         // TODO: I did not check this during git merging, but I hope its gonna work
         //   Tomas Oberhuber
         //    copySparseMatrix( biEllpackMatrix, csrMatrix ); // TODO:Fix the getRow method to be compatible with othr formats
        std::cout << "Copying matrix to GPU... ";
         /*if( ! cudaBiEllpackMatrix.copyFrom( biEllpackMatrix, rowLengthsCuda ) )
         {
           std::cerr << "I am not able to transfer the matrix on GPU." <<std::endl;
            writeTestFailed( logFile, 3 );
         }
         else*/
         {
           std::cout << " done.    \r";
            benchmarkMatrix( cudaBiEllpackMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "BiEllpack Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaBiEllpackMatrix.reset();
#endif
      }

      typedef Matrices::SlicedEllpackSymmetric< Real, Devices::Host, int > SlicedEllpackSymmetricType;
      SlicedEllpackSymmetricType slicedEllpackSymmetric;
      if( ! Matrices::MatrixReader< SlicedEllpackSymmetricType >::readMtxFile( inputFileName, slicedEllpackSymmetric, verbose, true ) )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = slicedEllpackSymmetric.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding <<std::endl;
         benchmarkMatrix( slicedEllpackSymmetric,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "SlicedEllpackSym Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         slicedEllpackSymmetric.reset();
#ifdef HAVE_CUDA
         typedef Matrices::SlicedEllpackSymmetric< Real, Devices::Cuda, int > SlicedEllpackSymmetricCudaType;
         SlicedEllpackSymmetricCudaType cudaSlicedEllpackSymmetric;
        std::cout << "Copying matrix to GPU... ";
         for( int i = 0; i < rowLengthsHost.getSize(); i++ )
             rowLengthsHost[ i ] = slicedEllpackSymmetric.getRowLength( i );
         rowLengthsCuda = rowLengthsHost;
         // TODO: fiox the nest line
         //if( ! cudaSlicedEllpackSymmetric.copyFrom( slicedEllpackSymmetric, rowLengthsCuda ) )
         {
           std::cerr << "I am not able to transfer the matrix on GPU." <<std::endl;
            writeTestFailed( logFile, 3 );
         }
         //else
         {
           std::cout << " done.   \r";
            benchmarkMatrix( cudaSlicedEllpackSymmetric,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "SlicedEllpackSym Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaSlicedEllpackSymmetric.reset();
#endif
      }

      typedef Matrices::EllpackSymmetricGraph< Real, Devices::Host, int > EllpackSymmetricGraphMatrixType;
      EllpackSymmetricGraphMatrixType EllpackSymmetricGraphMatrix;
      if( ! Matrices::MatrixReader< EllpackSymmetricGraphMatrixType >::readMtxFile( inputFileName, EllpackSymmetricGraphMatrix, verbose, true ) ||
          ! EllpackSymmetricGraphMatrix.help() )
         writeTestFailed( logFile, 7 );
      else
      {
         allocatedElements = EllpackSymmetricGraphMatrix.getNumberOfMatrixElements();
         padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
         logFile << "    " << padding <<std::endl;
         benchmarkMatrix( EllpackSymmetricGraphMatrix,
                          hostX,
                          hostB,
                          nonzeroElements,
                          "Ellpack Graph Host",
                          stopTime,
                          baseline,
                          verbose,
                          logFile );
         EllpackSymmetricGraphMatrix.reset();
#ifdef HAVE_CUDA
         typedef Matrices::EllpackSymmetricGraph< Real, Devices::Cuda, int > EllpackSymmetricGraphMatrixCudaType;
         EllpackSymmetricGraphMatrixCudaType cudaEllpackSymmetricGraphMatrix;
        std::cout << "Copying matrix to GPU... ";
         for( int i = 0; i < rowLengthsHost.getSize(); i++ )
             rowLengthsHost[ i ] = EllpackSymmetricGraphMatrix.getRowLength( i );
         rowLengthsCuda = rowLengthsHost;
         // TODO: fix it
         //if( ! cudaEllpackSymmetricGraphMatrix.copyFrom( EllpackSymmetricGraphMatrix, rowLengthsCuda ) ) 
         {
            writeTestFailed( logFile, 3 );
         }
         //else if( ! cudaEllpackSymmetricGraphMatrix.help() )
         {
            writeTestFailed( logFile, 3 );
         } 
         //else
         {
            std::cout << " done.   \r";
            benchmarkMatrix( cudaEllpackSymmetricGraphMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "Ellpack Graph Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
         }
         cudaEllpackSymmetricGraphMatrix.reset();
#endif
      }

      
        typedef Matrices::AdEllpack< Real, Devices::Host, int > AdEllpackMatrixType;
        AdEllpackMatrixType adEllpackMatrix;
         // TODO: I did not check this during git merging, but I hope its gonna work
         //   Tomas Oberhuber
        //copySparseMatrix( adEllpackMatrix, csrMatrix ); // TODO:Fix the getRow method to be compatible with othr formats
        /*if( ! adEllpackMatrix.copyFrom( csrMatrix, rowLengthsHost ) )
           writeTestFailed( logFile, 7 );
        else*/
        {
           allocatedElements = adEllpackMatrix.getNumberOfMatrixElements();
           padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
           logFile << "    " << padding <<std::endl;
           benchmarkMatrix( adEllpackMatrix,
                            hostX,
                            hostB,
                            nonzeroElements,
                            "AdEllpack Host",
                            stopTime,
                            baseline,
                            verbose,
                            logFile );
           adEllpackMatrix.reset();
        }
      
#ifdef HAVE_CUDA
         typedef Matrices::AdEllpack< Real, Devices::Cuda, int > AdEllpackMatrixCudaType;
         AdEllpackMatrixCudaType cudaAdEllpackMatrix;
         // TODO: I did not check this during git merging, but I hope its gonna work
         //   Tomas Oberhuber
        //copySparseMatrix( adEllpackMatrix, csrMatrix ); // TODO:Fix the getRow method to be compatible with othr formats
        std::cout << "Copying matrix to GPU... ";
         /*if( ! cudaAdEllpackMatrix.copyFrom( csrMatrix, rowLengthsCuda ) )
         {
           std::cerr << "I am not able to transfer the matrix on GPU." <<std::endl;
            writeTestFailed( logFile, 3 );
         }
         else*/
         {
	    allocatedElements = cudaAdEllpackMatrix.getNumberOfMatrixElements();
	    padding = ( double ) allocatedElements / ( double ) nonzeroElements * 100.0 - 100.0;
            logFile << "    " << padding <<std::endl;
           std::cout << " done.    \r";
            benchmarkMatrix( cudaAdEllpackMatrix,
                             cudaX,
                             cudaB,
                             nonzeroElements,
                             "AdEllpack Cuda",
                             stopTime,
                             baseline,
                             verbose,
                             logFile );
           cudaAdEllpackMatrix.reset();
	}
#endif
   }
   return true;
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );
 
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc.printUsage( argv[ 0 ] );
      return 1;
   }
   const String& precision = parameters.getParameter< String >( "precision" );
   if( precision == "float" )
      if( ! setupBenchmark< float >( parameters ) )
         return EXIT_FAILURE;
   if( precision == "double" )
      if( ! setupBenchmark< double >( parameters ) )
         return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
