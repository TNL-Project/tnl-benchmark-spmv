/***************************************************************************
                          tnl-benchmark-spmv.h  -  description
                             -------------------
    begin                : March 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

#include <Benchmarks/BLAS/array-operations.h>
#include <Benchmarks/BLAS/vector-operations.h>
#include "spmv.h"

#include <TNL/Matrices/MatrixReader.h>
using namespace TNL::Matrices;

using namespace TNL;
using namespace TNL::Benchmarks;

//template< typename Matrix >
//void printMatrixInfo( const String& inputFileName,
//                      const Matrix& matrix,
//                      std::ostream& str )
//{
//   str << " Rows: " << std::setw( 8 ) << matrix.getRows();
//   str << " Columns: " << std::setw( 8 ) << matrix.getColumns();
//   str << " Nonzero Elements: " << std::setw( 10 ) << matrix.getNumberOfNonzeroMatrixElements();
//}

template< typename Real >
void
runSpMVBenchmarks( Benchmark & benchmark,
                   Benchmark::MetadataMap metadata,
                   const String & inputFileName )
{
    const String precision = getType< Real >();
    metadata["precision"] = precision;

    // Sparse matrix-vector multiplication
    benchmark.newBenchmark( String("Sparse matrix-vector multiplication (") + precision + ")",
                            metadata );
    benchmarkSpmvSynthetic< Real >( benchmark, inputFileName );
}

void
setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file name." );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-spmv.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "all" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String & inputFileName = parameters.getParameter< String >( "input-file" );
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName.getString(), mode );

   // init benchmark and common metadata
   Benchmark benchmark( loops, verbose );

   // prepare global metadata
   Benchmark::MetadataMap metadata = getHardwareMetadata();
   
   
   // DO: Pass the inputFileName parameter and get rows and cols from it to create the cout GUI.
   if( precision == "all" || precision == "float" )
      runSpMVBenchmarks< float >( benchmark, metadata, inputFileName );
   if( precision == "all" || precision == "double" )
      runSpMVBenchmarks< double >( benchmark, metadata, inputFileName );

   if( ! benchmark.save( logFile ) ) {
      std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
      return EXIT_FAILURE;
   }

   std::cout << "\n== BENCHMARK FINISHED ==" << std::endl;
   return EXIT_SUCCESS;
}
