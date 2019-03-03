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

using namespace TNL;
using namespace TNL::Benchmarks;


template< typename Real >
void
runSpMVBenchmarks( Benchmark & benchmark,
                   Benchmark::MetadataMap metadata,
                   const std::size_t & size,
                   const int & elementsPerRow )
{
   const String precision = getType< Real >();
   metadata["precision"] = precision;

   // Array operations
   benchmark.newBenchmark( String("Array operations (") + precision + ")",
                           metadata );
   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "size", convertToString( size ) }, } ));
   benchmarkArrayOperations< Real >( benchmark, size );

   // Vector operations
   benchmark.newBenchmark( String("Vector operations (") + precision + ")",
                           metadata );
   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "size", convertToString( size ) }, } ));
   benchmarkVectorOperations< Real >( benchmark, size );

   // Sparse matrix-vector multiplication
   benchmark.newBenchmark( String("Sparse matrix-vector multiplication (") + precision + ")",
                           metadata );
   benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         { "rows", convertToString( size ) },
         { "columns", convertToString( size ) },
         { "elements per row", convertToString( elementsPerRow ) },
      } ));
   benchmarkSpmvSynthetic< Real >( benchmark, size, elementsPerRow );
}

void
setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file name." );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-blas.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "size", "Size of arrays/vectors used in the benchmark.", 100000 );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "elements-per-row", "Number of elements per row of the sparse matrix used in the matrix-vector multiplication benchmark.", 5 );
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

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
//   const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
//   const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const std::size_t size = parameters.getParameter< int >( "size" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int elementsPerRow = parameters.getParameter< int >( "elements-per-row" );
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

   if( precision == "all" || precision == "float" )
      runSpMVBenchmarks< float >( benchmark, metadata, size, elementsPerRow );
   if( precision == "all" || precision == "double" )
      runSpMVBenchmarks< double >( benchmark, metadata, size, elementsPerRow );

   if( ! benchmark.save( logFile ) ) {
      std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
