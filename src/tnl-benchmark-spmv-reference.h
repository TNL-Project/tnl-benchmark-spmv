#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>

#include "spmv-reference.h"

#ifdef HAVE_PETSC
   #include <petscmat.h>
#endif

using namespace TNL::Matrices;

#include <exception>

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Real >
void
runSpMVBenchmarks( Benchmark& benchmark,
                   const String& inputFileName,
                   const Config::ParameterContainer& parameters,
                   bool verboseMR = false )
{
   // Start the actual benchmark in spmv.h
   try {
      TNL::Benchmarks::SpMV::benchmarkSpmv< Real >( benchmark, inputFileName, parameters, verboseMR );
   }
   catch( const std::exception& ex ) {
      std::cerr << ex.what() << std::endl;
   }
}

void
setupConfig( Config::ConfigDescription& config )
{
   Benchmark::configSetup( config );

   config.addDelimiter( "SpMV reference benchmark settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file name." );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "verbose-MReader", "Verbose mode for Matrix Reader.", 0 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
#ifdef HAVE_PETSC
   PetscInitialize( &argc, &argv, nullptr, nullptr );
#endif

   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   Devices::Host::setup( parameters );
   Devices::Cuda::setup( parameters );

   const String& inputFileName = parameters.getParameter< String >( "input-file" );
   const String& precision = parameters.getParameter< String >( "precision" );
   const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );

   if( inputFileName.empty() ) {
      std::cerr << "ERROR: Input file name is required." << std::endl;
      return EXIT_FAILURE;
   }

   Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   // Initiate setup of benchmarks
   if( precision == "all" || precision == "float" )
      runSpMVBenchmarks< float >( benchmark, inputFileName, parameters, verboseMR );
   if( precision == "all" || precision == "double" )
      runSpMVBenchmarks< double >( benchmark, inputFileName, parameters, verboseMR );

   // Confirm that the benchmark has finished
   std::cout << "\n==> BENCHMARK FINISHED" << std::endl;
   return EXIT_SUCCESS;
}
