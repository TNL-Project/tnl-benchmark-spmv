// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>

#include "spmv-legacy.h"
#include "matrix-list.h"

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

   config.addDelimiter( "SpMV legacy benchmark settings:" );
   config.addEntry< String >(
      "input-file", "Input file name. Required, unless --matrix-list is given instead.", "" );
   config.addEntry< String >( "matrix-name", "Matrix name used in the log. Defaults to the input file name.", "" );
   config.addEntry< String >(
      "matrix-list",
      "File listing the remaining matrices to benchmark, one path per line (the format "
      "scripts/run-tnl-benchmark-spmv's directory scan writes). When given, takes priority over "
      "--input-file: every matrix in the list is benchmarked in this single process, and removed "
      "from the list only once it is fully done, so a crash leaves it at the head of the list for "
      "the next run to resume (together with --progress-dir).",
      "" );
   config.addEntry< String >(
      "matrix-list-base-path",
      "Base path stripped from each --matrix-list entry to derive its matrix name.",
      "./mtx_matrices" );
   config.addEntry< String >(
      "progress-dir",
      "Directory for the per-matrix progress files, used to resume a matrix that was interrupted "
      "partway through its sweep of formats/launch configs instead of redoing it from scratch.",
      "progress" );
   config.addEntry< bool >( "with-ellpack-formats", "Perform benchmark for Ellpack based matrix formats.", true );
   config.addEntry< bool >( "with-cpu-tests",
                            "Benchmark also the CPU. The GPU builds skip it by default, the reference result "
                            "the other formats are compared with is computed anyway.",
                            TNL::Benchmarks::SpMV::cpuTestsByDefault );
   config.addEntry< bool >(
      "with-all-cpu-tests", "Benchmark all matrix formats on the CPU, not only CSR. Requires with-cpu-tests.", false );
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
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   Devices::Host::setup( parameters );
   Devices::Cuda::setup( parameters );

   Benchmark benchmark;
   benchmark.setup( parameters, argv[ 0 ] );

   auto processCurrentMatrix = [ & ]()
   {
      const String& inputFileName = parameters.getParameter< String >( "input-file" );
      const String& precision = parameters.getParameter< String >( "precision" );
      const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );
      if( precision == "all" || precision == "float" )
         runSpMVBenchmarks< float >( benchmark, inputFileName, parameters, verboseMR );
      if( precision == "all" || precision == "double" )
         runSpMVBenchmarks< double >( benchmark, inputFileName, parameters, verboseMR );
   };

   const String& matrixListFile = parameters.getParameter< String >( "matrix-list" );
   if( ! matrixListFile.empty() ) {
      const std::string basePath = parameters.getParameter< String >( "matrix-list-base-path" ).getString();
      const std::string progressDir = parameters.getParameter< String >( "progress-dir" ).getString();
      SpMV::runOverMatrixList( parameters, matrixListFile.getString(), basePath, progressDir, processCurrentMatrix );
   }
   else {
      if( parameters.getParameter< String >( "input-file" ).empty() ) {
         std::cerr << "ERROR: Input file name is required (or pass --matrix-list)." << std::endl;
         return EXIT_FAILURE;
      }
      processCurrentMatrix();
   }

   // Confirm that the benchmark has finished
   std::cout << "\n==> BENCHMARK FINISHED" << std::endl;
   return EXIT_SUCCESS;
}
