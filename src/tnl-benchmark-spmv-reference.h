#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/MPI/ScopedInitializer.h>

#include "spmv-reference.h"
#include "matrix-list.h"

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
   config.addEntry< String >(
      "with-transposed-matrix",
      "Benchmark also (true), not (false) or only (only) the transposed matrix. The transposition of a structurally "
      "symmetric matrix is never benchmarked, since it has the same pattern.",
      "true" );
   config.addEntryEnum( "false" );
   config.addEntryEnum( "true" );
   config.addEntryEnum( "only" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< bool >( "with-cpu-tests",
                            "Benchmark also the CPU. The GPU builds skip it by default, the reference result "
                            "the other formats are compared with is computed anyway.",
                            TNL::Benchmarks::SpMV::cpuTestsByDefault );
   config.addEntry< int >( "verbose-MReader", "Verbose mode for Matrix Reader.", 0 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   // MPI must be initialized because this binary links against Hypre/Ginkgo,
   // which require it even when running on a single rank.
#ifdef HAVE_MPI
   TNL::MPI::ScopedInitializer mpi( argc, argv );
#endif

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
