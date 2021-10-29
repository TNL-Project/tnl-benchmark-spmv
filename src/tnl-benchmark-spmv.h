/***************************************************************************
                          tnl-benchmark-spmv.h  -  description
                             -------------------
    begin                : March 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>

#include "spmv.h"

#include <TNL/Matrices/MatrixReader.h>

#ifdef HAVE_PETSC
#include <petscmat.h>
#endif

using namespace TNL::Matrices;

#include <exception>
#include <ctime> // Used for file naming, so logs don't get overwritten.
#include <experimental/filesystem> // check file existence

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Real >
void
runSpMVBenchmarks( TNL::Benchmarks::SpMV::BenchmarkType & benchmark,
                   TNL::Benchmarks::SpMV::BenchmarkType::MetadataMap metadata,
                   const String & inputFileName,
                   const Config::ParameterContainer& parameters,
                   bool verboseMR = false )
{
   const String precision = getType< Real >();
   metadata["precision"] = precision;

   // Sparse matrix-vector multiplication
   benchmark.newBenchmark( String("Sparse matrix-vector multiplication (") + precision + ")",
                           metadata );
   // Start the actual benchmark in spmv.h
   try {
      TNL::Benchmarks::SpMV::benchmarkSpmv< Real >( benchmark, inputFileName, parameters, verboseMR );
   }
   catch( const std::exception& ex ) {
      std::cerr << ex.what() << std::endl;
   }
}

// Get current date time to have different log files names and avoid overwriting.
std::string getCurrDateTime()
{
   time_t rawtime;
   struct tm * timeinfo;
   char buffer[ 80 ];
   time( &rawtime );
   timeinfo = localtime( &rawtime );
   strftime( buffer, sizeof( buffer ), "%d-%m-%Y--%H:%M:%S", timeinfo );
   std::string curr_date_time( buffer );
   return curr_date_time;
}

void
setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "input-file", "Input file name.", "" );
   config.addEntry< bool >( "with-symmetric-matrices", "Perform benchmark even for symmetric matrix formats.", true );
   config.addEntry< bool >( "with-legacy-matrices", "Perform benchmark even for legacy TNL matrix formats.", true );
   config.addEntry< bool >( "with-all-cpu-tests", "All matrix formats are tested on both CPU and GPU. ", false );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-spmv::" + getCurrDateTime() + ".log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file - 'close' will only finalize the log file.", "append" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntryEnum( "close" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
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

   // FIXME: When ./tnl-benchmark-spmv-dbg is called without parameters:
   //           * The guide on what parameters to use prints twice.
   // FIXME: When ./tnl-benchmark-spmv-dbg is called with '--help':
   //           * The guide on what parameter to use print once.
   //              But then it CRASHES due to segfault:
   //              The program attempts to get unknown parameter openmp-enabled
   //              Aborting the program.
   //              terminate called after throwing an instance of 'int'
   //      [1]    17156 abort (core dumped)  ~/tnl-dev/Debug/bin/./tnl-benchmark-spmv-dbg --help

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String & inputFileName = parameters.getParameter< String >( "input-file" );
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   String outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );

   // open log file
   if( outputMode == "close" )
   {
      std::fstream file;
      file.open( logFileName.getString(), std::ios::out | std::ios::app );
      file << std::endl << "   ]" << std::endl << "}";
      return EXIT_SUCCESS;
   }
   if( inputFileName == "" )
   {
      std::cerr << "ERROR: Input file name is required." << std::endl;
      return EXIT_FAILURE;
   }
   bool logFileAppend( false );
   if( std::experimental::filesystem::exists(logFileName.getString()) )
   {
      logFileAppend = true;
      std::cout << "Log file " << logFileName << " exists and ";
      if( outputMode == "append" )
         std::cout << "new logs will be appended." << std::endl;
      else
         std::cout << "will be overwritten." << std::endl;
   }

   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName.getString(), mode );

   // init benchmark and common metadata
   TNL::Benchmarks::SpMV::BenchmarkType benchmark( loops, verbose, outputMode, logFileAppend );

   // prepare global metadata
   TNL::Benchmarks::SpMV::BenchmarkType::MetadataMap metadata = getHardwareMetadata< Logging >();

   // Initiate setup of benchmarks
   if( precision == "all" || precision == "float" )
      runSpMVBenchmarks< float >( benchmark, metadata, inputFileName, parameters, verboseMR );
   if( precision == "all" || precision == "double" )
      runSpMVBenchmarks< double >( benchmark, metadata, inputFileName, parameters, verboseMR );

   if( ! benchmark.save( logFile ) ) {
      std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
      return EXIT_FAILURE;
   }

   // Confirm that the benchmark has finished
   std::cout << "\n== BENCHMARK FINISHED ==" << std::endl;
   return EXIT_SUCCESS;
}
