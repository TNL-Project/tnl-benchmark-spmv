#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Config/parseCommandLine.h>

#include "matrix-info.h"

#include <exception>

using namespace TNL;
using namespace TNL::Benchmarks;

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Matrix info settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file name." );
   config.addEntry< String >(
      "matrix-name", "Matrix name used as the join key with the SpMV benchmark logs. Defaults to the input file name.", "" );
   config.addEntry< String >( "output-file", "JSONL file the matrix info is appended to.", "matrix-info.jsonl" );
   config.addEntry< String >( "output-mode", "Mode for opening the output file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >(
      "with-transposed-matrix",
      "Also write info for the transposed matrix (true), not (false) or only (only). Mirrors the "
      "with-transposed-matrix option of tnl-benchmark-spmv; the transposition of a structurally "
      "symmetric matrix is never written, since it has the same pattern.",
      "true" );
   config.addEntryEnum( "false" );
   config.addEntryEnum( "true" );
   config.addEntryEnum( "only" );
   config.addEntry< String >(
      "filling-dir",
      "If set, also write one file per matrix orientation into this directory, with the nonzero "
      "count of each row (one count per line, in row order).",
      "" );
   config.addEntry< int >( "verbose-MReader", "Verbose mode for Matrix Reader.", 0 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
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

   const String& inputFileName = parameters.getParameter< String >( "input-file" );
   const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );

   if( inputFileName.empty() ) {
      std::cerr << "ERROR: Input file name is required." << std::endl;
      return EXIT_FAILURE;
   }

   try {
      TNL::Benchmarks::SpMV::writeMatrixInfo( inputFileName, parameters, verboseMR );
   }
   catch( const std::exception& ex ) {
      std::cerr << ex.what() << std::endl;
      return EXIT_FAILURE;
   }

   std::cout << "\n==> MATRIX INFO FINISHED" << std::endl;
   return EXIT_SUCCESS;
}
