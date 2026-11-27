#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Config/parseCommandLine.h>

#include "matrix-info.h"
#include "matrix-list.h"

#include <exception>

using namespace TNL;
using namespace TNL::Benchmarks;

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Matrix info settings:" );
   config.addEntry< String >(
      "input-file", "Input file name. Required, unless --matrix-list is given instead.", "" );
   config.addEntry< String >(
      "matrix-name", "Matrix name used as the join key with the SpMV benchmark logs. Defaults to the input file name.", "" );
   config.addEntry< String >(
      "matrix-list",
      "File listing the remaining matrices to process, one path per line (the format "
      "scripts/run-tnl-benchmark-spmv-matrix-info's directory scan writes). When given, takes "
      "priority over --input-file: every matrix in the list is processed in this single process, "
      "and removed from the list only once it is fully done, so a crash leaves it at the head of "
      "the list for the next run to resume.",
      "" );
   config.addEntry< String >(
      "matrix-list-base-path",
      "Base path stripped from each --matrix-list entry to derive its matrix name.",
      "./mtx_matrices" );
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
   config.addEntry< String >(
      "histogram-dir",
      "If set, also write one file per matrix orientation into this directory, with the histogram "
      "of the nonzero counts of the rows in a sparse form: one line '<nonzeros per row> <number of "
      "rows>' for each row length occurring in the matrix, in ascending order of the row length.",
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

   auto processCurrentMatrix = [ & ]()
   {
      const String& inputFileName = parameters.getParameter< String >( "input-file" );
      const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );
      TNL::Benchmarks::SpMV::writeMatrixInfo( inputFileName, parameters, verboseMR );
   };

   try {
      const String& matrixListFile = parameters.getParameter< String >( "matrix-list" );
      if( ! matrixListFile.empty() ) {
         const std::string basePath = parameters.getParameter< String >( "matrix-list-base-path" ).getString();
         // no --progress-dir: tnl-benchmark-spmv-matrix-info is atomic per matrix, nothing to resume mid-matrix
         SpMV::runOverMatrixList( parameters, matrixListFile.getString(), basePath, "", processCurrentMatrix );
      }
      else {
         if( parameters.getParameter< String >( "input-file" ).empty() ) {
            std::cerr << "ERROR: Input file name is required (or pass --matrix-list)." << std::endl;
            return EXIT_FAILURE;
         }
         processCurrentMatrix();
      }
   }
   catch( const std::exception& ex ) {
      std::cerr << ex.what() << std::endl;
      return EXIT_FAILURE;
   }

   std::cout << "\n==> MATRIX INFO FINISHED" << std::endl;
   return EXIT_SUCCESS;
}
