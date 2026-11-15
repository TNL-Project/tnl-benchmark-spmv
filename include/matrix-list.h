#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark-progress.h"
#include "matrix-name.h"

namespace TNL::Benchmarks::SpMV {

// Manages a flat, newline-separated list of remaining matrix file paths (the same format the
// orchestration scripts' directory scan already produces). The list is read fully into memory
// once; removeFirst() rewrites the file with the first entry gone.
//
// A matrix must be removed only once it is FULLY processed, never optimistically at the start:
// the benchmark process can be killed at any time (segfault, SIGKILL, wall-clock limit, ...), and
// as long as a matrix stays at the head of the list, re-running with the same list file (and the
// same --progress-dir, see benchmark-progress.h) resumes it instead of silently skipping it.
class MatrixList
{
public:
   explicit MatrixList( std::string path )
   : path( std::move( path ) )
   {
      std::ifstream file( this->path );
      for( std::string line; std::getline( file, line ); )
         if( ! line.empty() )
            entries.push_back( line );
   }

   [[nodiscard]] bool
   empty() const
   {
      return entries.empty();
   }

   [[nodiscard]] const std::string&
   front() const
   {
      return entries.front();
   }

   [[nodiscard]] std::size_t
   size() const
   {
      return entries.size();
   }

   // Removes the first entry and durably rewrites the list file: written to a temporary file,
   // flushed and closed, then renamed over the original - the rename is atomic, so a crash right
   // after this call never leaves the list file truncated or half-written.
   void
   removeFirst()
   {
      entries.erase( entries.begin() );

      const std::string tmpPath = path + ".tmp";
      {
         std::ofstream file( tmpPath, std::ios::trunc );
         for( const std::string& entry : entries )
            file << entry << '\n';
         file.flush();
      }
      std::filesystem::rename( tmpPath, path );
   }

private:
   std::string path;
   std::vector< std::string > entries;
};

// Loads `listFile` and calls `processMatrix()` once for every remaining matrix, in order,
// replacing the former per-matrix shell loop in scripts/run-tnl-benchmark-spmv. Before each call,
// "input-file"/"matrix-name" are set on `parameters` (matrix name derived from the path via
// matrixNameFromPath(), the same convention the orchestration scripts use). A matrix is removed
// from the list - and its progress file(s) deleted, when `progressDir` is not empty - only after
// `processMatrix()` returns, whether it threw a (caught) exception or not: a crash that takes the
// whole process down leaves the matrix at the head of the list, so re-running with the same
// listFile/progressDir resumes it instead of silently skipping it.
template< typename ProcessMatrix >
void
runOverMatrixList( Config::ParameterContainer& parameters,
                    const std::string& listFile,
                    const std::string& basePath,
                    const std::string& progressDir,
                    ProcessMatrix processMatrix )
{
   MatrixList matrixList( listFile );
   const std::size_t total = matrixList.size();

   while( ! matrixList.empty() ) {
      const std::string path = matrixList.front();
      const std::string name = matrixNameFromPath( path, basePath );
      std::cout << "\n==> " << ( total - matrixList.size() + 1 ) << " / " << total << " : matrix " << name << " (" << path
                << ")" << std::endl;

      parameters.setParameter( "input-file", String( path ) );
      parameters.setParameter( "matrix-name", String( name ) );
      try {
         processMatrix();
      }
      catch( const std::exception& e ) {
         std::cerr << "Unable to benchmark matrix " << name << ": " << e.what() << std::endl;
      }

      if( ! progressDir.empty() )
         removeMatrixProgress( progressDir, String( name ) );
      matrixList.removeFirst();
   }
}

}  // namespace TNL::Benchmarks::SpMV
