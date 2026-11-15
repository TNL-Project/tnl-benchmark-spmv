#pragma once

#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <system_error>

#include "matrix-name.h"

namespace TNL::Benchmarks::SpMV {

// Tracks which (precision, format, device, launch cfg) combinations have already been timed and
// logged for the matrix orientation currently being benchmarked, so a resumed run can skip
// straight to the first not-yet-done combination instead of redoing the whole matrix. The backing
// file is <dir>/<matrixName>[.transposed].progress, one "precision\tformat\tdevice\tlaunch cfg"
// line per completed combination - the same key TNL.BenchmarkLogs' duplicate_key uses to identify
// a benchmark record, minus the matrix name/transposed (implicit: one file per matrix orientation).
class BenchmarkProgress
{
public:
   BenchmarkProgress( const std::string& dir, const String& matrixName, bool transposed )
   : filePath( matrixFilePath( dir, matrixName, transposed, "progress" ) )
   {
      std::ifstream file( filePath );
      for( std::string line; std::getline( file, line ); )
         if( ! line.empty() )
            done.insert( line );
   }

   [[nodiscard]] bool
   isDone( const std::string& precision,
           const std::string& format,
           const std::string& device,
           const std::string& launchConfig ) const
   {
      return done.count( key( precision, format, device, launchConfig ) ) > 0;
   }

   // Appends and flushes immediately: the benchmark process can be terminated at any time, so a
   // completed combination must be durable on disk before moving on to the next one, not just
   // sitting in a buffer.
   void
   markDone( const std::string& precision,
             const std::string& format,
             const std::string& device,
             const std::string& launchConfig )
   {
      const std::string k = key( precision, format, device, launchConfig );
      done.insert( k );
      std::ofstream file( filePath, std::ios::app );
      file << k << '\n';
      file.flush();
   }

private:
   static std::string
   key( const std::string& precision, const std::string& format, const std::string& device, const std::string& launchConfig )
   {
      return precision + '\t' + format + '\t' + device + '\t' + launchConfig;
   }

   std::filesystem::path filePath;
   std::set< std::string > done;
};

// Deletes both of a matrix's progress files (original and transposed orientation), if present.
// Called once a matrix has been fully processed - every precision, every orientation - right
// before it is removed from the matrix list: its progress files are no longer needed at that
// point. Safe to call even if a file was never created (e.g. the transposed orientation wasn't
// benchmarked for a structurally symmetric matrix).
inline void
removeMatrixProgress( const std::string& dir, const String& matrixName )
{
   std::error_code ec;
   std::filesystem::remove( matrixFilePath( dir, matrixName, false, "progress" ), ec );
   std::filesystem::remove( matrixFilePath( dir, matrixName, true, "progress" ), ec );
}

}  // namespace TNL::Benchmarks::SpMV
