#pragma once

#include <cstdint>
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
// file is <dir>/<matrixName>[.transposed][.rankN].progress, one "precision\tformat\tdevice\tlaunch
// cfg" line per completed combination - the same key TNL.BenchmarkLogs' duplicate_key uses to
// identify a benchmark record, minus the matrix name/transposed (implicit: one file per matrix
// orientation, and per MPI rank when running with more than one).
//
// When `worldSize > 1` (several MPI ranks, each bound to its own GPU, splitting one matrix's
// sweep between them), each *format* is additionally owned by exactly one rank, chosen by hashing
// its name - isDone() reports "done" (i.e. skip) for every format not owned by `rank`, in addition
// to combinations this rank has already actually completed. The owning rank is recomputed from the
// format name alone, independently by every rank, so no coordination between ranks is needed to
// agree on who owns what.
class BenchmarkProgress
{
public:
   BenchmarkProgress( const std::string& dir, const String& matrixName, bool transposed, int rank = 0, int worldSize = 1 )
   : filePath( matrixFilePath( dir,
                               matrixName,
                               transposed,
                               worldSize > 1 ? "rank" + std::to_string( rank ) + ".progress" : std::string( "progress" ) ) ),
     rank( rank ),
     worldSize( worldSize )
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
      if( ! isMine( format ) )
         return true;
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

   // FNV-1a: a small, fully deterministic string hash. Used instead of std::hash<std::string>,
   // which the C++ standard does not guarantee to be stable across processes, builds or standard
   // library versions - every MPI rank must compute exactly the same hash for the same format name
   // to independently agree on who owns it, with no coordination at all.
   static std::uint64_t
   fnv1a( const std::string& s )
   {
      std::uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
      for( unsigned char c : s ) {
         h ^= c;
         h *= 0x100000001b3ULL;  // FNV prime
      }
      return h;
   }

   [[nodiscard]] bool
   isMine( const std::string& format ) const
   {
      if( worldSize <= 1 )
         return true;
      return static_cast< int >( fnv1a( format ) % static_cast< std::uint64_t >( worldSize ) ) == rank;
   }

   std::filesystem::path filePath;
   std::set< std::string > done;
   int rank;
   int worldSize;
};

// Deletes a matrix's progress files (original and transposed orientation, every rank's when
// `worldSize > 1`), if present. Called once a matrix has been fully processed - every precision,
// every orientation, every rank - right before it is removed from the matrix list: its progress
// files are no longer needed at that point. Safe to call even if a file was never created (e.g.
// the transposed orientation wasn't benchmarked for a structurally symmetric matrix).
inline void
removeMatrixProgress( const std::string& dir, const String& matrixName, int worldSize = 1 )
{
   std::error_code ec;
   if( worldSize <= 1 ) {
      std::filesystem::remove( matrixFilePath( dir, matrixName, false, "progress" ), ec );
      std::filesystem::remove( matrixFilePath( dir, matrixName, true, "progress" ), ec );
      return;
   }
   for( int rank = 0; rank < worldSize; rank++ ) {
      const std::string ext = "rank" + std::to_string( rank ) + ".progress";
      std::filesystem::remove( matrixFilePath( dir, matrixName, false, ext ), ec );
      std::filesystem::remove( matrixFilePath( dir, matrixName, true, ext ), ec );
   }
}

}  // namespace TNL::Benchmarks::SpMV
