#pragma once

#include <algorithm>
#include <string>

#include <TNL/Devices/Host.h>

namespace TNL::Benchmarks::SpMV {

// Name of the build of the benchmark written to the "build" metadata column,
// so that the records of the host and the GPU builds can be told apart even
// in merged log files.
inline std::string
getBuildName()
{
#if defined( __CUDACC__ )
   return "cuda";
#elif defined( __HIP__ )
   return "hip";
#else
   return "host";
#endif
}

// The GPU builds are compiled without OpenMP and they usually run on a different
// machine than the host build, so their CPU results are not comparable with the
// results of the host build. They benchmark the CPU only on request.
#if defined( __CUDACC__ ) || defined( __HIP__ )
constexpr bool cpuTestsByDefault = false;
#else
constexpr bool cpuTestsByDefault = true;
#endif

// Number of threads the CPU computations really run with - Devices::Host::getMaxThreadsCount()
// returns the configured value even when OpenMP is disabled at run time.
inline int
getCpuThreadsCount()
{
   if( ! Devices::Host::isOMPEnabled() )
      return 1;
   return std::max( 1, Devices::Host::getMaxThreadsCount() );
}

// Launch config of a CPU benchmark written to the "launch cfg." metadata column
inline std::string
getCpuLaunchConfig( int threads )
{
   if( threads == 1 )
      return "1 thread";
   return std::to_string( threads ) + " threads";
}

// Call f( launchConfig ) with the number of threads set for the CPU computations
// to 1, 2, 4, ... up to the maximal number of threads if `sweep` is true, or just
// to the maximal number of threads otherwise. The maximal number of threads is
// set again when the function returns.
template< typename Function >
void
forEachCpuThreadsCount( bool sweep, Function&& f )
{
   const int maxThreadsCount = getCpuThreadsCount();
   int threads = sweep ? 1 : maxThreadsCount;
   while( true ) {
      Devices::Host::setMaxThreadsCount( threads );
      f( getCpuLaunchConfig( threads ) );
      if( threads == maxThreadsCount )
         break;
      threads = std::min( 2 * threads, maxThreadsCount );
   }
}

}  // namespace TNL::Benchmarks::SpMV
