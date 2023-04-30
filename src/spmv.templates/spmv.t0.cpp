#include "../spmv.h"
namespace TNL::Benchmarks::SpMV {
template void dispatchLegacy< float >( BenchmarkType&, const Containers::Vector< float, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL::Benchmarks::SpMV
