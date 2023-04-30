#include "../spmv.h"
namespace TNL::Benchmarks::SpMV {
template void dispatchSpMV< double >( BenchmarkType&, const Containers::Vector< double, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL::Benchmarks::SpMV
