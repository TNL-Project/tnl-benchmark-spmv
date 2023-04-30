#include "../spmv.h"
namespace TNL::Benchmarks::SpMV {
template void dispatchSymmetricBinary< float >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< float, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL::Benchmarks::SpMV
