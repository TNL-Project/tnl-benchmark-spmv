#include "../spmv.h"
namespace TNL::Benchmarks::SpMV {
template void dispatchSymmetricBinary< double >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< double, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL::Benchmarks::SpMV
