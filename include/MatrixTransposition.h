#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Devices/Host.h>
#include <TNL/Functional.h>

namespace TNL::Benchmarks::SpMV {

// Symmetry of the benchmarked matrix, written to the log. The transposition of
// a structurally symmetric matrix has the same pattern, so benchmarking it
// would only repeat the results for the original matrix.
struct MatrixSymmetry
{
   // the MTX file declares the matrix as symmetric
   bool symmetric = false;
   // the pattern of the nonzero elements is symmetric, values may differ
   bool structurallySymmetric = false;
};

// Check if the header of the MTX file declares a symmetric matrix, i.e. it reads
// "%%MatrixMarket matrix coordinate <field> symmetric".
inline bool
isMtxFileSymmetric( const std::string& fileName )
{
   std::ifstream file( fileName );
   std::string header;
   if( ! std::getline( file, header ) )
      return false;
   std::transform( header.begin(),
                   header.end(),
                   header.begin(),
                   []( unsigned char c )
                   {
                      return std::tolower( c );
                   } );
   std::istringstream stream( header );
   std::vector< std::string > words;
   for( std::string word; stream >> word; )
      words.push_back( word );
   return words.size() >= 5 && words[ 0 ] == "%%matrixmarket" && words[ 4 ] == "symmetric";
}

// Check if the pattern of the nonzero elements of a CSR matrix on the host is
// symmetric. The column indexes in each row must be sorted, as they are in a
// matrix read by MatrixReader, so the transposed element can be found by
// binary search.
template< typename Matrix >
bool
isStructurallySymmetric( const Matrix& matrix )
{
   using Index = typename Matrix::IndexType;

   if( matrix.getRows() != matrix.getColumns() )
      return false;
   const Index* offsets = matrix.getSegments().getOffsets().getData();
   const Index* columns = matrix.getColumnIndexes().getData();
   auto fetch = [ = ]( Index row ) -> bool
   {
      for( Index i = offsets[ row ]; i < offsets[ row + 1 ]; i++ ) {
         const Index column = columns[ i ];
         if( ! std::binary_search( columns + offsets[ column ], columns + offsets[ column + 1 ], row ) )
            return false;
      }
      return true;
   };
   return Algorithms::reduce< Devices::Host >( (Index) 0, matrix.getRows(), fetch, TNL::LogicalAnd{}, true );
}

template< typename Matrix >
MatrixSymmetry
getMatrixSymmetry( const std::string& fileName, const Matrix& matrix )
{
   MatrixSymmetry symmetry;
   symmetry.symmetric = isMtxFileSymmetric( fileName );
   symmetry.structurallySymmetric = symmetry.symmetric || isStructurallySymmetric( matrix );
   return symmetry;
}

// Compute the transposition of a matrix on the host with the column indexes in
// each row sorted. SparseMatrix::getTransposition fills the rows of the
// transposition in parallel, so with OpenMP the order of the elements in a row
// would depend on the scheduling of the threads, which affects the performance
// of the SpMV. In one thread, the rows are filled in the order of the columns.
template< typename Matrix >
void
getSortedTransposition( Matrix& transposition, const Matrix& matrix )
{
   struct SequentialScope
   {
      bool ompEnabled = Devices::Host::isOMPEnabled();

      SequentialScope()
      {
         Devices::Host::disableOMP();
      }

      ~SequentialScope()
      {
         if( ompEnabled )
            Devices::Host::enableOMP();
      }
   } sequential;
   transposition.getTransposition( matrix );
}

}  // namespace TNL::Benchmarks::SpMV
