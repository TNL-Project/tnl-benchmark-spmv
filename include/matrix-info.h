#pragma once

#include <filesystem>
#include <fstream>

#include <TNL/Benchmarks/JsonLogging.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/SparseMatrix.h>

#include "MatrixStatistics.h"
#include "MatrixTransposition.h"
#include "matrix-name.h"

using namespace TNL::Matrices;

namespace TNL::Benchmarks::SpMV {

// Writes the per-row nonzero counts of `matrix` into
// `<dir>/<matrixName>[.transposed].txt`, one count per line, in row order.
// Used for offline analysis of the row fill-in without re-parsing the MTX file.
template< typename Matrix >
void
writeFillingFile( const std::string& dir, const String& matrixName, bool transposed, const Matrix& matrix )
{
   using IndexType = typename Matrix::IndexType;

   TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > rowLengths( matrix.getRows() );
   matrix.getCompressedRowLengths( rowLengths );

   const std::filesystem::path path = matrixFilePath( dir, matrixName, transposed, "txt" );
   std::ofstream file( path );
   for( IndexType i = 0; i < rowLengths.getSize(); i++ )
      file << rowLengths[ i ] << '\n';
   file.flush();
}

// Writes one JSON line with the structural info (dimensions, symmetry,
// MatrixStatistics) of `matrix` - the original or transposed orientation of
// the matrix named `matrixName` - to `logger`. Optionally also writes the
// per-row fill-in file when `fillingDir` is not empty.
template< typename Matrix >
void
writeMatrixInfoRecord( TNL::Benchmarks::JsonLogging& logger,
                       const String& matrixName,
                       bool transposed,
                       const Matrix& matrix,
                       const MatrixSymmetry& symmetry,
                       const std::string& fillingDir )
{
   const auto nonzeros = matrix.getNonzeroElementsCount();
   MatrixStatistics< Matrix > stats( matrix );

   logger.setMetadataColumns( { { "matrix name", matrixName },
                                 { "transposed", transposed ? "true" : "false" },
                                 { "symmetric", symmetry.symmetric ? "true" : "false" },
                                 { "structurally symmetric", symmetry.structurallySymmetric ? "true" : "false" },
                                 { "rows", convertToString( matrix.getRows() ) },
                                 { "columns", convertToString( matrix.getColumns() ) },
                                 { "nonzeros", convertToString( nonzeros ) },
                                 { "nonzeros/row average", convertToString( stats.getAverage() ) },
                                 { "nonzeros/row std_dev", convertToString( stats.getStddev() ) },
                                 { "nonzeros/row P50", convertToString( stats.getP50() ) },
                                 { "nonzeros/row P95", convertToString( stats.getP95() ) },
                                 { "nonzeros/row P99", convertToString( stats.getP99() ) },
                                 { "nonzeros/row R50", convertToString( stats.getR50() ) },
                                 { "nonzeros/row R95", convertToString( stats.getR95() ) },
                                 { "nonzeros/row R99", convertToString( stats.getR99() ) },
                                 { "warps/row average", convertToString( stats.getAverageWarpsPerRow() ) },
                                 { "rho ELL", convertToString( stats.getRhoELL() ) },
                                 { "rho SELL 2", convertToString( stats.getRhoSELL()[ 0 ] ) },
                                 { "rho SELL 4", convertToString( stats.getRhoSELL()[ 1 ] ) },
                                 { "rho SELL 8", convertToString( stats.getRhoSELL()[ 2 ] ) },
                                 { "rho SELL 16", convertToString( stats.getRhoSELL()[ 3 ] ) },
                                 { "rho SELL 32", convertToString( stats.getRhoSELL()[ 4 ] ) },
                                 { "eta SELL 2", convertToString( stats.getEtaSELL()[ 0 ] ) },
                                 { "eta SELL 4", convertToString( stats.getEtaSELL()[ 1 ] ) },
                                 { "eta SELL 8", convertToString( stats.getEtaSELL()[ 2 ] ) },
                                 { "eta SELL 16", convertToString( stats.getEtaSELL()[ 3 ] ) },
                                 { "eta SELL 32", convertToString( stats.getEtaSELL()[ 4 ] ) },
                                 { "entropy", convertToString( stats.getEntropy() ) },
                                 { "Gini", convertToString( stats.getGini() ) } } );
   logger.writeRow( {}, {}, "" );

   if( ! fillingDir.empty() )
      writeFillingFile( fillingDir, matrixName, transposed, matrix );
}

// Reads the matrix named by the "input-file" parameter and appends its
// structural info (and the transposed orientation's, when applicable) to the
// "output-file". Mirrors the symmetry/transposition logic of benchmarkSpmv()
// in spmv.h, so that the records written here match exactly what
// tnl-benchmark-spmv (and the other SpMV benchmark binaries) will benchmark.
template< typename Index = int >
void
writeMatrixInfo( const String& inputFileName, const Config::ParameterContainer& parameters, bool verboseMR )
{
   using CSRHostMatrix = Matrices::SparseMatrix< double, TNL::Devices::Host, Index >;

   CSRHostMatrix csrHostMatrix;
   MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
   TNL::Matrices::compressSparseMatrix( csrHostMatrix );

   String matrixName = parameters.getParameter< String >( "matrix-name" );
   if( matrixName.empty() )
      matrixName = inputFileName;

   const MatrixSymmetry symmetry = getMatrixSymmetry( inputFileName.getString(), csrHostMatrix );

   const String& outputFileName = parameters.getParameter< String >( "output-file" );
   const String& outputMode = parameters.getParameter< String >( "output-mode" );
   auto mode = std::ios::out;
   if( outputMode == "append" )
      mode |= std::ios::app;
   std::ofstream outputFile( outputFileName.getString(), mode );
   TNL::Benchmarks::JsonLogging logger( outputFile );

   const std::string fillingDir = parameters.getParameter< String >( "filling-dir" ).getString();

   const String& withTransposedMatrix = parameters.getParameter< String >( "with-transposed-matrix" );
   const bool withTransposition = withTransposedMatrix != "false" && ! symmetry.structurallySymmetric;

   if( withTransposedMatrix != "only" )
      writeMatrixInfoRecord( logger, matrixName, false, csrHostMatrix, symmetry, fillingDir );

   if( withTransposition ) {
      CSRHostMatrix transposedMatrix;
      getSortedTransposition( transposedMatrix, csrHostMatrix );
      writeMatrixInfoRecord( logger, matrixName, true, transposedMatrix, symmetry, fillingDir );
   }
}

}  // namespace TNL::Benchmarks::SpMV
