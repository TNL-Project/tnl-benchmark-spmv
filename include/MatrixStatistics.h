// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/sort.h>

template< typename MatrixType >
struct MatrixStatistics
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using DeviceType = typename MatrixType::DeviceType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using HostVector = TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType >;

   MatrixStatistics( const MatrixType& matrix )
   {
      const double warpSize = 32;
      VectorType rowLengths_( matrix.getRows() );
      matrix.getCompressedRowLengths( rowLengths_ );
      HostVector rowLengths( rowLengths_ );
      TNL::Algorithms::ascendingSort( rowLengths );
      const std::size_t rows = matrix.getRows();
      average = sum( rowLengths ) / rows;                       // Average number of nonzeros per row
      variance = sum( pow( rowLengths - average, 2 ) ) / rows;  // Variance of nonzeros per row
      stddev = sqrt( variance );                                // Standard deviation of  nonzeros per row
      P_50 = rowLengths[ rows * 0.5 ];                          // 50th percentile (median) of nonzeros per row
      P_95 = rowLengths[ (int) ( 0.95 * rows ) ];               // 95th percentile of nonzeros per row
      P_99 = rowLengths[ (int) ( 0.99 * rows ) ];               // 99th percentile of nonzeros per row
      R_50 = P_50 / average;                                    // Ratio of 50th percentile to average
      R_95 = P_95 / average;                                    // Ratio of 95th percentile to average
      R_99 = P_99 / average;                                    // Ratio of 99th percentile to average
      average_warps_per_row = sum( maximum( 1.0, rowLengths / warpSize ) ) / rows;  // Average number of warps per row
      rho_ELL = rows * max( rowLengths ) / matrix.getNonzeroElementsCount();        // ELLPACK fill-in ratio

      // Sliced ELLPACK fill-in ratio and efficiency calculation for different slice sizes
      rho_SELL.resize( 5 );
      for( int s = 0; s < 5; ++s ) {
         int sliceSize = 1 << ( s + 1 );  // Slice size (2, 4, 8, 16, 32)
         std::size_t allocatedNonzeros = 0;
         for( std::size_t i = 0; i < rows; i += sliceSize ) {
            allocatedNonzeros += sliceSize * max( rowLengths.getView( i, std::min( i + sliceSize, rows ) ) );
         }
         rho_SELL[ s ] = (double) allocatedNonzeros / (double) matrix.getNonzeroElementsCount();
      }

      // Sliced ELLPACK efficiency calculation
      eta_SELL.resize( 5 );
      for( int s = 0; s < 5; ++s ) {
         int sliceSize = 1 << ( s + 1 );  // Slice size (2, 4, 8, 16, 32)
         double average_to_max_ratio = 0.0;
         for( std::size_t i = 0; i < rows; i += sliceSize ) {
            std::size_t maxRowLength = max( rowLengths.getView( i, std::min( i + sliceSize, rows ) ) );
            double averageRowLength = sum( rowLengths.getView( i, std::min( i + sliceSize, rows ) ) ) / sliceSize;
            average_to_max_ratio += maxRowLength != 0 ? averageRowLength / maxRowLength : 1.0;
         }
         const std::size_t num_slices = std::ceil( static_cast< double >( rows ) / sliceSize );
         eta_SELL[ s ] = average_to_max_ratio / num_slices;
      }

      // Entropy calculation
      TNL::Containers::Vector< double > histogram( matrix.getColumns() + 1, 0.0 );
      for( std::size_t i = 0; i < rows; ++i )
         histogram[ rowLengths[ i ] ] += 1.0;
      histogram /= rows;  // Normalize to get probabilities
      H = -sum( histogram * log( histogram + 1e-10 ) );

      // Gini coefficient calculation
      double sum_i = 0.0;
      for( std::size_t i = 0; i < rows; ++i )
         sum_i += rowLengths[ i ] * i;
      Gini = ( 2 * sum_i ) / ( rows * sum( rowLengths ) ) - ( rows + 1.0 ) / rows;
   }

   //! \brief Get the average number of nonzeros per row.
   [[nodiscard]] double
   getAverage() const
   {
      return average;
   }

   //! \brief Get the variance of nonzeros per row.
   [[nodiscard]] double
   getVariance() const
   {
      return variance;
   }

   //! \brief Get the standard deviation of nonzeros per row.
   [[nodiscard]] double
   getStddev() const
   {
      return stddev;
   }

   //! \brief Get the 50th percentile (median) of nonzeros per row.
   [[nodiscard]] double
   getP50() const
   {
      return P_50;
   }

   //! \brief Get the 95th percentile of nonzeros per row.
   [[nodiscard]] double
   getP95() const
   {
      return P_95;
   }

   //! \brief Get the 99th percentile of nonzeros per row.
   [[nodiscard]] double
   getP99() const
   {
      return P_99;
   }

   //! \brief Get the ratio of the 50th percentile to the average.
   [[nodiscard]] double
   getR50() const
   {
      return R_50;
   }

   //! \brief Get the ratio of the 95th percentile to the average.
   [[nodiscard]] double
   getR95() const
   {
      return R_95;
   }

   //! \brief Get the ratio of the 99th percentile to the average.
   [[nodiscard]] double
   getR99() const
   {
      return R_99;
   }

   //! \brief Get the average number of warps per row.
   [[nodiscard]] double
   getAverageWarpsPerRow() const
   {
      return average_warps_per_row;
   }

   //! \brief Get the ELLPACK fill-in ratio.
   [[nodiscard]] double
   getRhoELL() const
   {
      return rho_ELL;
   }

   //! \brief Get the Sliced ELLPACK fill-in ratio for a given slice size index (0 to 4).
   [[nodiscard]] const std::vector< double >&
   getRhoSELL() const
   {
      return rho_SELL;
   }

   //! \brief Get the Sliced ELLPACK efficiency for a given slice size index (0 to 4).
   [[nodiscard]] const std::vector< double >&
   getEtaSELL() const
   {
      return eta_SELL;
   }

   //! \brief Get the entropy of the row length distribution.
   [[nodiscard]] double
   getEntropy() const
   {
      return H;
   }

   //! \brief Get the Gini coefficient of the row length distribution.
   [[nodiscard]] double
   getGini() const
   {
      return Gini;
   }

private:
   double average;                  // Average number of nonzeros per row
   double variance;                 // Variance of nonzeros per row
   double stddev;                   // Standard deviation of nonzeros per row
   double P_50;                     // 50th percentile (median) of nonzeros per row
   double P_95;                     // 95th percentile of nonzeros per row
   double P_99;                     // 99th percentile of nonzeros per row
   double R_50;                     // Ratio of 50th percentile to average
   double R_95;                     // Ratio of 95th percentile to average
   double R_99;                     // Ratio of 99th percentile to average
   double average_warps_per_row;    // Average number of warps per row
   double rho_ELL;                  // ELLPACK fill-in ratio
   std::vector< double > rho_SELL;  // Sliced ELLPACK fill-in ratio for different slice sizes
   std::vector< double > eta_SELL;  // Sliced ELLPACK efficiency for different slice sizes
   double H;                        // Entropy of the row length distribution
   double Gini;                     // Gini coefficient of the row length distribution
};
