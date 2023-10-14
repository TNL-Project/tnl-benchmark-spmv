#pragma once

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real, typename Index >
__cuda_callable__
MultidiagonalRow< Real, Index >::
MultidiagonalRow()
: values( 0 ),
  diagonals( 0 ),
  row( 0 ),
  columns( 0 ),
  maxRowLength( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
MultidiagonalRow< Real, Index >::
MultidiagonalRow( Real* values,
                           Index* diagonals,
                           const Index maxRowLength,
                           const Index row,
                           const Index columns,
                           const Index step )
: values( values ),
  diagonals( diagonals ),
  row( row ),
  columns( columns ),
  maxRowLength( maxRowLength ),
  step( step )
{
}

template< typename Real, typename Index >
__cuda_callable__
void
MultidiagonalRow< Real, Index >::
bind( Real* values,
      Index* diagonals,
      const Index maxRowLength,
      const Index row,
      const Index columns,
      const Index step )
{
   this->values = values;
   this->diagonals = diagonals;
   this->row = row;
   this->columns = columns;
   this->maxRowLength = maxRowLength;
   this->step = step;
}

template< typename Real, typename Index >
__cuda_callable__
void
MultidiagonalRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   TNL_ASSERT_TRUE( this->values, "" );
   TNL_ASSERT_GT( this->step, 0, "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->columns, "" );
   TNL_ASSERT_GE( elementIndex, 0, "" );
   TNL_ASSERT_LT( elementIndex, this->maxRowLength, "" );

   Index aux = elementIndex;
   while( row + this->diagonals[ aux ] < column ) aux++;
   TNL_ASSERT_EQ( row + this->diagonals[ aux ], column, "" );

   //printf( "Setting element %d column %d value %f \n", aux * this->step, column, value );
   this->values[ aux * this->step ] = value;
}

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
