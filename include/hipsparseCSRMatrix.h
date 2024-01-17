#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/SparseMatrixBase.h>
#ifdef __HIP__
#include <hipsparse/hipsparse.h>
#endif

#include <cstddef>

namespace TNL {

template< typename Real >
class HipsparseCSRBase
{
   public:
      using RealType = Real;
      using DeviceType = TNL::Devices::Hip;
      using MatrixType = TNL::Matrices::SparseMatrixBase< Real,
                                                          TNL::Devices::Hip,
                                                          int,
                                                          Matrices::GeneralMatrix,
                                                          Algorithms::Segments::CSRView< TNL::Devices::Cuda, int >,
                                                          Real >;

      HipsparseCSRBase() = default;

#ifdef __HIP__
      void init( const MatrixType& matrix,
                 hipsparseHandle_t* hipsparseHandle )
      {
         this->hipsparseHandle = hipsparseHandle;
         this->matrix = &matrix;
#if CUDART_VERSION < 11000
         hipsparseCreateMatDescr( & this->matrixDescriptor );
#endif
      };
#endif

      int getRows() const
      {
         return matrix->getRows();
      }

      int getColumns() const
      {
         return matrix->getColumns();
      }

      int getNumberOfMatrixElements() const
      {
         return matrix->getAllocatedElementsCount();
      }


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         throw std::runtime_error( "Unsupported Real type for cusparse." );
      }

   protected:

      const MatrixType* matrix = nullptr;
#ifdef __HIP__
      hipsparseHandle_t* hipsparseHandle;
#if CUDART_VERSION < 11000
      hipsparseMatDescr_t matrixDescriptor;
#else
      hipsparseSpMatDescr_t matA;
      TNL::Containers::Array< std::byte, TNL::Devices::Hip > buffer;
#endif
#endif

};

template< typename Real >
class HipsparseCSR
{};

template<>
class HipsparseCSR< double > : public HipsparseCSRBase< double >
{
   public:

#ifdef __HIP__
      template< typename InVector,
                typename OutVector >
      void init( MatrixType& matrix,
                 const InVector& inVector,
                 OutVector& outVector,
                 hipsparseHandle_t* hipsparseHandle )
      {
         HipsparseCSRBase< double >::init( matrix, hipsparseHandle );
         hipsparseCreateMatDescr( & this->matrixDescriptor );
      };
#endif


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef __HIP__
	      double a = 1.0;
         double b = 0.0;
         double* alpha = &a;
         double* beta = &b;
         hipsparseDcsrmv( *( this->hipsparseHandle ),
                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->getValues().getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->getValues().getData(),
                         this->matrix->getSegments().getOffsets().getData(),
                         this->matrix->getColumnIndexes().getData(),
                         inVector.getData(),
                         beta,
                         outVector.getData() );
#endif
      }
   protected:
};

template<>
class HipsparseCSR< float > : public HipsparseCSRBase< float >
{
   public:

#ifdef __HIP__
      template< typename InVector,
                typename OutVector >
      void init( MatrixType& matrix,
                 const InVector& inVector,
                 OutVector& outVector,
                 hipsparseHandle_t* hipsparseHandle )
      {
         HipsparseCSRBase< float >::init( matrix, hipsparseHandle );
         hipsparseCreateMatDescr( & this->matrixDescriptor );
      };
#endif


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef __HIP__
         float a = 1.0;
         float b = 0.0;
         float* alpha = &a;
         float* beta = &b;
         hipsparseScsrmv( *( this->hipsparseHandle ),
                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->getValues().getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->getValues().getData(),
                         this->matrix->getSegments().getOffsets().getData(),
                         this->matrix->getColumnIndexes().getData(),
                         inVector.getData(),
                         beta,
                         outVector.getData() );
#endif
      }
   protected:
};

} // namespace TNL
