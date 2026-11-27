#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/SparseMatrixBase.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#ifdef __CUDACC__
   #include <cusparse.h>
#endif

#include <cstddef>

// cuSPARSE's Sliced ELLPACK matches TNL's SlicedEllpack with column-major element
// organization: slices of `sliceSize` rows, padded to the widest row in the slice,
// stored column-major within a slice, and `-1` used as the column index of padding
// elements in both formats. That means TNL's segments data can be handed to cuSPARSE
// directly, with no format conversion.
#ifndef CHECK_CUSPARSE
   #ifdef __CUDACC__
      #define CHECK_CUSPARSE( func )                                                                                          \
         {                                                                                                                    \
            cusparseStatus_t status = ( func );                                                                               \
            if( status != CUSPARSE_STATUS_SUCCESS ) {                                                                         \
               std::cerr << "CUSPARSE API failed at line " << __LINE__ << " with error: "                                    \
                         << cusparseGetErrorString( status ) << " " << status << std::endl;                                   \
               throw std::runtime_error( cusparseGetErrorString( status ) );                                                  \
            }                                                                                                                 \
         }
   #endif
#endif

namespace TNL {

template< typename Real >
class CusparseSlicedEllBase
{
public:
   using RealType = Real;
   using DeviceType = TNL::Devices::Cuda;
   using MatrixType = TNL::Matrices::SparseMatrixBase< Real,
                                                       TNL::Devices::Cuda,
                                                       int,
                                                       Matrices::GeneralMatrix,
                                                       Algorithms::Segments::SlicedEllpackView< TNL::Devices::Cuda, int >,
                                                       Real >;

   CusparseSlicedEllBase() = default;

#ifdef __CUDACC__
   void
   init( const MatrixType& matrix, cusparseHandle_t* cusparseHandle )
   {
      this->cusparseHandle = cusparseHandle;
      this->matrix = &matrix;
   }
#endif

   int
   getRows() const
   {
      return matrix->getRows();
   }

   int
   getColumns() const
   {
      return matrix->getColumns();
   }

   int
   getNumberOfMatrixElements() const
   {
      return matrix->getAllocatedElementsCount();
   }

   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector, OutVector& outVector ) const
   {
      throw std::runtime_error( "Unsupported Real type for cusparse." );
   }

protected:
   const MatrixType* matrix = nullptr;
#ifdef __CUDACC__
   cusparseHandle_t* cusparseHandle;
   cusparseSpMatDescr_t matA;
   TNL::Containers::Array< std::byte, TNL::Devices::Cuda > buffer;
   cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT;
#endif
};

template< typename Real >
class CusparseSlicedEll
{};

template<>
class CusparseSlicedEll< double > : public CusparseSlicedEllBase< double >
{
public:
   CusparseSlicedEll() = default;
   CusparseSlicedEll( const CusparseSlicedEll& ) = delete;
   CusparseSlicedEll&
   operator=( const CusparseSlicedEll& ) = delete;

#ifdef __CUDACC__
   ~CusparseSlicedEll()
   {
      if( this->matrix != nullptr ) {
         cusparseDestroySpMat( this->matA );
         cusparseDestroyDnVec( this->vecX );
         cusparseDestroyDnVec( this->vecY );
      }
   }

   template< typename InVector, typename OutVector >
   void
   init( MatrixType& matrix,
         const InVector& inVector,
         OutVector& outVector,
         cusparseHandle_t* cusparseHandle,
         cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT )
   {
      CusparseSlicedEllBase< double >::init( matrix, cusparseHandle );
      this->algorithm = alg;
      double alpha = 1.0;
      double beta = 0.0;
      CHECK_CUSPARSE( cusparseCreateSlicedEll( &this->matA,                                    // cusparseSpMatDescr_t* spMatDescr,
                                               matrix.getRows(),                                // int64_t              rows,
                                               matrix.getColumns(),                             // int64_t              cols,
                                               matrix.getNonzeroElementsCount(),                // int64_t              nnz,
                                               matrix.getSegments().getStorageSize(),            // int64_t sellValuesSize,
                                               matrix.getSegments().getSliceSize(),              // int64_t sliceSize,
                                               matrix.getSegments().getSliceOffsetsView().getData(),  // void* sellSliceOffsets,
                                               matrix.getColumnIndexes().getData(),              // void*  sellColInd,
                                               matrix.getValues().getData(),                     // void*  sellValues,
                                               CUSPARSE_INDEX_32I,        // cusparseIndexType_t   sellSliceOffsetsType,
                                               CUSPARSE_INDEX_32I,        // cusparseIndexType_t   sellColIndType,
                                               CUSPARSE_INDEX_BASE_ZERO,  // cusparseIndexBase_t   idxBase,
                                               CUDA_R_64F ) );            // cudaDataType          valueType
      CHECK_CUSPARSE( cusparseCreateDnVec( &this->vecX, matrix.getColumns(), (void*) inVector.getData(), CUDA_R_64F ) );
      CHECK_CUSPARSE( cusparseCreateDnVec( &this->vecY, matrix.getRows(), (void*) outVector.getData(), CUDA_R_64F ) );
      size_t buffer_size;
      CHECK_CUSPARSE( cusparseSpMV_bufferSize( *( this->cusparseHandle ),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               this->matA,
                                               this->vecX,
                                               &beta,
                                               this->vecY,
                                               CUDA_R_64F,
                                               this->algorithm,
                                               &buffer_size ) );
      this->buffer.setSize( buffer_size );
      // One-time analysis that lets cusparseSpMV() pick a faster kernel/load-balancing
      // strategy on every subsequent call with the same sparsity pattern.
      CHECK_CUSPARSE( cusparseSpMV_preprocess( *( this->cusparseHandle ),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               this->matA,
                                               this->vecX,
                                               &beta,
                                               this->vecY,
                                               CUDA_R_64F,
                                               this->algorithm,
                                               (void*) this->buffer.getData() ) );
   }
#endif

   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector, OutVector& outVector ) const
   {
      TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef __CUDACC__
      double alpha = 1.0;
      double beta = 0.0;
      CHECK_CUSPARSE( cusparseSpMV( *( this->cusparseHandle ),
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    this->matA,
                                    this->vecX,
                                    &beta,
                                    this->vecY,
                                    CUDA_R_64F,
                                    this->algorithm,
                                    (void*) this->buffer.getData() ) );
#endif
   }

protected:
#ifdef __CUDACC__
   cusparseDnVecDescr_t vecX, vecY;
#endif
};

template<>
class CusparseSlicedEll< float > : public CusparseSlicedEllBase< float >
{
public:
   CusparseSlicedEll() = default;
   CusparseSlicedEll( const CusparseSlicedEll& ) = delete;
   CusparseSlicedEll&
   operator=( const CusparseSlicedEll& ) = delete;

#ifdef __CUDACC__
   ~CusparseSlicedEll()
   {
      if( this->matrix != nullptr ) {
         cusparseDestroySpMat( this->matA );
         cusparseDestroyDnVec( this->vecX );
         cusparseDestroyDnVec( this->vecY );
      }
   }

   template< typename InVector, typename OutVector >
   void
   init( MatrixType& matrix,
         const InVector& inVector,
         OutVector& outVector,
         cusparseHandle_t* cusparseHandle,
         cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT )
   {
      CusparseSlicedEllBase< float >::init( matrix, cusparseHandle );
      this->algorithm = alg;
      float alpha = 1.0;
      float beta = 0.0;
      CHECK_CUSPARSE( cusparseCreateSlicedEll( &this->matA,                                    // cusparseSpMatDescr_t* spMatDescr,
                                               matrix.getRows(),                                // int64_t              rows,
                                               matrix.getColumns(),                             // int64_t              cols,
                                               matrix.getNonzeroElementsCount(),                // int64_t              nnz,
                                               matrix.getSegments().getStorageSize(),            // int64_t sellValuesSize,
                                               matrix.getSegments().getSliceSize(),              // int64_t sliceSize,
                                               matrix.getSegments().getSliceOffsetsView().getData(),  // void* sellSliceOffsets,
                                               matrix.getColumnIndexes().getData(),              // void*  sellColInd,
                                               matrix.getValues().getData(),                     // void*  sellValues,
                                               CUSPARSE_INDEX_32I,        // cusparseIndexType_t   sellSliceOffsetsType,
                                               CUSPARSE_INDEX_32I,        // cusparseIndexType_t   sellColIndType,
                                               CUSPARSE_INDEX_BASE_ZERO,  // cusparseIndexBase_t   idxBase,
                                               CUDA_R_32F ) );            // cudaDataType          valueType
      CHECK_CUSPARSE( cusparseCreateDnVec( &this->vecX, matrix.getColumns(), (void*) inVector.getData(), CUDA_R_32F ) );
      CHECK_CUSPARSE( cusparseCreateDnVec( &this->vecY, matrix.getRows(), (void*) outVector.getData(), CUDA_R_32F ) );
      size_t buffer_size;
      CHECK_CUSPARSE( cusparseSpMV_bufferSize( *( this->cusparseHandle ),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               this->matA,
                                               this->vecX,
                                               &beta,
                                               this->vecY,
                                               CUDA_R_32F,
                                               this->algorithm,
                                               &buffer_size ) );
      this->buffer.setSize( buffer_size );
      // One-time analysis that lets cusparseSpMV() pick a faster kernel/load-balancing
      // strategy on every subsequent call with the same sparsity pattern.
      CHECK_CUSPARSE( cusparseSpMV_preprocess( *( this->cusparseHandle ),
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               this->matA,
                                               this->vecX,
                                               &beta,
                                               this->vecY,
                                               CUDA_R_32F,
                                               this->algorithm,
                                               (void*) this->buffer.getData() ) );
   }
#endif

   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector, OutVector& outVector ) const
   {
      TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef __CUDACC__
      float alpha = 1.0;
      float beta = 0.0;
      CHECK_CUSPARSE( cusparseSpMV( *( this->cusparseHandle ),
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    this->matA,
                                    this->vecX,
                                    &beta,
                                    this->vecY,
                                    CUDA_R_32F,
                                    this->algorithm,
                                    (void*) this->buffer.getData() ) );
#endif
   }

protected:
#ifdef __CUDACC__
   cusparseDnVecDescr_t vecX, vecY;
#endif
};

}  // namespace TNL
