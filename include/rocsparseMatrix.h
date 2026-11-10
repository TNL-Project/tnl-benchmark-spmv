#pragma once

// SpMV of rocSPARSE called directly through its generic API (rocsparse_v2_spmv).
//
// hipSPARSE exposes only the default algorithm and two CSR algorithms of
// rocSPARSE, so to benchmark all CSR algorithms of the AMD vendor library,
// similarly as all algorithms of cuSPARSE are benchmarked on NVIDIA GPUs,
// rocSPARSE is called directly. The analysis stage is executed once in init(),
// i.e. it is not included in the measured time, which corresponds to
// cusparseSpMV_preprocess in cusparseCSRMatrix.h.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <TNL/Containers/Array.h>
#include <TNL/Devices/Hip.h>

#if defined( __HIP__ ) && defined( HAVE_ROCSPARSE )
   #include <rocsparse/rocsparse.h>

   // rocsparse-version.h defines only the components of the version,
   // the same encoding as returned by rocsparse_get_version is used here.
   #define TNL_ROCSPARSE_VERSION ( ROCSPARSE_VERSION_MAJOR * 100000 + ROCSPARSE_VERSION_MINOR * 100 + ROCSPARSE_VERSION_PATCH )
   // nnzsplit is available since rocSPARSE 4.1 (ROCm 7.1), Sliced ELL since 4.2 (ROCm 7.2)
   #if TNL_ROCSPARSE_VERSION >= 400100
      #define TNL_ROCSPARSE_HAS_NNZSPLIT
   #endif
   #if TNL_ROCSPARSE_VERSION >= 400200
      #define TNL_ROCSPARSE_HAS_SELL
   #endif

   #define CHECK_ROCSPARSE( func )                                                                                 \
      {                                                                                                            \
         rocsparse_status status = ( func );                                                                       \
         if( status != rocsparse_status_success ) {                                                                \
            throw std::runtime_error( std::string( "rocSPARSE call failed at " ) + __FILE__ + ":"                  \
                                      + std::to_string( __LINE__ ) + " with status " + std::to_string( status ) ); \
         }                                                                                                         \
      }

namespace TNL {

// An SpMV algorithm of rocSPARSE to benchmark and its name for the "launch cfg." column.
struct RocsparseAlgVariant
{
   std::string name;
   rocsparse_spmv_alg alg;
};

// All CSR algorithms of rocSPARSE available in the installed version.
inline std::vector< RocsparseAlgVariant >
getRocsparseCSRAlgorithms()
{
   return {
      { "Default", rocsparse_spmv_alg_default },
      { "CSR adaptive", rocsparse_spmv_alg_csr_adaptive },
      { "CSR rowsplit", rocsparse_spmv_alg_csr_rowsplit },
      { "CSR LRB", rocsparse_spmv_alg_csr_lrb },
   #ifdef TNL_ROCSPARSE_HAS_NNZSPLIT
      { "CSR nnzsplit", rocsparse_spmv_alg_csr_nnzsplit },
   #endif
   };
}

   #ifdef TNL_ROCSPARSE_HAS_SELL
// All Sliced ELL algorithms of rocSPARSE.
inline std::vector< RocsparseAlgVariant >
getRocsparseSlicedEllAlgorithms()
{
   return {
      { "Default", rocsparse_spmv_alg_default },
      { "SELL", rocsparse_spmv_alg_sell },
   };
}
   #endif

template< typename Real >
constexpr rocsparse_datatype
getRocsparseDatatype()
{
   static_assert( std::is_same_v< Real, double > || std::is_same_v< Real, float >,
                  "rocSPARSE SpMV is benchmarked only in single and double precision." );
   if constexpr( std::is_same_v< Real, double > )
      return rocsparse_datatype_f64_r;
   else
      return rocsparse_datatype_f32_r;
}

// Common part of the CSR and Sliced ELL wrappers: the SpMV descriptor,
// the vector descriptors, the buffer and the analysis stage.
template< typename Real >
class RocsparseSpMVBase
{
public:
   RocsparseSpMVBase() = default;
   RocsparseSpMVBase( const RocsparseSpMVBase& ) = delete;
   RocsparseSpMVBase&
   operator=( const RocsparseSpMVBase& ) = delete;

   ~RocsparseSpMVBase()
   {
      // Errors cannot be reported from the destructor, the return values are ignored.
      if( spmvDescr != nullptr )
         rocsparse_destroy_spmv_descr( spmvDescr );
      if( matA != nullptr )
         rocsparse_destroy_spmat_descr( matA );
      if( vecX != nullptr )
         rocsparse_destroy_dnvec_descr( vecX );
      if( vecY != nullptr )
         rocsparse_destroy_dnvec_descr( vecY );
   }

   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector, OutVector& outVector ) const
   {
      if( inVector.getData() != xData || outVector.getData() != yData )
         throw std::runtime_error( "rocSPARSE SpMV: the vectors differ from those passed to init()." );
      const Real alpha = 1;
      const Real beta = 0;
      CHECK_ROCSPARSE( rocsparse_v2_spmv( handle,
                                          spmvDescr,
                                          &alpha,
                                          matA,
                                          vecX,
                                          &beta,
                                          vecY,
                                          rocsparse_v2_spmv_stage_compute,
                                          computeBuffer.getSize(),
                                          (void*) computeBuffer.getData(),
                                          nullptr ) );
   }

protected:
   // Creates the vector and SpMV descriptors and runs the analysis stage.
   // `matA` must be created by the derived class before calling this.
   template< typename InVector, typename OutVector >
   void
   setup( rocsparse_handle handle,
          rocsparse_spmv_alg alg,
          std::int64_t rows,
          std::int64_t columns,
          const InVector& inVector,
          OutVector& outVector )
   {
      this->handle = handle;
      this->xData = inVector.getData();
      this->yData = outVector.getData();
      constexpr rocsparse_datatype datatype = getRocsparseDatatype< Real >();
      CHECK_ROCSPARSE( rocsparse_create_dnvec_descr( &vecX, columns, (void*) inVector.getData(), datatype ) );
      CHECK_ROCSPARSE( rocsparse_create_dnvec_descr( &vecY, rows, (void*) outVector.getData(), datatype ) );

      CHECK_ROCSPARSE( rocsparse_create_spmv_descr( &spmvDescr ) );
      const rocsparse_operation operation = rocsparse_operation_none;
      CHECK_ROCSPARSE( rocsparse_spmv_set_input( handle, spmvDescr, rocsparse_spmv_input_alg, &alg, sizeof( alg ), nullptr ) );
      CHECK_ROCSPARSE( rocsparse_spmv_set_input(
         handle, spmvDescr, rocsparse_spmv_input_operation, &operation, sizeof( operation ), nullptr ) );
      CHECK_ROCSPARSE( rocsparse_spmv_set_input(
         handle, spmvDescr, rocsparse_spmv_input_scalar_datatype, &datatype, sizeof( datatype ), nullptr ) );
      CHECK_ROCSPARSE( rocsparse_spmv_set_input(
         handle, spmvDescr, rocsparse_spmv_input_compute_datatype, &datatype, sizeof( datatype ), nullptr ) );

      // The analysis stage is mandatory and it is executed only once.
      std::size_t bufferSize = 0;
      CHECK_ROCSPARSE( rocsparse_v2_spmv_buffer_size(
         handle, spmvDescr, matA, vecX, vecY, rocsparse_v2_spmv_stage_analysis, &bufferSize, nullptr ) );
      Containers::Array< std::byte, Devices::Hip > analysisBuffer( bufferSize );
      const Real alpha = 1;
      const Real beta = 0;
      CHECK_ROCSPARSE( rocsparse_v2_spmv( handle,
                                          spmvDescr,
                                          &alpha,
                                          matA,
                                          vecX,
                                          &beta,
                                          vecY,
                                          rocsparse_v2_spmv_stage_analysis,
                                          analysisBuffer.getSize(),
                                          (void*) analysisBuffer.getData(),
                                          nullptr ) );

      bufferSize = 0;
      CHECK_ROCSPARSE( rocsparse_v2_spmv_buffer_size(
         handle, spmvDescr, matA, vecX, vecY, rocsparse_v2_spmv_stage_compute, &bufferSize, nullptr ) );
      computeBuffer.setSize( bufferSize );
   }

   rocsparse_handle handle = nullptr;
   rocsparse_spmat_descr matA = nullptr;
   rocsparse_dnvec_descr vecX = nullptr;
   rocsparse_dnvec_descr vecY = nullptr;
   rocsparse_spmv_descr spmvDescr = nullptr;
   Containers::Array< std::byte, Devices::Hip > computeBuffer;
   const void* xData = nullptr;
   const void* yData = nullptr;
};

// SpMV of rocSPARSE with a matrix in the CSR format. The data of the TNL CSR
// matrix are passed to rocSPARSE directly, without a conversion.
template< typename Real >
class RocsparseCSR : public RocsparseSpMVBase< Real >
{
public:
   template< typename Matrix, typename InVector, typename OutVector >
   void
   init( Matrix& matrix, const InVector& inVector, OutVector& outVector, rocsparse_handle handle, rocsparse_spmv_alg alg )
   {
      static_assert( std::is_same_v< typename Matrix::RealType, Real >, "Wrong real type of the matrix." );
      static_assert( std::is_same_v< typename Matrix::IndexType, int >, "rocSPARSE is benchmarked with 32-bit indices." );
      CHECK_ROCSPARSE( rocsparse_create_csr_descr( &this->matA,
                                                   matrix.getRows(),
                                                   matrix.getColumns(),
                                                   matrix.getValues().getSize(),
                                                   (void*) matrix.getSegments().getOffsets().getData(),
                                                   (void*) matrix.getColumnIndexes().getData(),
                                                   (void*) matrix.getValues().getData(),
                                                   rocsparse_indextype_i32,
                                                   rocsparse_indextype_i32,
                                                   rocsparse_index_base_zero,
                                                   getRocsparseDatatype< Real >() ) );
      this->setup( handle, alg, matrix.getRows(), matrix.getColumns(), inVector, outVector );
   }
};

   #ifdef TNL_ROCSPARSE_HAS_SELL
// SpMV of rocSPARSE with a matrix in the Sliced ELL format. Like cuSPARSE, it
// is assumed to match TNL's column-major SlicedEllpack (padding elements with
// the column index -1), see cusparseSlicedEllMatrix.h. The correctness of this
// assumption is checked by the comparison with the reference result.
template< typename Real >
class RocsparseSlicedEll : public RocsparseSpMVBase< Real >
{
public:
   template< typename Matrix, typename InVector, typename OutVector >
   void
   init( Matrix& matrix, const InVector& inVector, OutVector& outVector, rocsparse_handle handle, rocsparse_spmv_alg alg )
   {
      static_assert( std::is_same_v< typename Matrix::RealType, Real >, "Wrong real type of the matrix." );
      static_assert( std::is_same_v< typename Matrix::IndexType, int >, "rocSPARSE is benchmarked with 32-bit indices." );
      CHECK_ROCSPARSE( rocsparse_create_sell_descr( &this->matA,
                                                    matrix.getRows(),
                                                    matrix.getColumns(),
                                                    matrix.getNonzeroElementsCount(),
                                                    matrix.getSegments().getSliceSize(),
                                                    matrix.getSegments().getStorageSize(),
                                                    (void*) matrix.getSegments().getSliceOffsetsView().getData(),
                                                    (void*) matrix.getColumnIndexes().getData(),
                                                    (void*) matrix.getValues().getData(),
                                                    rocsparse_indextype_i32,
                                                    rocsparse_indextype_i32,
                                                    rocsparse_index_base_zero,
                                                    getRocsparseDatatype< Real >() ) );
      this->setup( handle, alg, matrix.getRows(), matrix.getColumns(), inVector, outVector );
   }
};
   #endif

}  // namespace TNL

#endif  // defined( __HIP__ ) && defined( HAVE_ROCSPARSE )
