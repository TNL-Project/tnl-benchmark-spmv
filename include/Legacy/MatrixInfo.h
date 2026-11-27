#pragma once

#include <TNL/Matrices/MatrixInfo.h>
#include "CSR.h"
#include "Ellpack.h"
#include "SlicedEllpack.h"
#include "ChunkedEllpack.h"
#include "BiEllpack.h"

namespace TNL::Matrices {

/////
// Legacy matrices
template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::BiEllpack< Real, Device, Index > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy BiEllpack";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRScalar > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Scalar";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRVector > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Vector";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Light";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight2 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Light2";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight3 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Light3";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight4 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Light4";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight5 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Light5";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight6 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Light6";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRAdaptive > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR Adaptive";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRMultiVector > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR MultiVector";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLightWithoutAtomic > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy CSR LightWithoutAtomic";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::ChunkedEllpack< Real, Device, Index > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy ChunkedEllpack";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack< Real, Device, Index > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy Ellpack";
   }
};

template< typename Real, typename Device, typename Index, int SliceSize >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index, SliceSize > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Legacy SlicedEllpack";
   }
};

}  // namespace TNL::Matrices
