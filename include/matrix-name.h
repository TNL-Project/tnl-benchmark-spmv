#pragma once

#include <filesystem>
#include <string>

#include <TNL/String.h>

namespace TNL::Benchmarks::SpMV {

// Ports scripts/matrix-name-from-path.sh (now removed): derives a system-independent matrix name
// from its path relative to `basePath`, e.g.
// ./mtx_matrices/suitesparse-collection-website.herokuapp.com/MM/ANSYS/Delor64K/Delor64K.mtx
// becomes ANSYS/Delor64K.mtx.
inline std::string
matrixNameFromPath( const std::string& path, const std::string& basePath )
{
   std::string name = path;

   // strip the base path prefix, with or without its own trailing slash
   std::string base = basePath;
   while( ! base.empty() && base.back() == '/' )
      base.pop_back();
   const std::string prefix = base + "/";
   if( name.rfind( prefix, 0 ) == 0 )
      name = name.substr( prefix.size() );
   if( ! name.empty() && name.front() == '/' )
      name = name.substr( 1 );

   // strip the prefix of a known matrix collection: keep only what follows the LAST
   // occurrence of one of these markers
   for( const std::string& marker : { std::string( "/MM/" ), std::string( "/MatrixMarket2/" ) } ) {
      const auto pos = name.rfind( marker );
      if( pos != std::string::npos ) {
         name = name.substr( pos + marker.size() );
         break;
      }
   }

   // SuiteSparse archives unpack into a directory named after the matrix, so
   // Group/Name/Name.mtx is collapsed into Group/Name.mtx
   const std::filesystem::path p( name );
   const std::string file = p.filename().string();
   std::string stem = file;
   const std::string mtxExt = ".mtx";
   if( stem.size() >= mtxExt.size() && stem.compare( stem.size() - mtxExt.size(), mtxExt.size(), mtxExt ) == 0 )
      stem.erase( stem.size() - mtxExt.size() );
   const std::filesystem::path dir = p.parent_path();
   if( dir.filename().string() == stem ) {
      name = ( dir.parent_path() / file ).string();
      if( name.rfind( "./", 0 ) == 0 )
         name = name.substr( 2 );
   }

   return name;
}

// Builds "<dir>/<matrixName>[.transposed].<extension>", creating the parent directories. Shared by
// the per-matrix "filling" files (matrix-info.h) and the per-matrix progress files
// (benchmark-progress.h), which use the same naming convention.
inline std::filesystem::path
matrixFilePath( const std::string& dir, const String& matrixName, bool transposed, const std::string& extension )
{
   const std::filesystem::path path =
      std::filesystem::path( dir )
      / ( std::string( matrixName.getString() ) + ( transposed ? ".transposed" : "" ) + "." + extension );
   std::filesystem::create_directories( path.parent_path() );
   return path;
}

}  // namespace TNL::Benchmarks::SpMV
