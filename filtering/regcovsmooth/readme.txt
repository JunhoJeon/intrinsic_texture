                                                                     
                                                                     
                                                                     
                                             
The code and the algorithm are for non-commercial use only.

Paper : "Structure-Preserving Image Smoothing via Region Covariances"

Author: Levent Karacan, Erkut Erdem, Aykut Erdem 

        (karacan@cs.hacettepe.edu.tr, erkut@cs.hacettepe.edu.tr, aykut@cs.hacettepe.edu.tr)

Date  : 11/05/2013

Version : 1.0 

Copyright 2013, Hacettepe University, Turkey.



Notes:

- Model2 requires mtimesx,mmx,multiinv and Intel MKL linear algebra libraries.

  (found in ext directory)



  mtimesx library is used for multidimensional matrix multiplication, and requires Lapack and Blas libraries. 

  These libraries can be easily set by packet manager on linux systems

    

  % Linux Systems 

     matlabroot='MATLAB/R2013a'

     libmwblas.so and libmwlapack.so files may be under the glnxa64/ directory.

     lib_lapack = [matlabroot '/bin/glnxa64/lcc/libmwlapack.so'];

     lib_blas = [matlabroot '/bin/glnxa64/lcc/libmwblas.so'];

     mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',lib_blas) ;



  % Windows Systems

     Visual Studio 2010 C++ Compiler

     mex mtimesx.c



  mmx library is used for inverting a multidimensional matrix, and requires Intel MKL library  on your system.

  

  "build_mmx.m" under the mmx/mmx_package/ is used to compile mmx files. 



  multiinv library is also used for inverting a multidimensional matrix but slower than mmx function.



 