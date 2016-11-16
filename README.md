# Intrinsic Images using Texture Filtering
Unoptimized implementation of <b>"Intrinsic Image Decomposition Using Structure-Texture Separation and Surface Normals"</b> in Matlab and C++ MEX.

## How to use it
You can try demo.m using included RGB & depth image.<br>
Included input RGB and depth images are extracted from [NYU Depth V2 dataset](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).<br>
Algorithm requires texture-removed RGB image. Authors' code of [regcovsmoothing](http://web.cs.hacettepe.edu.tr/~karacan/projects/regcovsmoothing/) and [RGF](http://www.cse.cuhk.edu.hk/leojia/projects/rollguidance/) are included.<br>
You can use any structure-texture separation algorithm instead of it.

## Hardware/software requirements
1. Original code is tested on Matlab 2015a 64bit, Windows 7.
2. For other platform (32bit Windows, Linux, MacOS), compile the C++ mex source codes in 'mex' directory.
3. To compile mex source codes, corresponding <b>ann</b> and <b>opencv</b> library is required (please refer the compile.m).

## Contributors
[Junho Jeon](http://junhojeon.github.io/) (zwitterion27@postech.ac.kr)

## Citation
Cite our papers if you find this software useful.<br>
1. Junho Jeon, Sunghyun Cho, Xin Tong, Seungyong Lee, "[Intrinsic Image Decomposition using Structure-Texture Separation and Surface Normals](http://cg.postech.ac.kr/research/intrinsic/)", European Conference on Computer Vision (ECCV 2014), September 2014. 
