mex -I/usr/local/include/opencv -L/usr/local/lib -lopencv_core getConstraintsMatrix.cpp mexBase.cpp
mex -I/usr/local/include/opencv -L/usr/local/lib -lopencv_core getContinuousConstraintMatrix.cpp mexBase.cpp
mex -I/usr/local/include/opencv -Iann_x64/include -L/usr/local/lib -lopencv_core -lann getGridLLEMatrix.cpp mexBase.cpp LLE.cpp
mex -I/usr/local/include/opencv -Iann_x64/include -L/usr/local/lib -lopencv_core -lann getGridLLEMatrixNormal.cpp mexBase.cpp LLE.cpp
mex -I/usr/local/include/opencv -L/usr/local/lib -lopencv_core getNormalConstraintMatrix.cpp mexBase.cpp