#include "mexBase.h";

void pushSparseMatrix(CvSparseMat *tcm, char* matName)
{
	int nzCnt = 0;
	CvSparseMatIterator it;
	for(CvSparseNode *node = cvInitSparseMatIterator(tcm, &it); node != 0; node = cvGetNextSparseNode( &it))
	{
		nzCnt++;
	}
	double *mData = new double[3 * nzCnt];
	int ii=0;
	for(CvSparseNode *node = cvInitSparseMatIterator( tcm, &it );
		node != 0; node = cvGetNextSparseNode( &it )) 
	{
		int* idx = CV_NODE_IDX(tcm,node); 
		float val = ((float*)cvPtrND(tcm, idx))[0]; 

		mData[ii*3 + 0] = idx[0];
		mData[ii*3 + 1] = idx[1];
		mData[ii*3 + 2] = val;
		ii++;
	}
	mexPrintf("%s nzCnt: %d\n", matName, ii);
	mxArray *mArray = mxCreateDoubleMatrix(3, nzCnt, mxREAL);
	memcpy((void*)mxGetPr(mArray), (void*)mData, sizeof(double)*3*nzCnt);
	mexPutVariable("caller", "consMat", mArray);
	int pixCnt = tcm->size[0];
	char buffer[512];
	//sprintf_s(buffer,"%s = sparse(consMat(1,:)+1,consMat(2,:)+1,consMat(3,:),%d,%d);", matName, pixCnt,pixCnt);
    sprintf(buffer,"%s = sparse(consMat(1,:)+1,consMat(2,:)+1,consMat(3,:),%d,%d);", matName, pixCnt,pixCnt);
	mexEvalString(buffer);
	delete [] mData;
	mxDestroyArray(mArray);	
}

