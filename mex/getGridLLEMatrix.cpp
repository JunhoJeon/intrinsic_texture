#include "mexBase.h"
#include "LLE.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int feature_dim = mxGetPr(prhs[3])[0];
	int img_h = mxGetM(prhs[0]);
	int img_w = mxGetN(prhs[0])/feature_dim;
	int N = img_h*img_w;
	int K = mxGetPr(prhs[2])[0];
	int dims[2] = {N, N};
	CvSparseMat* affinityMatrix = cvCreateSparseMat(2, dims, CV_32FC1);
	double *nMap = mxGetPr(prhs[0]); // normal Map
	double *vMap = mxGetPr(prhs[1]); // variance Map
	int g_size = mxGetPr(prhs[4])[0];

	int ngrid_w = ceil(img_w / (float)g_size);
	int ngrid_h = ceil(img_h / (float)g_size);
	int Ngrid = ngrid_w * ngrid_h;
	int *x_pos = new int[Ngrid];
	int *y_pos = new int[Ngrid];
	cv::Mat1f X(Ngrid, feature_dim);

	for(int j=0, n=0;j<img_h;j+=g_size) { // grid iteration
		for(int i=0;i<img_w;i+=g_size) {
			double vmin = 99999;
			for(int gj=0;gj<g_size && j+gj < img_h;gj++) { // y pos in grid
				for(int gi=0;gi<g_size && i+gi < img_w;gi++) { // x pos
					double var = vMap[(i+gi)*img_h + (j+gj)];
					if(var < vmin) {
						vmin = var;
						x_pos[n] = (i+gi);
						y_pos[n] = (j+gj);
					}
				}
			}
			for(int k=0;k<feature_dim;k++)
				X(n, k) = nMap[k*img_h*img_w+x_pos[n]*img_h+y_pos[n]];
			n++;
		}
	}

	cv::Mat1f W(Ngrid, K);
	cv::Mat1i neighbors(Ngrid, K);
	LLE3(X, W, neighbors, Ngrid, feature_dim, K);
	
	//plhs[0] = mxCreateDoubleMatrix(img_w, img_h, mxREAL);
	plhs[0] = mxCreateDoubleMatrix(Ngrid, K+1, mxREAL);
	double *neighborPixels = mxGetPr(plhs[0]);
	for(int n=0;n<Ngrid;n++) {
		int xp = x_pos[n];
		int yp = y_pos[n];
		//int p = yp * img_w + xp;
		int p = xp * img_h + yp;
		neighborPixels[n] = p + 1;
		for(int k=0;k<K;k++) {
			if(W(n, k) != 0) {
				int nIdx = neighbors(n, k);
				if(nIdx >= 0) {
					int xq = x_pos[nIdx];
					int yq = y_pos[nIdx];
					//int q = yq * img_w + xq;
					int q = xq * img_h + yq;
					((float*)cvPtr2D(affinityMatrix, p, q))[0] = W(n, k);
					neighborPixels[(k+1)*Ngrid + n] = q + 1;
				}
			}
		}
	}
    pushSparseMatrix(affinityMatrix, "LLEGRID");
	cvReleaseSparseMat(&affinityMatrix);
	
	delete [] x_pos;
	delete [] y_pos;
}