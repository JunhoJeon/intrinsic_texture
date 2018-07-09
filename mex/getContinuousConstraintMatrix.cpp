#include "mexBase.h"
#define MIN(X, Y) (X > Y ? Y : X)
#define MAX(X, Y) (X > Y ? X : Y)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int h = mxGetM(prhs[0]);
	int w = mxGetN(prhs[0])/3;
	int i, j, k;
	int nx[] = {0, 0, 1, -1, -1, 1, 1, -1};
	int ny[] = {1, -1, 0, 0, -1, 1, -1, 1};
	double cp[3], cq[3];
	double *chroma = mxGetPr(prhs[0]);
	double dist;
	plhs[0] = mxCreateDoubleMatrix(h*w, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(h, w, mxREAL);
	int dims[2] = {h*w, h*w};
	CvSparseMat* m_refConsMat = cvCreateSparseMat(2, dims, CV_32FC1);
	float sig_c= mxGetPr(prhs[1])[0];
	float sig_i= mxGetPr(prhs[2])[0];
	double *image = mxGetPr(prhs[3]);
	double ip[3], iq[3];
	double lp, lq;

	for(j=0;j<h;j++)
	{
		for(i=0;i<w;i++)
		{
			int p = i*h+j;
			cp[0] = chroma[i*h+j];
			cp[1] = chroma[h*w+i*h+j];
			cp[2] = chroma[2*h*w+i*h+j];
			ip[0] = image[i*h+j];
			ip[1] = image[h*w+i*h+j];
			ip[2] = image[2*h*w+i*h+j];
			lp = log(MAX(sqrt(ip[0]*ip[0] + ip[1]*ip[1] + ip[2]*ip[2]), 0.0001));
			for(k=0;k<8;k++)
			{
				int qi = i + nx[k];
				int qj = j + ny[k];
				int q = qi*h+qj;
				if(qi < 0 || qj < 0 || qi >= w || qj >= h)
				{
					continue;
				}
				cq[0] = chroma[qi*h+qj];
				cq[1] = chroma[h*w+qi*h+qj];
				cq[2] = chroma[2*h*w+qi*h+qj];
				iq[0] = image[qi*h+qj];
				iq[1] = image[h*w+qi*h+qj];
				iq[2] = image[2*h*w+qi*h+qj];
				lq = log(MAX(sqrt(iq[0]*iq[0] + iq[1]*iq[1] + iq[2]*iq[2]), 0.0001));

				dist = 2.0 * (1.0 - (cp[0]*cq[0]+cp[1]*cq[1]+cp[2]*cq[2]));	
				float weight = (1 + exp(-exp(lp) * exp(lp) / (sig_i*sig_i) - exp(lq)*exp(lq) / (sig_i*sig_i)));


				weight = weight * (exp(-dist*dist/(sig_c*sig_c)));
				if(k == 2)
					mxGetPr(plhs[1])[p] = weight;

				if(std::isnan(weight)) weight = 0;
				((float*)cvPtr2D(m_refConsMat, p, p))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, q, q))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, p, q))[0] += -weight;
				((float*)cvPtr2D(m_refConsMat, q, p))[0] += -weight;

				float dI = lp - lq;
				mxGetPr(plhs[0])[p] += weight * dI;
				mxGetPr(plhs[0])[q] -= weight * dI;
			}
		}
	}
	pushSparseMatrix(m_refConsMat, "WRC");

	cvReleaseSparseMat(&m_refConsMat);
}