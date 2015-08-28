#include "mexBase.h"

#define MIN(X, Y) (X > Y ? Y : X)
#define MAX(X, Y) (X > Y ? X : Y)
// [consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int h = mxGetM(prhs[0]);
	int w = mxGetN(prhs[0])/3;
	int i, j, k;
	int nx[] = {0, 0, 1, -1};
	int ny[] = {1, -1, 0, 0};
	double cp[3], cq[3], np[3], nq[3];
	double *chroma = mxGetPr(prhs[0]);
	double dist;
	plhs[0] = mxCreateDoubleMatrix(h*w, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(h, w, mxREAL);
	int dims[2] = {h*w, h*w};
	CvSparseMat* m_refConsMat = cvCreateSparseMat(2, dims, CV_32FC1);
	CvSparseMat* m_shadingConsMat = cvCreateSparseMat(2, dims, CV_32FC1);
	double *wr = mxGetPr(prhs[1]);
	float ws = mxGetPr(prhs[2])[0];
	double thres = mxGetPr(prhs[3])[0];
	int useNormal = mxGetPr(prhs[4])[0];
	double *normal;
	double nthres;
	double *image;
	double ip[3], iq[3];
	if(useNormal) {
		mexPrintf("Using Normal Map for Shading Constraints\n");
		normal = mxGetPr(prhs[5]);
		nthres = mxGetPr(prhs[6])[0];
	}
	image = mxGetPr(prhs[7]);
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
			for(k=0;k<4;k++)
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
				((float*)cvPtr2D(m_shadingConsMat, p, p))[0] += ws;
				((float*)cvPtr2D(m_shadingConsMat, q, q))[0] += ws;
				((float*)cvPtr2D(m_shadingConsMat, p, q))[0] += -ws;
				((float*)cvPtr2D(m_shadingConsMat, q, p))[0] += -ws;
				dist = (cp[0]*cq[0]+cp[1]*cq[1]+cp[2]*cq[2]);
				dist = acos(MAX(dist, 0.9998));
				dist = 2.0 * (1.0 - (cp[0]*cq[0]+cp[1]*cq[1]+cp[2]*cq[2]));	
				/*
				float weight = exp(MAX(exp( -exp(lp) * exp(lp) / (0.8*0.8) - exp(lq)*exp(lq) / (0.8*0.8)), 0));
				weight = weight * (exp(-dist*dist/0.000001));
				if(_isnan(weight)) weight = 0;
				((float*)cvPtr2D(m_refConsMat, p, p))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, q, q))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, p, q))[0] += -weight;
				((float*)cvPtr2D(m_refConsMat, q, p))[0] += -weight;

				float dI = lp - lq;
				mxGetPr(plhs[0])[p] += weight * dI;
				mxGetPr(plhs[0])[q] -= weight * dI;
				if(k == 0)
					mxGetPr(plhs[1])[p] = weight;
				*/
				
				if(dist < thres)
				{
					((float*)cvPtr2D(m_refConsMat, p, p))[0] += MIN(wr[p], wr[q]);
					((float*)cvPtr2D(m_refConsMat, q, q))[0] += MIN(wr[p], wr[q]);
					((float*)cvPtr2D(m_refConsMat, p, q))[0] += -MIN(wr[p], wr[q]);
					((float*)cvPtr2D(m_refConsMat, q, p))[0] += -MIN(wr[p], wr[q]);
					float dI = lp - lq;
					mxGetPr(plhs[0])[p] += MIN(wr[p], wr[q]) * dI;
					mxGetPr(plhs[0])[q] -= MIN(wr[p], wr[q]) * dI;
				} else {
					mxGetPr(plhs[1])[p] = 1.0;
				}
				
			}
		}
	}
	pushSparseMatrix(m_refConsMat, "WR");
	pushSparseMatrix(m_shadingConsMat, "WS");

	cvReleaseSparseMat(&m_refConsMat);
	cvReleaseSparseMat(&m_shadingConsMat);
}