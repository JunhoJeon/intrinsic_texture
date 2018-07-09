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
	double *normal = mxGetPr(prhs[0]);
	double dist;
	plhs[0] = mxCreateDoubleMatrix(h, w, mxREAL);
	int dims[2] = {h*w, h*w};
	CvSparseMat* m_refConsMat = cvCreateSparseMat(2, dims, CV_32FC1);
	float sig_c= mxGetPr(prhs[1])[0];

	for(j=0;j<h;j++)
	{
		for(i=0;i<w;i++)
		{
			int p = i*h+j;
			cp[0] = normal[i*h+j];
			cp[1] = normal[h*w+i*h+j];
			cp[2] = normal[2*h*w+i*h+j];
			for(k=0;k<8;k++)
			{
				int qi = i + nx[k];
				int qj = j + ny[k];
				int q = qi*h+qj;
				if(qi < 0 || qj < 0 || qi >= w || qj >= h)
				{
					continue;
				}
				cq[0] = normal[qi*h+qj];
				cq[1] = normal[h*w+qi*h+qj];
				cq[2] = normal[2*h*w+qi*h+qj];

				dist = 2.0 * (1.0 - (cp[0]*cq[0]+cp[1]*cq[1]+cp[2]*cq[2]));	

				float weight = (exp(-dist*dist/(sig_c*sig_c)));
				if(k == 2)
					mxGetPr(plhs[0])[p] = weight;

				if(std::isnan(weight)) weight = 0;
				((float*)cvPtr2D(m_refConsMat, p, p))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, q, q))[0] += weight;
				((float*)cvPtr2D(m_refConsMat, p, q))[0] += -weight;
				((float*)cvPtr2D(m_refConsMat, q, p))[0] += -weight;
			}
		}
	}
	pushSparseMatrix(m_refConsMat, "WSC");

	cvReleaseSparseMat(&m_refConsMat);
}