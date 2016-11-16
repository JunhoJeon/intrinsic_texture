#include "LLE.h"

void LLE(cv::Mat1f &X, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K, int wSize, int width, int height)
{
	if(X.rows != N || X.cols != D) {
		printf("Failded. Invalid size X.\n");
		return;
	}
	printf("LLE running on %d points in %d dimensions\n", N, D);

	// % STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
	printf("--> Finding %d nearest neighbors for each pixel in %dx%d windows.\n", K, wSize, wSize);
	
	int *dataIdx = new int[wSize*wSize];
	for(int j=0; j<N; j++) {
		//if(X(j, 0) < 0.001 && X(j, 1) < 0.001, X(j, 2) < 0.001)
		//	continue;
		int jx = j % width;
		int jy = j / width;
		//if(jx < wSize/2 || jy < wSize/2 || jx >= width-wSize/2 || jy >= height-wSize/2)
		//	continue;
		/*
		if(jx == 0 && jy == 0)
			continue;
		if(jx == width-1 && jy == height-1)
			continue;
		if(jx == 0 && jy == height-1)
			continue;
		if(jx == width-1 && jy == 0)
			continue;
			*/
		ANNpointArray dataPts;
		ANNpoint queryPt;
		ANNidxArray nnIdx;
		ANNdistArray dists;
		ANNkd_tree * kdTree;

		queryPt = annAllocPt(D);
		dataPts = annAllocPts(wSize*wSize, D);
		nnIdx = new ANNidx[K];
		dists = new ANNdist[K];
		memset(dataIdx, 0, sizeof(int)*wSize*wSize);
		int nPts = 0;
		//for(;nPts < N; nPts++) {
		for(int wY = -wSize/2; wY <= wSize/2; wY++) {
			for(int wX = -wSize/2; wX <= wSize/2; wX++) {
				int x = j % width + wX;
				int y = j / width + wY;
				int idx = y * width + x;
				if(x < 0 || y < 0 || x >= width || y >= height)
					continue;
				if(idx == j) // except itself
					continue;
				//if(X(idx, 0) < 0.001 && X(idx, 1) < 0.001, X(idx, 2) < 0.001)
				//	continue;
				for(int d = 0; d < D; d++) {
					dataPts[nPts][d] = X(idx, d);
					dataIdx[nPts] = idx;
				}
				nPts++;
			}
		}
		//if(nPts < K)
		//	continue;
		//}

		//printf("---->kd_tree building...");
		kdTree = new ANNkd_tree(dataPts, nPts, D);

		double eps;
		//for(int n=0;n<N;n++) {
		for(int d=0; d<D;d++)
			queryPt[d] = X(j, d);//[j][d];

		kdTree->annkSearch(queryPt, K, nnIdx, dists, eps);
		int wrong = 0;
		for(int k=0;k<K;k++) {
			if(nnIdx[k] < 0 || nnIdx[k] > wSize*wSize) {
				//mexPrintf("%d %d %d\n", nnIdx[k], j, k);
				continue;
				//wrong++;
			}
			neighbors(j, k) = dataIdx[nnIdx[k]];
		}
		//if(wrong > 0)
		//	continue;
			/*			
			printf("\tNN:\tIndex\tDistance\n");
			for (int i = 0; i < K; i++) {			// print summary
				dists[i] = sqrt(dists[i]);			// unsquare distance
				printf("\t%d\t%d\t%f (%f %f %f)\n", i, nnIdx[i], dists[i], dataPts[nnIdx[i]][0], dataPts[nnIdx[i]][1], dataPts[nnIdx[i]][2]);
			}
			*/
		//}
		//printf("%d nearest neighbors found.\n", K);
		delete [] nnIdx;
		delete [] dists;
		//delete kdTree;
		//annDeallocPt(queryPt);
		//annDeallocPts(dataPts);
		annClose();
	}

	// % STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
	float tol;
	printf("--> Solving for reconstruction weights.\n");
	if(K > D) {
		printf("---->[note: K>D; regularization will be used]\n");
		tol = 1e-3; // % regularlizer in case constrained fits are ill conditioned
	} else {
		tol = 0;
	}

	cv::Mat1f z(K, D);
	
	for(int n=0; n<N; n++) {
		//if(X(n, 0) < 0.001 && X(n, 1) < 0.001, X(n, 2) < 0.001)
		//	continue;
		z.setTo(0);
		// % shift ith pt to origin
		for(int k=0;k<K;k++) {										
			int nidx = neighbors(n, k);
			if(nidx >= N || nidx < 0)
				continue;
			for(int d=0;d<D;d++) {							
				z(k, d) = X(nidx, d) - X(n, d);		
			}
		}

		// % local covariance
		cv::Mat1f C = z * z.t(); // C = KxK matrix

		// % regularlization (K>D)
		float t = cv::trace(C)[0];
		C = C + tol*t*cv::Mat1f::eye(K, K);

		// % solve Cw=1
		cv::Mat1f w(K, 1);
		cv::solve(C, cv::Mat1f::ones(K, 1), w);
		float ws = 0;
		for(int k=0;k<K;k++) {
			W(n, k) = w(k, 1);
			ws += w(k, 1);
		}
		// % enforce sum(w)=1
		for(int k=0;k<K;k++) {
			W(n, k) /= ws;
		}
	}
	printf("Done.\n");
	delete [] dataIdx;
	return;
}

/*
	input X: float[N][D]
	input D: input dimension, for RGB D=3
	intput N: the number of input data points
	output W: float [N][k] - k neighbor weight for N point
	output neighbors: int [N][k] - k neighbor index for N point
*/
void LLE(cv::Mat1f &X, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K = 12)
{
	if(X.rows != N || X.cols != D) {
		printf("Failded. Invalid size X.\n");
		return;
	}

	printf("LLE running on %d points in %d dimensions\n", N, D);


	// % STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
	printf("--> Finding %d nearest neighbors.\n", K);

	ANNpointArray dataPts;
	ANNpoint queryPt;
	ANNidxArray nnIdx;
	ANNdistArray dists;
	ANNkd_tree * kdTree;

	queryPt = annAllocPt(D);
	dataPts = annAllocPts(N, D);
	nnIdx = new ANNidx[K+1];
	dists = new ANNdist[K+1];

	int nPts = 0;
	for(;nPts < N; nPts++) {
		for(int d = 0; d < D; d++)
			dataPts[nPts][d] = X(nPts, d);
	}

	printf("---->kd_tree building...");
	kdTree = new ANNkd_tree(dataPts, nPts, D);
	printf("done.\n");

	double eps;
	for(int n=0;n<N;n++) {
		for(int d=0; d<D;d++)
			queryPt[d] = dataPts[n][d];

		kdTree->annkSearch(queryPt, K+1, nnIdx, dists, eps);
		for(int k=0;k<K;k++) {
			neighbors(n, k) = nnIdx[k+1];
		}
	}
	printf("%d nearest neighbors of %d points found.\n", K, N);
	/*
	delete [] nnIdx;
	delete [] dists;
	delete kdTree;
	annDeallocPt(queryPt);
	annDeallocPts(dataPts);
	annClose();
	*/
	// % STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
	float tol;
	printf("--> Solving for reconstruction weights.\n");
	if(K > D) {
		printf("---->[note: K>D; regularization will be used]\n");
		tol = 1e-3; // % regularlizer in case constrained fits are ill conditioned
	} else {
		tol = 0;
	}

	cv::Mat1f z(K, D);
	for(int n=0; n<N; n++) {
		z.setTo(0);
		// % shift ith pt to origin
		for(int k=0;k<K;k++) {										
			int nidx = neighbors(n, k);
			for(int d=0;d<D;d++) {							
				z(k, d) = X(nidx, d) - X(n, d);		
			}
		}

		// % local covariance
		cv::Mat1f C = z * z.t(); // C = KxK matrix

		// % regularlization (K>D)
		float t = cv::trace(C)[0];
		C = C + tol*t*cv::Mat1f::eye(K, K);

		// % solve Cw=1
		cv::Mat1f w(K, 1);
		cv::solve(C, cv::Mat1f::ones(K, 1), w);
		float ws = 0;
		for(int k=0;k<K;k++) {
			W(n, k) = w(k, 1);
			ws += w(k, 1);
		}
		// % enforce sum(w)=1
		float ws2 = 0;
		for(int k=0;k<K;k++) {
			W(n, k) /= ws;
		}
	}
	printf("Done.\n");
	return;
}

void LLE3(cv::Mat1f &X, cv::Mat1f &W, cv::Mat1i &neighbors, int N, int D, int K = 12)
{
	if(X.rows != N || X.cols != D) {
		printf("Failded. Invalid size X.\n");
		return;
	}

	printf("LLE running on %d points in %d dimensions\n", N, D);


	// % STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
	printf("--> Finding %d nearest neighbors.\n", K);

	ANNpointArray dataPts;
	ANNpoint queryPt;
	ANNidxArray nnIdx;
	ANNdistArray dists;
	ANNkd_tree * kdTree;

	queryPt = annAllocPt(D);
	dataPts = annAllocPts(N, D);
	nnIdx = new ANNidx[K+1];
	dists = new ANNdist[K+1];

	int nPts = 0;
	for(;nPts < N; nPts++) {
		for(int d = 0; d < D; d++)
			dataPts[nPts][d] = X(nPts, d);
	}

	printf("---->kd_tree building...");
	kdTree = new ANNkd_tree(dataPts, nPts, D);
	printf("done.\n");

	double eps;
	for(int n=0;n<N;n++) {
		for(int d=0; d<D;d++)
			queryPt[d] = dataPts[n][d];

		kdTree->annkSearch(queryPt, K+1, nnIdx, dists, eps);
		for(int k=0;k<K;k++) {
			neighbors(n, k) = nnIdx[k+1];
		}
	}
	printf("%d nearest neighbors of %d points found.\n", K, N);
	/*
	delete [] nnIdx;
	delete [] dists;
	delete kdTree;
	annDeallocPt(queryPt);
	annDeallocPts(dataPts);
	annClose();
	*/
	// % STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
	float tol;
	printf("--> Solving for reconstruction weight.\n");
	if(K > D-3) {
		printf("---->[note: K>D; regularization will be used]\n");
		tol = 1e-3; // % regularlizer in case constrained fits are ill conditioned
	} else {
		tol = 0;
	}

	cv::Mat1f z(K, D-3);
	for(int n=0; n<N; n++) {
		z.setTo(0);
		// % shift ith pt to origin
		for(int k=0;k<K;k++) {										
			int nidx = neighbors(n, k);
			for(int d=0;d<D-3;d++) {							
				z(k, d) = X(nidx, d) - X(n, d);		
			}
		}

		// % local covariance
		cv::Mat1f C = z * z.t(); // C = KxK matrix

		// % regularlization (K>D)
		float t = cv::trace(C)[0];
		C = C + tol*t*cv::Mat1f::eye(K, K);

		// % solve Cw=1
		cv::Mat1f w(K, 1);
		cv::solve(C, cv::Mat1f::ones(K, 1), w);
		float ws = 0;
		for(int k=0;k<K;k++) {
			W(n, k) = w(k, 1);
			ws += w(k, 1);
		}
		// % enforce sum(w)=1
		float ws2 = 0;
		for(int k=0;k<K;k++) {
			W(n, k) /= ws;
		}
	}
	printf("Done.\n");
	return;
}