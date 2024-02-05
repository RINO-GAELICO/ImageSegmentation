#include "msImageProcessor.h"

#include <string.h>
#include <stdlib.h>
#include <iostream>


void msImageProcessor::FilterSerial(float sigmaS, float sigmaR)
{

	// Declare Variables
	int iterationCount, i, j, k;
	double mvAbs, diff, el;

	// make sure that a lattice height and width have
	// been defined...
	if (!height)
	{
		ErrorHandler("msImageProcessor", "LFilter", "Lattice height and width are undefined.");
		return;
	}

	// re-assign bandwidths to sigmaS and sigmaR
	if (((h[0] = sigmaS) <= 0) || ((h[1] = sigmaR) <= 0))
	{
		ErrorHandler("msImageProcessor", "Segment", "sigmaS and/or sigmaR is zero or negative.");
		return;
	}

	// define input data dimension with lattice
	int lN = N + 2;

	// Traverse each data point applying mean shift
	// to each data point

	// Allcocate memory for yk
	double *yk = new double[lN];

	// Allocate memory for Mh
	double *Mh = new double[lN];

	// let's use some temporary data
	double *sdata;
	sdata = new double[lN * L];

	// copy the scaled data
	int idxs, idxd;
	idxs = idxd = 0;

	
	// WE FOCUS ON GRAYSCALE
	if (N == 1)
	{
		for (i = 0; i < L; i++)
		{
			sdata[idxs++] = (i % width) / sigmaS;
			sdata[idxs++] = (i / width) / sigmaS;
			sdata[idxs++] = data[idxd++] / sigmaR;
		}
	}
	
	// index the data in the 3d buckets (x, y, L)
	int *buckets;
	int *slist;
	slist = new int[L];
	int bucNeigh[27]; // 27 because it is 3x3x3

	double sMins;	 // just for L
	double sMaxs[3]; // for all

	// we store the max values of each dimension
	//  the range of the scaled values for the intensity
	sMaxs[0] = width / sigmaS;
	sMaxs[1] = height / sigmaS;
	sMins = sMaxs[2] = sdata[2];
	idxs = 2;
	double cval;
	// find the min and max values of the intensity
	for (i = 0; i < L; i++)
	{
		cval = sdata[idxs];
		if (cval < sMins)
			sMins = cval;
		else if (cval > sMaxs[2])
			sMaxs[2] = cval;

		idxs += lN;
	}

	int nBuck1, nBuck2, nBuck3;
	int cBuck1, cBuck2, cBuck3, cBuck;
	nBuck1 = (int)(sMaxs[0] + 3);
	nBuck2 = (int)(sMaxs[1] + 3);
	nBuck3 = (int)(sMaxs[2] - sMins + 3);
	buckets = new int[nBuck1 * nBuck2 * nBuck3];
	for (i = 0; i < (nBuck1 * nBuck2 * nBuck3); i++)
	{
		buckets[i] = -1;
	}

	idxs = 0;
	for (i = 0; i < L; i++)
	{
		// find bucket for current data and add it to the list
		cBuck1 = (int)sdata[idxs] + 1;
		cBuck2 = (int)sdata[idxs + 1] + 1;
		cBuck3 = (int)(sdata[idxs + 2] - sMins) + 1;
		cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);

		slist[i] = buckets[cBuck];
		buckets[cBuck] = i;

		idxs += lN; // jump of three ahead
	}
	// let's print the size of the buckets

	std::cerr << "Size of buckets: " << nBuck1 * nBuck2 * nBuck3 << std::endl;
	std::cerr << "Size of slist: " << L << std::endl;
	std::cerr << "Size of sdata: " << lN * L << std::endl;
	// size of each nBuck
	std::cerr << "Size of nBuck1: " << nBuck1 << std::endl;
	std::cerr << "Size of nBuck2: " << nBuck2 << std::endl;
	std::cerr << "Size of nBuck3: " << nBuck3 << std::endl;

	// init bucNeigh
	idxd = 0;
	for (cBuck1 = -1; cBuck1 <= 1; cBuck1++)
	{
		for (cBuck2 = -1; cBuck2 <= 1; cBuck2++)
		{
			for (cBuck3 = -1; cBuck3 <= 1; cBuck3++)
			{
				bucNeigh[idxd++] = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3); // neighbors in the adjacent cube
			}
		}
	}

	// bucNeigh stores all the adjacent buckets using the bucket index
	//  we can use this to find the adjacent pixels
	
	double wsuml, weight;
	double hiLTr = 80.0 / sigmaR;
	// done indexing/hashing

	// proceed ...
	/**********************************************************************************/
	// Loop over each pixel
	for (i = 0; i < L; i++)
	{

		// Assign window center (window centers are
		// initialized by createLattice to be the point
		// data[i])
		idxs = i * lN; // skip over first two dimensions (x, y)

		for (j = 0; j < lN; j++)
		{
			// we take all the dimensions and store them in yk
			yk[j] = sdata[idxs + j];
		}
		// now in yk we have the x, y and intensity values for this specific pixel

		// Calculate the mean shift vector using the lattice
		// LatticeMSVector(Mh, yk);
		/*****************************************************/
		// Initialize mean shift vector with 0,0,0
		for (j = 0; j < lN; j++)
		{
			Mh[j] = 0;
		}

		wsuml = 0;
		// uniformLSearch(Mh, yk_ptr); // modify to new
		// find bucket of yk
		cBuck1 = (int)yk[0] + 1;
		cBuck2 = (int)yk[1] + 1;
		cBuck3 = (int)(yk[2] - sMins) + 1;

		cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);
		for (j = 0; j < 27; j++)
		{
			idxd = buckets[cBuck + bucNeigh[j]];
			// list parse, crt point is cHeadList
			while (idxd >= 0) 
			{
				idxs = lN * idxd;
				// determine if inside search window
				el = sdata[idxs + 0] - yk[0]; // first dimension
				diff = el * el;
				el = sdata[idxs + 1] - yk[1]; // second dimension
				diff += el * el;

				if (diff < 1.0) // if it is inside the search window
				{
					el = sdata[idxs + 2] - yk[2]; // store the intensity difference of the pixel
					if (yk[2] > hiLTr)
						diff = 4 * el * el;
					else
						diff = el * el;

					if (diff < 1.0)
					{
							// weightMap is a floating point array of size         */
							/*        (height x width) specifying for each pixel   */
							/*        edge strength.   
							*/
						weight = 1 - weightMap[idxd]; 
						for (k = 0; k < lN; k++)
							Mh[k] += weight * sdata[idxs + k]; // store the weighted sum of the pixel
						wsuml += weight;
					}
				}
				idxd = slist[idxd];
			}
		}
		// DONE WITH THE SEARCH
		if (wsuml > 0)
		{
			for (j = 0; j < lN; j++){
				Mh[j] = Mh[j] / wsuml - yk[j];
			}
				
		}
		else
		{
			for (j = 0; j < lN; j++){
				Mh[j] = 0;
			}
				
		}
		/*****************************************************/

		// Calculate its magnitude squared
		mvAbs = 0;
		for (j = 0; j < lN; j++)
			mvAbs += Mh[j] * Mh[j];

		// Keep shifting window center until the magnitude squared of the
		// mean shift vector calculated at the window center location is
		// under a specified threshold (Epsilon)

		// NOTE: iteration count is for speed up purposes only - it
		//       does not have any theoretical importance
		iterationCount = 1;
		while ((mvAbs >= EPSILON) && (iterationCount < LIMIT))
		{

			// Shift window location
			for (j = 0; j < lN; j++)
				yk[j] += Mh[j]; // new Mean/Center

			// Calculate the mean shift vector at the new
			// window location using lattice
			// LatticeMSVector(Mh, yk);
			/*****************************************************/
			// Initialize mean shift vector
			for (j = 0; j < lN; j++)
				Mh[j] = 0;
			wsuml = 0;
			// uniformLSearch(Mh, yk_ptr); // modify to new
			// find bucket of yk
			cBuck1 = (int)yk[0] + 1;
			cBuck2 = (int)yk[1] + 1;
			cBuck3 = (int)(yk[2] - sMins) + 1;
			cBuck = cBuck1 + nBuck1 * (cBuck2 + nBuck2 * cBuck3);

			// 
			for (j = 0; j < 27; j++)
			{
				idxd = buckets[cBuck + bucNeigh[j]];
				// list parse, crt point is cHeadList
				while (idxd >= 0)
				{
					idxs = lN * idxd;
					// determine if inside search window
					el = sdata[idxs + 0] - yk[0];
					diff = el * el;
					el = sdata[idxs + 1] - yk[1];
					diff += el * el;

					if (diff < 1.0)
					{
						el = sdata[idxs + 2] - yk[2];
						if (yk[2] > hiLTr)
							diff = 4 * el * el;
						else
							diff = el * el;

						if (diff < 1.0)
						{
							weight = 1 - weightMap[idxd];
							for (k = 0; k < lN; k++)
								Mh[k] += weight * sdata[idxs + k];
							wsuml += weight;
						}
					}
					idxd = slist[idxd];
				}
			}
			if (wsuml > 0)
			{
				for (j = 0; j < lN; j++)
					Mh[j] = Mh[j] / wsuml - yk[j];
			}
			else
			{
				for (j = 0; j < lN; j++)
					Mh[j] = 0;
			}
			/*****************************************************/

			// Calculate its magnitude squared
			// mvAbs = 0;
			// for(j = 0; j < lN; j++)
			//	mvAbs += Mh[j]*Mh[j];
			mvAbs = ((Mh[0] * Mh[0]) + (Mh[1] * Mh[1])) * sigmaS * sigmaS + (Mh[2] * Mh[2]) * sigmaR * sigmaR;

			// Increment iteration count
			iterationCount++;

		}//repeat until mvAbs < EPSILON or LIMIT is reached

		// Shift window location
		for (j = 0; j < lN; j++)
		{
			yk[j] += Mh[j];
		}

		// store result into msRawData...
		msRawData[i] = (float)(yk[2] * sigmaR);
	}

	// de-allocate memory
	delete[] buckets;
	delete[] slist;
	delete[] sdata;

	delete[] yk;
	delete[] Mh;

	// done.
	return;
}

