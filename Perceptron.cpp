#include "stdafx.h"
#include "Perceptron.h"

Perceptron::Perceptron(int imagesCount)
{
	// Input neurons
	n = width * height;

	// Hidden neurons
	h = width * height;
	g = new double[h];

	// Output neurons
	m = imagesCount;
	y = new double[m];

	// Weights V
	v = new double[n * h];

	// Weights W
	w = new double[m * h];

	// Threshold Q
	Q = new double[h];

	// Threshold T
	T = new double[m];

	// Error correction d
	d = new double[m];

	// Error correction e
	e = new double[h];

	Initialize();
}

double Perceptron::Sigm(double x)
{
	return 1 / (1 + exp(-x));
}

void Perceptron::Initialize()
{
	//---------------------------------
	// Simple network initialization
	//---------------------------------
	std::uniform_real_distribution<double> unif(-1, 1);
	std::default_random_engine re;
	re.seed(time(NULL));

	// V matrix
	for (int i = 0; i < n * h; i++)
	{
		v[i] = unif(re);
	}

	// W matrix
	for (int i = 0; i < h * m; i++)
	{
		w[i] = unif(re);
	}

	// Q threshold
	for (int i = 0; i < h; i++)
	{
		Q[i] = unif(re);
	}

	// T threshold
	for (int i = 0; i < m; i++)
	{
		T[i] = unif(re);
	}
}

void Perceptron::Process(double* image)
{
	//---------------------------------
	// Calculate matrixes
	//---------------------------------
	for (int j = 0; j < h; j++)
	{
		double sum = 0;
		for (int i = 0; i < n; i++)
		{
			sum += v[j*n + i] * image[i];
		}
		g[j] = Sigm(sum + Q[j]);
	}

	for (int k = 0; k < m; k++)
	{
		double sum = 0;
		for (int j = 0; j < h; j++)
		{
			sum += w[k*h + j] * g[j];
		}
		y[k] = Sigm(sum + T[k]);
	}
}

void Perceptron::Correct(double* image, double* ethalon)
{
	//---------------------------------
	// Correction of values
	//---------------------------------

	// Calculate d
	for (int k = 0; k < m; k++)
	{
		d[k] = ethalon[k] - y[k];
	}

	// Calculate e
	for (int j = 0; j < h; j++)
	{
		double sum = 0;
		for (int k = 0; k < m; k++)
		{
			sum += d[k] * y[k] * (1 - y[k]) * w[k*h + j];
		}
		e[j] = sum;
	}

	// Recalculate w
	for (int k = 0; k < m; k++)
	{
		for (int j = 0; j < h; j++)
		{
			w[k*h + j] = w[k*h + j] + a * y[k] * (1 - y[k]) * d[k] * g[j];
		}
	}

	// Recalculate T
	for (int k = 0; k < m; k++)
	{
		T[k] = T[k] + a * y[k] * (1 - y[k]) * d[k];
	}

	// Recalculate v
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < h; j++)
		{
			v[i*h + j] = v[i*h + j] + b * g[j] * (1 - g[j]) * e[j] * image[i];
		}
	}

	// Recalculate Q
	for (int j = 0; j < h; j++)
	{
		Q[j] = Q[j] + b * g[j] * (1 - g[j]) * e[j];
	}
}

bool Perceptron::IsFinished()
{
	double maxD = abs(d[0]);
	//cout << maxD << endl;
	for (int k = 1; k < m; k++)
	{
		//cout << d[k] << endl;
		if (abs(d[k]) > maxD) maxD = abs(d[k]);
	}
	return maxD < D;
}

Perceptron::~Perceptron()
{
	delete[] d;
	delete[] e;
	delete[] v;
	delete[] w;
	delete[] Q;
	delete[] T;
	delete[] g;
	delete[] y;
}
