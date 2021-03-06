// COSI_LabWork4.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <random>
#include <conio.h>
#include <ctime>

#include "Perceptron.h"

using namespace std;

double image1[] =
{
	0,1,1,1,1,0,
	1,0,1,1,0,1,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	1,0,1,1,0,1,
	0,1,1,1,1,0
};

double image2[] =
{
	0,1,1,1,1,0,
	1,1,0,0,1,1,
	1,0,1,1,0,1,
	1,0,1,1,0,1,
	1,1,0,0,1,1,
	0,1,1,1,1,0
};

double image3[] =
{
	1,0,0,0,0,1,
	0,1,0,0,1,0,
	0,0,1,1,0,0,
	0,0,1,1,0,0,
	0,1,0,0,1,0,
	1,0,0,0,0,1
};

double image4[] =
{
	0,0,1,1,0,0,
	0,0,0,0,0,0,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	0,0,0,0,0,0,
	0,0,1,1,0,0
};

double image5[] =
{
	0,0,1,1,0,0,
	1,1,1,1,1,1,
	1,1,1,1,1,1,
	0,0,1,1,0,0,
	0,0,0,0,0,0,
	1,1,1,1,1,1
};

double yr1[] = { 1,0,0,0,0 };
double yr2[] = { 0,1,0,0,0 };
double yr3[] = { 0,0,1,0,0 };
double yr4[] = { 0,0,0,1,0 };
double yr5[] = { 0,0,0,0,1 };

const int width = 6;
const int height = 6;

int main()
{
	// Prepare test vector
	vector<double*> xs;
	xs.push_back(image1);
	xs.push_back(image2);
	xs.push_back(image3);
	xs.push_back(image4);
	xs.push_back(image5);

	// Prepare ethalon output vectors
	vector<double*> yr;
	yr.push_back(yr1);
	yr.push_back(yr2);
	yr.push_back(yr3);
	yr.push_back(yr4);
	yr.push_back(yr5);

	Perceptron perceptron(xs.size());

	// Show all test images from test vector
	for (int k = 0; k < xs.size(); k++)
	{
		cout << "Image " << k << endl << endl;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				char ch = (xs[k][i*width + j] == 1) ? '0' : ' ';
				cout << ch << ' ';
			}
			cout << endl;
		}
		cout << endl;
	}

	int round = 0;
	while(true)
	{
		//cout << "Round " << round << endl;
		for (int k = 0; k < xs.size(); k++)
		{
			perceptron.Process(xs[k]);
			perceptron.Correct(xs[k], yr[k]);
		}

		bool isFinished = perceptron.IsFinished();

		if (isFinished) break;
		else round++;
	}

	cout << "Network has achieved 95% of recognition after " << round << " rounds." << endl;


	double test[36] =
	{
		0,0,1,1,1,0,
		0,0,1,1,0,1,
		1,1,1,1,1,1,
		1,1,1,1,1,1,
		0,0,0,0,0,1,
		0,1,1,1,1,0
	};

	perceptron.Process(test);

	cout << "Test image was gotten" << endl;
	for (int k = 0; k < perceptron.m; k++)
	{
		cout << perceptron.y[k] << endl;
	}

	_getch();
    return 0;
}

