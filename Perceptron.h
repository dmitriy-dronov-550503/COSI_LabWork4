#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
using namespace std;

class Perceptron
{
private:
	double Sigm(double x);
public:
	// Images width and height
	constexpr static int width = 6;
	constexpr static int height = 6;

	constexpr static double a = 1;		// Alpha
	constexpr static double b = 1/20;	// Beta

	constexpr static double D = 0.05;	// Finish reliability

	int n;		// Input neurons

	int h;		// Hidden neurons count
	double* g;	// Hidden neurons
	
	int m;		// Output neurons count
	double* y;	// Output neurons

	double* v;	// Weights V
	double* w;	// Weights W
	double* Q;	// Threshold Q
	double* T;	// Threshold T
	double* d;	// Error correction d
	double* e;	// Error correction e

	Perceptron(int imagesCount);
	void Initialize();
	void Process(double* image);
	void Correct(double* image, double* ethalon);
	bool IsFinished();
	~Perceptron();
};

