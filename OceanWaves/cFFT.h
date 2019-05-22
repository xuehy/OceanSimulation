#pragma once
#include <math.h>
#include <complex>
using complex = std::complex<float>;

class cFFT {
private:
	unsigned int N, which;
	unsigned int log_2_N;
	float pi2;
	unsigned int* reversed;
	complex** T;
	complex* c[2];
protected:
public:
	cFFT(unsigned int N);
	~cFFT();
	unsigned int reverse(unsigned int i);
	complex t(unsigned int x, unsigned int N);
	void fft(complex* input, complex* output, int stride, int offset);
};
