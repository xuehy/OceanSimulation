#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <random>
#include "loadShader.h"
#include "cFFT.h"
#include <cufft.h>
#include <cuda.h>
#include <cudalibxt.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <complex>
#include "common.h"
using complex = std::complex<float>;


struct wave_vertex
{
	glm::vec3 v; // current position
	glm::vec3 normal;
	glm::vec3 htilde0;
	glm::vec3 htilde0mk;
	glm::vec3 position; // original position
};

struct complex_vector_normal
{
	complex h;
	glm::vec2 D;
	glm::vec3 n;
};

class Ocean
{
private:
	const float PI = 3.141592654f;
	const int repeatN = 30;
	bool option;
	float g;
	int N, Nplus1;
	float A;
	glm::vec2 w; // wind
	float length;
	complex* h_tilde, * h_tilde_slopex, * h_tilde_slopez;
	complex* h_tilde_dx, * h_tilde_dz;
	cFFT* fft;
	wave_vertex* vertices;
	unsigned int* indices;
	unsigned int indices_count;
	GLuint vbo_vertices, vbo_indices;
	GLuint glProgram;
	GLint light_direction, projection, view, viewPosID, skyBoxID;
	GLuint waterID;
	GLuint oceanTexture;
	GLuint skyTexture;
	std::random_device rd{};
	std::mt19937 generator{ rd() };
	std::normal_distribution<float> gaussian{ 0.0f, 1.0f };
	GLuint modelbuffer;
	cufftComplex* host_in, * host_out, * device_in, * device_out;
	cufftHandle /*cufftForwrdHandleRow, cufftForwrdHandleCol, */cufftForwrdHandle;
public:
	Ocean(const int N, const float A, const glm::vec2 w, const float length, bool option, GLuint cubemapTexture, GLuint oceanTexture);
	~Ocean();
	void release();
	float dispersion(int n_prime, int m_prime);
	float phillips(int n_prime, int m_prime);
	complex hTilde_0(int n_prime, int m_prime);
	complex hTilde(float t, int n_prime, int m_prime);
	complex_vector_normal h_D_and_n(glm::vec2 x, float t);
	void evaluateWaves(float t);
	void evaluateWavesFFT(float t);
	void render(float t, glm::vec3 light_dir, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, glm::vec3 viewPos, bool use_fft);
#ifdef USE_GPU
	void cuFFT(complex** input, complex** output);
#endif
};