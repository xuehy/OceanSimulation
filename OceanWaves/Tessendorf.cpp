#include "Tessendorf.h"
void Check(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		std::cout << "ÐÐºÅ:" << __LINE__ << std::endl;
		std::cout << "´íÎó:" << cudaGetErrorString(status) << std::endl;
	}
}

Ocean::Ocean(const int N, const float A, const glm::vec2 w, const float length, bool option, GLuint cubemapTexture,
	GLuint oceanTexture) :
	g(9.81), option(option), N(N), Nplus1(N + 1), A(A), w(w), length(length),
	vertices(0), indices(0), h_tilde(0), h_tilde_slopex(0), h_tilde_slopez(0),
	h_tilde_dx(0), h_tilde_dz(0), fft(0)
{
	generator.seed(12131324);

	h_tilde = new complex[N * N];
	h_tilde_slopex = new complex[N * N];
	h_tilde_slopez = new complex[N * N];
	h_tilde_dx = new complex[N * N];
	h_tilde_dz = new complex[N * N];
#ifndef USE_GPU
	fft = new cFFT(N);
#endif
	vertices = new wave_vertex[(long long)Nplus1 * (long long)Nplus1];
	indices = new unsigned int[(long long)Nplus1 * (long long)Nplus1 * 10];

	int index;
	complex htilde0, htilde0mk_conj;

	// initialize vertices of the ocean mesh
	for (int m_prime = 0; m_prime < Nplus1; m_prime++)
		for (int n_prime = 0; n_prime < Nplus1; n_prime++)
		{
			index = m_prime * Nplus1 + n_prime;

			htilde0 = hTilde_0(n_prime, m_prime);
			htilde0mk_conj = std::conj(hTilde_0(-n_prime, -m_prime));

			vertices[index].htilde0.x = htilde0.real();
			vertices[index].htilde0.y = htilde0.imag();
			vertices[index].htilde0mk.x = htilde0mk_conj.real();
			vertices[index].htilde0mk.y = htilde0mk_conj.imag();

			// position coordinates between -length/2 to length/2
			vertices[index].position.x = vertices[index].v.x = (n_prime - N / 2.0f) * length / N;
			vertices[index].position.y = vertices[index].v.y = 0.0f;
			vertices[index].position.z = vertices[index].v.z = (m_prime - N / 2.0f) * length / N;

			vertices[index].normal.x = 0.0f;
			vertices[index].normal.y = 1.0f;
			vertices[index].normal.z = 0.0f;
		}
	indices_count = 0;
	for (int m_prime = 0; m_prime < N; m_prime++)
		for (int n_prime = 0; n_prime < N; n_prime++)
		{
			index = m_prime * Nplus1 + n_prime;
			if (option) // draw three lines to render the mesh
			{
				indices[indices_count++] = index;
				indices[indices_count++] = index + 1;

				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1;

				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1 + 1;

				if (n_prime == N - 1)
				{
					indices[indices_count++] = index + 1;
					indices[indices_count++] = index + Nplus1 + 1;
				}
				if (m_prime == N - 1)
				{
					indices[indices_count++] = index + Nplus1;
					indices[indices_count++] = index + Nplus1 + 1;
				}
			}
			else // draw two triangles on each vertex
			{
				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1;
				indices[indices_count++] = index + Nplus1 + 1;

				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1 + 1;
				indices[indices_count++] = index + 1;
			}
		}
	glProgram = LoadShaders("VertexShader.glsl", "FragmentShader.glsl");

	light_direction = glGetUniformLocation(glProgram, "light_direction");
	projection = glGetUniformLocation(glProgram, "Projection");
	view = glGetUniformLocation(glProgram, "View");
	model = glGetUniformLocation(glProgram, "Model");
	viewPosID = glGetUniformLocation(glProgram, "viewPos");
	skyBoxID = glGetUniformLocation(glProgram, "sky");
	waterID = glGetUniformLocation(glProgram, "water");
	glGenBuffers(1, &vbo_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
	glBufferData(GL_ARRAY_BUFFER, sizeof(wave_vertex) * (Nplus1) * (Nplus1), vertices, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_indices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_count * sizeof(unsigned int), indices, GL_STATIC_DRAW);

	skyTexture = cubemapTexture;
	this->oceanTexture = oceanTexture;

#ifdef USE_GPU
	Check(cudaMallocHost((void**)& host_in, 5 * N * N * sizeof(cufftComplex)));
	Check(cudaMallocHost((void**)& host_out, 5 * N * N * sizeof(cufftComplex)));
	Check(cudaMalloc((void**)& device_in, 5 * N * N * sizeof(cufftComplex)));
	Check(cudaMalloc((void**)& device_out, 5 * N * N * sizeof(cufftComplex)));
	
	int n[2] = { N, N };
	int imbed[2] = { 5 * N, N};
	cufftPlanMany(&cufftForwrdHandle, 2, n, imbed, 1, N* N,
		imbed, 1, N* N, CUFFT_C2C, 5);
#endif
}
#ifdef USE_GPU
void Ocean::cuFFT(complex** input,  complex** output)
{
	
	for (int i = 0; i < 5; i++)
		for(int j = 0; j < N * N; ++j)
	{
		host_in[j + i * N * N].x = input[i][j].real();
		host_in[j + i * N * N].y = input[i][j].imag();
		
	}
	Check(cudaMemcpy(device_in, host_in, 5 * N * N *sizeof(cufftComplex), cudaMemcpyHostToDevice));
	//cufftExecC2C(cufftForwrdHandleRow, device_in, device_out, CUFFT_FORWARD);
	//cufftExecC2C(cufftForwrdHandleCol, device_out, device_in, CUFFT_FORWARD);
	cufftExecC2C(cufftForwrdHandle, device_in, device_out, CUFFT_FORWARD);
	Check(cudaMemcpy(host_out, device_out, 5 * N * N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	for (int j = 0; j < 5; j++)
		for(int i = 0; i < N * N; ++i)
	{
		output[j][i].real( host_out[j * N * N + i].x);
		output[j][i].imag( host_out[j* N * N + i].y);
		
	}
}
#endif
Ocean::~Ocean()
{
	if (h_tilde) delete[] h_tilde;
	if (h_tilde_slopex) delete[] h_tilde_slopex;
	if (h_tilde_slopez) delete[] h_tilde_slopez;
	if (h_tilde_dx) delete[] h_tilde_dx;
	if (h_tilde_dz) delete[] h_tilde_dz;
	if (fft) delete[] fft;
	if (vertices) delete[] vertices;
	if (indices) delete[] indices;
}

void Ocean::release()
{
	glDeleteBuffers(1, &vbo_indices);
	glDeleteBuffers(1, &vbo_vertices);
#ifdef USE_GPU
	Check(cudaFreeHost(host_in));
	Check(cudaFreeHost(host_out));
	Check(cudaFree(device_in));
	Check(cudaFree(device_out));
	Check(cudaDeviceReset());
#endif
}

float Ocean::dispersion(int n_prime, int m_prime)
{
	float w_0 = 2.0f * PI / 200.0f;
	float kx = (2 * n_prime - N) * PI / length;
	float kz = (2 * m_prime - N) * PI / length;
	return floor(sqrt(g * sqrt(kx * kx + kz * kz))) * w_0;
}

float Ocean::phillips(int n_prime, int m_prime)
{
	glm::vec2 k(PI * (2 * n_prime - N) / length,
		PI * (2 * m_prime - N) / length);
	float k_length = glm::length(k);
	if (k_length < 0.000001) return 0.0;
	float k_length2 = k_length * k_length;
	float k_length4 = k_length2 * k_length2;
	float k_dot_w = glm::dot(glm::normalize(k), glm::normalize(w));
	float k_dot_w2 = k_dot_w * k_dot_w;

	float w_length = glm::length(w);
	float L = w_length * w_length / g;
	float L2 = L * L;
	float damping = 0.001;
	float l2 = L2 * damping * damping;

	return A * exp(-1.0f / (k_length2 * L2)) / k_length4 * k_dot_w2 * exp(-k_length2 * l2);
}


complex Ocean::hTilde_0(int n_prime, int m_prime)
{
	complex r = 0.0;
	r = complex(gaussian(generator), gaussian(generator));
	return r * sqrt(phillips(n_prime, m_prime) / 2.0f);
}

complex Ocean::hTilde(float t, int n_prime, int m_prime)
{
	int index = m_prime * Nplus1 + n_prime;

	complex htilde0(vertices[index].htilde0.x, vertices[index].htilde0.y);
	complex htilde0mkconj(vertices[index].htilde0mk.x, vertices[index].htilde0mk.y);

	float omegat = dispersion(n_prime, m_prime) * t;
	float cos_ = cos(omegat);
	float sin_ = sin(omegat);
	complex c0(cos_, sin_);
	complex c1(cos_, -sin_);
	complex res = htilde0 * c0 + htilde0mkconj * c1;

	return res;
}

complex_vector_normal Ocean::h_D_and_n(glm::vec2 x, float t)
{
complex h(0.0f, 0.0f);
glm::vec2 D(0.0f, 0.0f);
glm::vec3 n(0.0f, 0.0f, 0.0f);

complex c, res, htilde_c;
glm::vec2 k;
float kx, kz, k_length, k_dot_x;

for (int m_prime = 0; m_prime < N; m_prime++)
{
	kz = 2.0f * PI * (m_prime - N / 2.0f) / length;
	for (int n_prime = 0; n_prime < N; n_prime++)
	{
		kx = 2.0f * PI * (n_prime - N / 2.0f) / length;
		k = glm::vec2(kx, kz);
		k_length = glm::length(k);
		k_dot_x = glm::dot(k, x);
		c = complex(cos(k_dot_x), sin(k_dot_x));
		htilde_c = hTilde(t, n_prime, m_prime) * c;
		h += htilde_c;
		n += glm::vec3(-kx * htilde_c.imag(), 0.0f, -kz * htilde_c.imag());
		if (k_length < 0.000001) continue;
		D += glm::vec2(kx / k_length * htilde_c.imag(),
			kz / k_length * htilde_c.imag());
	}
}
n = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f) - n);
complex_vector_normal cvn;
cvn.h = h;
cvn.D = D;
cvn.n = n;
return cvn;
}

void Ocean::evaluateWaves(float t)
{
	float lambda = -1.0;
	int index;
	glm::vec2 x;
	complex_vector_normal h_d_and_n;
	for (int m_prime = 0; m_prime < N; m_prime++)
	{
		for (int n_prime = 0; n_prime < N; n_prime++)
		{
			index = m_prime * Nplus1 + n_prime;
			x = glm::vec2(vertices[index].v.x, vertices[index].v.z);
			h_d_and_n = h_D_and_n(x, t);
			vertices[index].v.y = h_d_and_n.h.real();
			vertices[index].v.x = vertices[index].position.x + lambda * h_d_and_n.D.x;
			vertices[index].v.z = vertices[index].position.z + lambda * h_d_and_n.D.y;

			vertices[index].normal.x = h_d_and_n.n.x;
			vertices[index].normal.y = h_d_and_n.n.y;
			vertices[index].normal.z = h_d_and_n.n.z;

			// mesh grid is larger than N x N, just use the first row and column
			// to fill the last row and column
			if (n_prime == 0 && m_prime == 0)
			{
				vertices[index + N + Nplus1 * N].v.y = h_d_and_n.h.real();

				vertices[index + N + Nplus1 * N].v.x = vertices[index + N + Nplus1 * N].position.x + lambda * h_d_and_n.D.x;
				vertices[index + N + Nplus1 * N].v.z = vertices[index + N + Nplus1 * N].position.z + lambda * h_d_and_n.D.y;

				vertices[index + N + Nplus1 * N].normal.x = h_d_and_n.n.x;
				vertices[index + N + Nplus1 * N].normal.y = h_d_and_n.n.y;
				vertices[index + N + Nplus1 * N].normal.z = h_d_and_n.n.z;
			}
			if (n_prime == 0)
			{
				vertices[index + N].v.y = h_d_and_n.h.real();

				vertices[index + N].v.x = vertices[index + N].position.x + lambda * h_d_and_n.D.x;
				vertices[index + N].v.z = vertices[index + N].position.z + lambda * h_d_and_n.D.y;

				vertices[index + N].normal.x = h_d_and_n.n.x;
				vertices[index + N].normal.y = h_d_and_n.n.y;
				vertices[index + N].normal.z = h_d_and_n.n.z;
			}
			if (m_prime == 0) {
				vertices[index + Nplus1 * N].v.y = h_d_and_n.h.real();

				vertices[index + Nplus1 * N].v.x = vertices[index + Nplus1 * N].position.x + lambda * h_d_and_n.D.x;
				vertices[index + Nplus1 * N].v.z = vertices[index + Nplus1 * N].position.z + lambda * h_d_and_n.D.y;

				vertices[index + Nplus1 * N].normal.x = h_d_and_n.n.x;
				vertices[index + Nplus1 * N].normal.y = h_d_and_n.n.y;
				vertices[index + Nplus1 * N].normal.z = h_d_and_n.n.z;
			}
		}
	}
}

void Ocean::evaluateWavesFFT(float t)
{
	float kx, kz, len, lambda = -1.0f;
	int index, index1;

	for (int m_prime = 0; m_prime < N; m_prime++) {
		kz = PI * (2.0f * m_prime - N) / length;
		for (int n_prime = 0; n_prime < N; n_prime++) {
			kx = PI * (2 * n_prime - N) / length;
			len = sqrt(kx * kx + kz * kz);
			index = m_prime * N + n_prime;

			h_tilde[index] = hTilde(t, n_prime, m_prime);
			h_tilde_slopex[index] = h_tilde[index] * complex(0, kx);
			h_tilde_slopez[index] = h_tilde[index] * complex(0, kz);
			if (len < 0.000001f) {
				h_tilde_dx[index] = complex(0.0f, 0.0f);
				h_tilde_dz[index] = complex(0.0f, 0.0f);
			}
			else {
				h_tilde_dx[index] = h_tilde[index] * complex(0, -kx / len);
				h_tilde_dz[index] = h_tilde[index] * complex(0, -kz / len);
			}
		}
	}
#ifdef USE_GPU
	complex* in_[] = { h_tilde, h_tilde_slopex, h_tilde_slopez, h_tilde_dx, h_tilde_dz };
	complex* out_[] = { h_tilde, h_tilde_slopex, h_tilde_slopez, h_tilde_dx, h_tilde_dz };
	cuFFT(in_, out_);
#else
	for (int m_prime = 0; m_prime < N; m_prime++) {
		fft->fft(h_tilde, h_tilde, 1, m_prime * N);
		fft->fft(h_tilde_slopex, h_tilde_slopex, 1, m_prime * N);
		fft->fft(h_tilde_slopez, h_tilde_slopez, 1, m_prime * N);
		fft->fft(h_tilde_dx, h_tilde_dx, 1, m_prime * N);
		fft->fft(h_tilde_dz, h_tilde_dz, 1, m_prime * N);
	}
	for (int n_prime = 0; n_prime < N; n_prime++) {
		fft->fft(h_tilde, h_tilde, N, n_prime);
		fft->fft(h_tilde_slopex, h_tilde_slopex, N, n_prime);
		fft->fft(h_tilde_slopez, h_tilde_slopez, N, n_prime);
		fft->fft(h_tilde_dx, h_tilde_dx, N, n_prime);
		fft->fft(h_tilde_dz, h_tilde_dz, N, n_prime);
	}
#endif

	int sign;
	float signs[] = { 1.0f, -1.0f };
	glm::vec3 n;
	for (int m_prime = 0; m_prime < N; m_prime++) {
		for (int n_prime = 0; n_prime < N; n_prime++) {
			index = m_prime * N + n_prime;     // index into h_tilde..
			index1 = m_prime * Nplus1 + n_prime;    // index into vertices

			sign = signs[(n_prime + m_prime) & 1];

			h_tilde[index] = h_tilde[index] * (float) sign;

			// height
			vertices[index1].v.y = h_tilde[index].real();

			// displacement
			h_tilde_dx[index] = h_tilde_dx[index] * (float) sign;
			h_tilde_dz[index] = h_tilde_dz[index] * (float) sign;
			vertices[index1].v.x = vertices[index1].position.x + h_tilde_dx[index].real() * lambda;
			vertices[index1].v.z = vertices[index1].position.z + h_tilde_dz[index].imag() * lambda;

			// normal
			h_tilde_slopex[index] = h_tilde_slopex[index] * (float) sign;
			h_tilde_slopez[index] = h_tilde_slopez[index] * (float) sign;
			n = glm::normalize(glm::vec3(0.0f - h_tilde_slopex[index].real(), 1.0f, 0.0f - h_tilde_slopez[index].real()));
			vertices[index1].normal.x = n.x;
			vertices[index1].normal.y = n.y;
			vertices[index1].normal.z = n.z;

			// for tiling
			if (n_prime == 0 && m_prime == 0) {
				vertices[index1 + N + Nplus1 * N].v.y = h_tilde[index].real();

				vertices[index1 + N + Nplus1 * N].v.x = vertices[index1 + N + Nplus1 * N].position.x + h_tilde_dx[index].real() * lambda;
				vertices[index1 + N + Nplus1 * N].v.z = vertices[index1 + N + Nplus1 * N].position.z + h_tilde_dz[index].real() * lambda;

				vertices[index1 + N + Nplus1 * N].normal.x = n.x;
				vertices[index1 + N + Nplus1 * N].normal.y = n.y;
				vertices[index1 + N + Nplus1 * N].normal.z = n.z;
			}
			if (n_prime == 0) {
				vertices[index1 + N].v.y = h_tilde[index].real();

				vertices[index1 + N].v.x = vertices[index1 + N].position.x + h_tilde_dx[index].real() * lambda;
				vertices[index1 + N].v.z = vertices[index1 + N].position.z + h_tilde_dz[index].real() * lambda;

				vertices[index1 + N].normal.x = n.x;
				vertices[index1 + N].normal.y = n.y;
				vertices[index1 + N].normal.z = n.z;
			}
			if (m_prime == 0) {
				vertices[index1 + Nplus1 * N].v.y = h_tilde[index].real();

				vertices[index1 + Nplus1 * N].v.x = vertices[index1 + Nplus1 * N].position.x + h_tilde_dx[index].real() * lambda;
				vertices[index1 + Nplus1 * N].v.z = vertices[index1 + Nplus1 * N].position.z + h_tilde_dz[index].real() * lambda;

				vertices[index1 + Nplus1 * N].normal.x = n.x;
				vertices[index1 + Nplus1 * N].normal.y = n.y;
				vertices[index1 + Nplus1 * N].normal.z = n.z;
			}
		}
	}
}

void Ocean::render(float t, glm::vec3 light_dir, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, glm::vec3 viewPos, bool use_fft)
{
	static bool eval = false;
	if (!use_fft && !eval) {
		eval = true;
		evaluateWaves(t);
	}
	else if (use_fft) {
		evaluateWavesFFT(t);
	}

	glUseProgram(glProgram);
	glUniform3f(light_direction, light_dir.x, light_dir.y, light_dir.z);
	glUniformMatrix4fv(projection, 1, GL_FALSE, &Projection[0][0]);
	glUniformMatrix4fv(view, 1, GL_FALSE, &View[0][0]);
	glUniformMatrix4fv(model, 1, GL_FALSE, &Model[0][0]);
	glUniform3f(viewPosID, viewPos.x, viewPos.y, viewPos.z);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skyTexture);
	glUniform1i(skyBoxID, 0);
	glActiveTexture(GL_TEXTURE0+1);
	glBindTexture(GL_TEXTURE_2D, oceanTexture);
	glUniform1i(waterID, 1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
	// dont use glBufferData to avoid reallocationg the data store
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(wave_vertex) * Nplus1 * Nplus1, vertices);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 15 * sizeof(float), 0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 15 * sizeof(float), (void*)offsetof(wave_vertex, normal));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 15 * sizeof(float), (void*)offsetof(wave_vertex, htilde0));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	// repeat to make 10 x 10 fields
	int NN = 50;
	for (int j = 0; j < NN; j++) {
		for (int i = 0; i < NN; i++) {
			Model = glm::translate(glm::mat4(1.0f), glm::vec3(length * (i-NN/2), 0, length * -(j-NN/2)));
			Model = glm::scale(Model, glm::vec3(1.01f, 1.01f, 1.01f));
			
			glUniformMatrix4fv(model, 1, GL_FALSE, &Model[0][0]);
			glDrawElements(option ? GL_LINES : GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, 0);
		}
	}
}