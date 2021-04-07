#ifndef __MONTE_CARLO_CUDA_CUH
#define __MONTE_CARLO_CUDA_CUH

#include "MonteCarlo_Base.h"
#include "math_constants.h"

#define CUDA_RELEASE( _buff )\
do {\
	if (NULL == _buff) {\
		cudaFree(_buff); \
		_buff = NULL; \
	}\
} while (false)

#define NUM_THREADS_PER_BLOCK	320			//Keep above 192 to eliminate global memory access overhead
#define NUM_BLOCKS				84			//Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)
#define NUM_THREADS				26880
#define NUMSTEPS_GPU			14854570
#define PI						3.14159265f

typedef size_t u32;

class MonteCarlo_CUDA :
	public MonteCarlo_Base
{
public:
	MonteCarlo_CUDA(u32 time_range, u32 time_resolution,
		const Triple<u32>& spatial_range,
		const Triple<u32>& spatial_resolution);
	~MonteCarlo_CUDA();
public:
	virtual void Initialize() override;
	virtual void DoMC(double mu_a, double mu_s, double g, double n) override;
	virtual void DoMC(double g, double n,
		double mu_a_start, double mu_a_end, double mu_a_step,
		double mu_s_start, double mu_s_end, double mu_s_step) override;

private:
	static void SavePhoton(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter);
	static void SavePhoton2(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter);
	static void SavePhonen_BIN(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter);
	void ResetMemory() override;
	void LoadParamFile(const string& file_name) override;


public:
	MonteCarlo_CUDA*	pThisCudaCopy;
	void*				_cuda_buffer;
	unsigned int		xtest[NUM_THREADS];
	unsigned int		ctest[NUM_THREADS];
	unsigned int		atest[NUM_THREADS];
	double				spatial_delta_cm;
	double				spatial_range_cm;
};



#endif // __MONTE_CARLO_CUDA_CUH