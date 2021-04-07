#pragma once
#include "MonteCarlo_Base.h"


#define NUM_THREADS_PER_BLOCK	320			//Keep above 192 to eliminate global memory access overhead
#define NUM_BLOCKS				84			//Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)
#define NUM_THREADS				26880
#define THREAD_PHOTONS			200
#define PI						3.14159265f

struct OutputPhotonDetails {
	double x; 
	double y;
	double t;
	double phi;
};

class MonteCarlo_PhotonDetail :
	public MonteCarlo_Base
{
public:
	MonteCarlo_PhotonDetail(u32 time_range, u32 time_resolution,
		const Triple<u32>& spatial_range,
		const Triple<u32>& spatial_resolution);
	~MonteCarlo_PhotonDetail();

public:
	virtual void Initialize() override;
	virtual void DoMC(double mu_a, double mu_s, double g, double n);

	static void SavePhotons(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter);
	static bool Compare(const void* photon1, const void* photon2);

	MonteCarlo_PhotonDetail*	pThisCudaCopy;
	OutputPhotonDetails*		_cuda_buffer;
	unsigned int				xtest[NUM_THREADS];
	unsigned int				ctest[NUM_THREADS];
	unsigned int				atest[NUM_THREADS];

private:
	void ResetMemory();
	void LoadParamFile(const string& file_name);
};

