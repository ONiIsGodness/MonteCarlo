#include "MonteCarlo_API.h"
#include "MonteCarlo_CUDA.cuh"
#include "MonteCarlo_PhotonDetail.cuh"

void Initialize(MonteCarlo_Base** mc, u32 time_range_ps, u32 time_resolution, u32 spatial_range_mm, u32 spatial_resolution)
{
	*mc = new MonteCarlo_PhotonDetail(	time_range_ps, time_resolution,
												Triple<u32>({ spatial_range_mm }), Triple<u32>({spatial_resolution}));
	assert( *mc != NULL );
	(*mc)->Initialize();
}

void DoMC(MonteCarlo_Base* mc, double mu_a, double mu_s, double g, double n)
{
	assert( mc != NULL );
	mc->DoMC(mu_a, mu_s, g, n);
}

void ProgressData(MonteCarlo_Base* mc)
{
	mc->ProgressData();
}


void MonteCarlo_CUDA_Engine::Initialize(u32 time_range_ps, u32 time_resolution, u32 spatial_range_mm, u32 spatial_resolution)
{
	_mc = new MonteCarlo_PhotonDetail(time_range_ps, time_resolution,
		Triple<u32>({ spatial_range_mm }), Triple<u32>({ spatial_resolution }));
	assert( _mc != NULL);
	_mc->Initialize();
}
void MonteCarlo_CUDA_Engine::DoMC(double mu_a, double mu_s, double g, double n)
{
	_mc->DoMC(mu_a, mu_s, g, n);
}

void MonteCarlo_CUDA_Engine::DoMC( double mu_a_start, double mu_a_step, double mu_a_end, double mu_s_start, double mu_s_step, double mu_s_end, double g, double n)
{
	_mc->DoMC(g,n, mu_s_start, mu_s_end, mu_s_step, mu_a_start, mu_a_end, mu_a_step);
}

void MonteCarlo_CUDA_Engine::ProgressData( _process_data_cbk cbk )
{
	_mc->ProgressData( cbk );
}