#ifndef __MONTE_CARLO_API_
#define __MONTE_CARLO_API_

#include "MonteCarlo_Base.h"



extern "C" {
	void MonteCarlo_API Initialize(MonteCarlo_Base** mc, u32 time_range_ps, u32 time_resolution, u32 spatial_range_mm, u32 spatial_resolution);
	void MonteCarlo_API DoMC(MonteCarlo_Base* mc, double mu_a, double mu_s, double g, double n);
	void MonteCarlo_API ProgressData(MonteCarlo_Base* mc);
}


class MonteCarlo_CLASS MonteCarlo_CUDA_Engine
{
public:
	void Initialize(u32 time_range_ps, u32 time_resolution, u32 spatial_range_mm, u32 spatial_resolution);
	void DoMC(double mu_a, double mu_s, double g, double n);
	void DoMC(double mu_a_start, double mu_a_step, double mu_a_end, double mu_s_start, double mu_s_step, double mu_s_end, double g, double n);
	void ProgressData( _process_data_cbk cbk = NULL );
private:
	MonteCarlo_Base* _mc;
}; 


#endif // __MONTE_CARLO_API_