#pragma once
#include "MonteCarlo_Base.h"
#include <CL/opencl.h>


class MonteCarlo_CLASS MonteCarlo_OCL :
	public MonteCarlo_Base
{
public:
	MonteCarlo_OCL();
	

	virtual void ResetMemory() override;
	
private:
	virtual void Initialize() override;

private:
};




