#include "MonteCarlo_OCL.h"
#include <CL\opencl.h>
#include <assert.h>


MonteCarlo_OCL::MonteCarlo_OCL() :
	MonteCarlo_Base(0, 0,  Triple<u32>({ 0, 0, 0 }), Triple<u32>({0,0,0 }))
{
	Initialize();
}


void MonteCarlo_OCL::Initialize()
{

}

void MonteCarlo_OCL::ResetMemory()
{

}