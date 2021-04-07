#include "MonteCarlo_PhotonDetail.cuh"
#include <Windows.h>
#define N 20


__global__ void MCd_2(const MonteCarlo_PhotonDetail* pMC);
__device__ float rand_MWC_oc_2(unsigned long long* a, unsigned int* b);
__device__ float rand_MWC_co_2(unsigned long long* a, unsigned int* b);
__device__ void LaunchPhoton_2(float3* a, float3* b, float* c);
__device__ void Spin_2(float3* a1, float* a2, unsigned long long* a3, unsigned int* a4);
__device__ unsigned int Reflect_2(float3 *a1, float3 *a2, float* a3, float* a4, float* a5, float* a6, unsigned long long* a7, unsigned int* a8, u32 offset_base, u32* n, const MonteCarlo_PhotonDetail* pMC);

void MonteCarlo_PhotonDetail::SavePhotons(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter)
{
	MonteCarlo_Base* pThis = (MonteCarlo_Base*)(parameter);
	u8 buf[512] = { 0 };
	FILE* fd = fopen("photons_db.bin","ab");
	assert(fd != NULL);

	size_t offset = 0;
	size_t size;
	*buf = MC_OUT_PHOTON__size;
	offset += 1;
	size = sizeof(size_t);
	memcpy(buf + offset, &size, sizeof(size_t));
	offset += sizeof(size);
	size = sizeof(OutputPhotonDetails);
	memcpy(buf + offset, &size, sizeof(size_t));
	offset += sizeof(size);

	*(buf + offset) = MC_OUT_PHOTON__photons;
	offset += 1;
	size = pThis->_buffer_size;
	memcpy(buf + offset, &size, sizeof(size_t));
	offset += sizeof(size);

	fwrite(buf, offset, 1, fd);
	QuickSort(data, sizeof(OutputPhotonDetails), size / sizeof(OutputPhotonDetails), &Compare);
	fwrite(data, pThis->_buffer_size, 1, fd);

	fclose(fd);
}

bool MonteCarlo_PhotonDetail::Compare(const void* photon1, const void* photon2)
{
	const OutputPhotonDetails* p1 = static_cast< const OutputPhotonDetails*>(photon1);
	const OutputPhotonDetails* p2 = static_cast< const OutputPhotonDetails*>(photon2);
	return (p1->x * p1->x + p1->y * p1->y) < (p2->x * p2->x + p2->y * p2->y);
}

MonteCarlo_PhotonDetail::MonteCarlo_PhotonDetail(u32 time_range, u32 time_resolution,
	const Triple<u32>& spatial_range,
	const Triple<u32>& spatial_resolution):
	MonteCarlo_Base(time_range	, time_resolution, spatial_range, spatial_resolution)
{
	this->time_range_ps = time_range;
}


MonteCarlo_PhotonDetail::~MonteCarlo_PhotonDetail()
{
	
}

void MonteCarlo_PhotonDetail::Initialize()
{
	// Malloc CPU & GPU memory
	_buffer_size = THREAD_PHOTONS * NUM_THREADS * sizeof(OutputPhotonDetails);
	_stored_result = move(UP<u32[]>(new u32[_buffer_size / sizeof(u32) ]));
	
	cudaMalloc(&pThisCudaCopy, sizeof(MonteCarlo_PhotonDetail));
	cudaMalloc(&_cuda_buffer, _buffer_size);
	assert( _stored_result != NULL && 
			_cuda_buffer != NULL && 
			pThisCudaCopy != NULL);
	LoadParamFile("safeprimes_base32.txt");
	fflush(stdout);
}
void MonteCarlo_PhotonDetail::DoMC(double mu_a, double mu_s, double g, double n)
{
	this->mu_a = mu_a;
	this->mu_s = mu_s;
	this->n = n;
	this->g = g;
	MonteCarlo_Base::DoMC(mu_a, mu_s, g, n);
	cudaError_t cudastat;
	for (int i = 0; i < N; ++i) {
		ResetMemory();
		dim3 dimBlock(NUM_THREADS_PER_BLOCK);
		dim3 dimGrid(NUM_BLOCKS);
		MCd_2 << <dimGrid, dimBlock >> > (pThisCudaCopy);

		cudaMemcpy(_stored_result.get(), _cuda_buffer, _buffer_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(xtest, pThisCudaCopy->xtest, sizeof(xtest), cudaMemcpyDeviceToHost);
		cudaMemcpy(atest, pThisCudaCopy->atest, sizeof(atest), cudaMemcpyDeviceToHost);
		cudaMemcpy(ctest, pThisCudaCopy->ctest, sizeof(ctest), cudaMemcpyDeviceToHost);

		ProgressData(&MonteCarlo_PhotonDetail::SavePhotons);
		cudastat = cudaGetLastError();
		assert(0 == cudastat);

		printf("\033[%dD",10);
		printf("[%.2f%%]", (double)(i + 1) / N * 100.0f);
		fflush(stdout);
	}
}

void MonteCarlo_PhotonDetail::ResetMemory()
{
	assert(_stored_result != NULL &&
		_cuda_buffer != NULL);

	ZeroMemory(_stored_result.get(), _buffer_size);
	cudaMemcpy(_cuda_buffer, _stored_result.get(), _buffer_size, cudaMemcpyHostToDevice);
	cudaMemcpy(pThisCudaCopy, this, sizeof(MonteCarlo_PhotonDetail), cudaMemcpyHostToDevice);
}

void MonteCarlo_PhotonDetail::LoadParamFile(const string& file_name)
{
	FILE *fd;
	unsigned int begin = 0u;
	unsigned long long int xinit = 1ull;
	unsigned int cinit = 0u;
	unsigned int fora, tmp1, tmp2;
	fd = fopen(file_name.c_str(), "r");//use an expanded list containing 50000 safeprimes instead of Steven's shorter list
	assert(fd != NULL);


	// use begin as a multiplier to generate the initial x's for 
	// the other generators...
	fscanf(fd, "%u %u %u", &begin, &tmp1, &tmp2);
	for (unsigned int i = 0; i<NUM_THREADS; i++)
	{
		xinit = xinit*begin + cinit;
		cinit = xinit >> 32;
		xinit = xinit & 0xffffffffull;
		xtest[i] = (unsigned int)xinit;
		fscanf(fd, "%u %u %u", &fora, &tmp1, &tmp2);
		atest[i] = fora;

		xinit = xinit*begin + cinit;
		cinit = xinit >> 32;
		xinit = xinit & 0xffffffffull;
		ctest[i] = (unsigned int)((((double)xinit) / UINT_MAX)*fora);
	}
	fclose(fd);
}

__global__ void MCd_2(const MonteCarlo_PhotonDetail* pMC)
{
	unsigned int* xd = (unsigned int*)pMC->xtest;
	unsigned int* cd = (unsigned int*)pMC->ctest;
	unsigned int* ad = (unsigned int*)pMC->atest;

	//First element processed by the block
	//int begin=NUM_THREADS_PER_BLOCK*bx;
	unsigned long long int x = cd[NUM_THREADS_PER_BLOCK*blockIdx.x + threadIdx.x];
	x = (x << 32) + xd[NUM_THREADS_PER_BLOCK*blockIdx.x + threadIdx.x];
	unsigned int a = ad[NUM_THREADS_PER_BLOCK*blockIdx.x + threadIdx.x];

	float3 pos, dir; //float triplet to store the position and direction
	float t, s;	//float to store the time of flight, step length

				// Some properties which correspponding to OPTICAL
	float mus_max = pMC->mu_a + pMC->mu_s;	//[1/cm]
	float g = pMC->g;
	float n = pMC->n;
	float v = 0.03 / n; // Speed of light in meida. [cm/ps] (c=0.03 [cm/ps] v=c/n)
	float cos_crit = sqrt(1 - (1.0f / n / n));//the critical angle for total internal reflection at the border cos_crit=sqrt(1-(nt/ni)^2)

	unsigned int flag = 0;
	u32 offset_base = ((blockIdx.x * NUM_THREADS_PER_BLOCK) + threadIdx.x) * THREAD_PHOTONS;
	u32 i = 0;

	while(i < THREAD_PHOTONS)
	{
		LaunchPhoton_2(&pos, &dir, &t);//Launch the photon
		while (true) {
			s = __fdividef(-__logf(rand_MWC_oc_2(&x, &a)), mus_max);//sample step length 														  
																  //Perform boundary crossing check here
			if ((pos.z + dir.z*s) <= 0) {
				//photon crosses boundary within the next step
				flag = Reflect_2(&dir, &pos, &t, &v, &cos_crit, &n, &x, &a, offset_base, &i, pMC);
				break;
			}

			//Move (we can move the photons that have been terminated above since it improves our performance and does not affect our results)
			pos.x += s*dir.x;
			pos.y += s*dir.y;
			pos.z += s*dir.z;
			t += __fdividef(s, v);

			Spin_2(&dir, &g, &x, &a);

			if (t >= pMC->time_range_ps || flag >= 1) {//Kill photon and launch a new one
				flag = 0;
				//LaunchPhoton(&pos, &dir, &t);//Launch the photon
				break;
			}
		}
	}
	cd[NUM_THREADS_PER_BLOCK*blockIdx.x + threadIdx.x] = x>>32;
	xd[NUM_THREADS_PER_BLOCK*blockIdx.x + threadIdx.x] = (0xffffffff) & x;
	ad[NUM_THREADS_PER_BLOCK*blockIdx.x + threadIdx.x] = a;
	__syncthreads();//necessary?
}


__device__ unsigned int Reflect_2(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, unsigned long long* x,//unsigned int* c,
	unsigned int* a, u32 offset_base, u32* offset, const MonteCarlo_PhotonDetail* pMC)
{
	float r;
	

	//if (-dir->z <= *cos_crit) {
	if(false){
		r = 1.0f; //total internal reflection
	}
	else
	{
		if (-dir->z == 1.0f)//normal incident
		{
			r = __fdividef((1.0f - *n), (1 + *n));
			r *= r;//square
			//x
		}
		else
		{
			
			//long and boring calculations of r
			float sinangle_i = sqrtf(1.0f - dir->z*dir->z);
			float sinangle_t = *n*sinangle_i;
			float cosangle_t = sqrtf(1.0f - sinangle_t*sinangle_t);

			float cossumangle = (-dir->z*cosangle_t) - sinangle_i*sinangle_t;
			float cosdiffangle = (-dir->z*cosangle_t) + sinangle_i*sinangle_t;
			float sinsumangle = sinangle_i*cosangle_t + (-dir->z*sinangle_t);
			float sindiffangle = sinangle_i*cosangle_t - (-dir->z*sinangle_t);

			r = 0.5*sindiffangle*sindiffangle*__fdividef((cosdiffangle*cosdiffangle + cossumangle*cossumangle), (sinsumangle*sinsumangle*cosdiffangle*cosdiffangle));
		}
	}
	if (r < 1.0f)
	{
		
		if (rand_MWC_co_2(x/*,c*/, a) <= r)//reflect
			r = 1.0f;
		else//transmitt
		{
			//calculate x and y where the photon escapes the medium
			r = __fdividef(pos->z, -dir->z);//dir->z must be finite since we have a boundary cross!
			pos->x += dir->x*r;
			pos->y += dir->y*r;
			*t += __fdividef(r, *v); //calculate the time when the photon exits

			//check for detection here
			if ( *t < pMC->time_range_ps )
			{
				OutputPhotonDetails* photon = pMC->_cuda_buffer + offset_base + *offset;
				*offset = *offset + 1;
				photon->x = pos->x;
				photon->y = pos->y;
				photon->t = *t;
				photon->phi = dir->z;
				return 1;
			}
			else return 2;
		}
	}
	if (r == 1.0f) {//reflect (mirror z and dz in reflection plane)
		pos->z *= -1;//mirror the z-coordinate in the z=0 plane, equal to a reflection.
		dir->z *= -1;// do the same to the z direction vector
	}
	return 0;
}



__device__ float rand_MWC_co_2(unsigned long long* x,//unsigned int* c,
	unsigned int* a)
{
	*x = (*x & 0xffffffffull)*(*a) + (*x >> 32);
	return((float)((unsigned int)(*x & 0xffffffffull)) / (UINT_MAX));

}

__device__ float rand_MWC_oc_2(unsigned long long* x,//unsigned int* c,
	unsigned int* a)
{
	//Generate a random number (0,1]
	*x = (*x & 0xffffffffull)*(*a) + (*x >> 32);
	return(1.0f - (float)((unsigned int)(*x & 0xffffffffull)) / (UINT_MAX));
}


__device__ void LaunchPhoton_2(float3* pos, float3* dir, float* t)
{
	pos->x = 0.0f;
	pos->y = 0.0f;
	pos->z = 0.0f;

	dir->x = 0.0f;
	dir->y = 0.0f;
	dir->z = 1.0f;

	*t = 0.0f;
}

__device__ void Spin_2(float3* dir, float* g, unsigned long long* x,//unsigned int* c,
	unsigned int* a)
{
	float cost, sint;	// cosine and sine of the 
						// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
						// azimuthal angle psi. 
	float temp;

	float tempdir = dir->x;


	//This is more efficient for g!=0 but of course less efficient for g==0
	temp = __fdividef((1.0f - (*g)*(*g)), (1.0f - (*g) + 2.0f*(*g)*rand_MWC_co_2(x, a)));//Should be close close????!!!!!
	cost = __fdividef((1.0f + (*g)*(*g) - temp*temp), (2.0f*(*g)));
	if ((*g) == 0.0f)
		cost = 2.0f*rand_MWC_co_2(x, a) - 1.0f;


	sint = sqrtf(1.0f - cost*cost);
	__sincosf(2.0f*PI*rand_MWC_co_2(x, a), &cosp, &sinp);
	temp = sqrtf(1.0f - dir->z*dir->z);

	if (temp == 0.0f)// normal incident.
	{
		dir->x = sint*cosp;
		dir->y = sint*sinp;
		dir->z = copysignf(cost, dir->z*cost);
	}
	else // regular incident.
	{
		dir->x = __fdividef(sint*(dir->x*dir->z*cosp - dir->y*sinp), temp) + dir->x*cost;
		dir->y = __fdividef(sint*(dir->y*dir->z*cosp + tempdir*sinp), temp) + dir->y*cost;
		dir->z = -sint*cosp*temp + dir->z*cost;
	}

	//normalisation seems to be required as we are using floats! Otherwise the small numerical error will accumulate
	temp = rsqrtf(dir->x*dir->x + dir->y*dir->y + dir->z*dir->z);
	dir->x = dir->x*temp;
	dir->y = dir->y*temp;
	dir->z = dir->z*temp;
}