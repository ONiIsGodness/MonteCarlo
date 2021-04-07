#include "MonteCarlo_CUDA.cuh"
#include <Windows.h>

extern "C" {
	__global__ void MCd(const MonteCarlo_CUDA* pMC);
	__device__ float rand_MWC_oc(unsigned long long* a, unsigned int* b);
	__device__ float rand_MWC_co(unsigned long long* a, unsigned int* b);
	__device__ void LaunchPhoton(float3* a, float3* b, float* c);
	__device__ void Spin(float3* a1, float* a2, unsigned long long* a3, unsigned int* a4);
	__device__ unsigned int Reflect(float3 *a1, float3 *a2, float* a3, float* a4, float* a5, float* a6, unsigned long long* a7, unsigned int* a8, unsigned int* a9, const MonteCarlo_CUDA* pMC);
}

void MonteCarlo_CUDA::SavePhoton(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter)
{
	MonteCarlo_CUDA* pThis = (MonteCarlo_CUDA*)parameter;
	UP<char[]> buffer(new char[2048]);
	FILE* fd = fopen("database.txt", "a");
	assert(fd != NULL);
	
	unsigned int offset, rtn;
	offset = sprintf(buffer.get(), "\nmu_a: %.3f, mu_s: %.2f, g: %.2f, n: %.2f", pThis->mu_a, pThis->mu_s, pThis->g, pThis->n);
	fwrite(buffer.get(), offset, 1, fd);

	u32 sr = spatial_rslv[0];
	for (int i = 0; i < sr; ++i) {
		offset = 1;
		buffer[0] = '\n';
		for (int j = 0; j < time_rslv; ++j) {
			rtn = sprintf_s(buffer.get() + offset, 2048 - offset, "%d ", ((unsigned int*)data)[ i * time_rslv + j ]);
			offset += rtn;
			if (offset > 1024) {
				fwrite(buffer.get(), offset, 1, fd);
				offset = 0;
			}
		}
		fwrite(buffer.get(), offset, 1, fd);
	}
	fclose(fd);
	fd = NULL;
}

void MonteCarlo_CUDA::SavePhonen_BIN(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter)
{
	MonteCarlo_CUDA* pThis = (MonteCarlo_CUDA*)parameter;
	unsigned char buf[1024] = { 0 };
	FILE* fd = fopen("database.bin", "ab");
	assert(fd != NULL);

	u32 offset = 0,s ;
	
	vector<double> v({pThis->mu_a, pThis->mu_s, pThis->g, pThis->n});
	for (char i = 0; i < v.size(); ++i) {
		// 0 - mu_a, 1 - mu_s, 2 - g, 3 - n, 128 - data
		buf[offset] = i;
		++offset;

		*(u32*)(buf + offset) = sizeof(double);
		offset += sizeof(u32);

		*(double*)(buf + offset) = v[i];
		offset += sizeof(double);
	}
	
	buf[offset] = MC_OUT_PHOTON__curves;
	++offset;
	*(u32*)(buf + offset) = time_rslv * spatial_rslv[0] * sizeof(u32);
	offset += sizeof(u32);
	// Write Header
	fwrite(buf, offset, 1, fd);
	// Write Data
	fwrite((u8*)data, time_rslv * spatial_rslv[0] * sizeof(u32), 1, fd);
	
	fclose(fd);
	fd = NULL;
}

void MonteCarlo_CUDA::SavePhoton2(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter)
{
	UP<char[]> buffer(new char[2048]);
	FILE* fd = fopen("database_mid.txt", "a");
	assert( fd != NULL );

	unsigned int offset, rtn;
	u32 sr = spatial_rslv[0];

	offset = 0;
	for (int i = 0; i < time_rslv; ++i) {
		rtn = sprintf_s( buffer.get() + offset, 2048 - offset, "%d ", ((unsigned int*)data)[sr / 2 * time_rslv + i]);
		offset += rtn;
		if (offset > 1024) {
			fwrite(buffer.get(), offset, 1, fd);
			offset = 0;
		}
	}
	strcat(buffer.get(), "\n");
	fwrite(buffer.get(), offset + 1, 1, fd);
	fclose(fd);
	fd = NULL;
}

MonteCarlo_CUDA::MonteCarlo_CUDA(u32 time_range, u32 time_resolution,
	const Triple<u32>& spatial_range,
	const Triple<u32>& spatial_resolution) :
	spatial_delta_cm( radius_delta_mm[0] / 10.0f ),
	spatial_range_cm( radius_range_mm[0] / 10.0f ),
	MonteCarlo_Base(time_range, time_resolution, spatial_range, spatial_resolution),
	_cuda_buffer(NULL)
{

}

MonteCarlo_CUDA::~MonteCarlo_CUDA()
{
	CUDA_RELEASE( _cuda_buffer );
	CUDA_RELEASE(pThisCudaCopy);
}

void MonteCarlo_CUDA::Initialize()
{
	_buffer_size = time_resolution * spatial_resolution.Product();
	assert(_buffer_size <= 4 * 1024 * 1024 );
	MonteCarlo_Base::Initialize();

	_stored_result = move(UP<u32[]>(new u32[ _buffer_size ]));
	assert(_stored_result != NULL);
	cudaMalloc(&_cuda_buffer, _buffer_size * sizeof(u32));
	assert(_cuda_buffer != NULL );
	cudaMalloc( &pThisCudaCopy, sizeof(MonteCarlo_CUDA));
	LoadParamFile("safeprimes_base32.txt");

	_cbk = &MonteCarlo_CUDA::SavePhoton;
}

void MonteCarlo_CUDA::DoMC(double mu_a, double mu_s, double g, double n){
	MonteCarlo_Base::DoMC(mu_a, mu_s, g, n);
	ResetMemory();

	cudaError_t cudastat;
	
	dim3 dimBlock(NUM_THREADS_PER_BLOCK);
	dim3 dimGrid(NUM_BLOCKS);
	MCd <<<dimGrid, dimBlock >>>( pThisCudaCopy );

	cudaMemcpy(_stored_result.get(), _cuda_buffer, _buffer_size * sizeof(u32), cudaMemcpyDeviceToHost);	
	cudastat = cudaGetLastError();
	assert( 0 == cudastat );
}

void MonteCarlo_CUDA::DoMC(double g, double n,
	double mu_s_start, double mu_s_end, double mu_s_step,
	double mu_a_start, double mu_a_end, double mu_a_step)
{
	mu_a = mu_a_start;
	mu_s = mu_s_start;
	this->n = n;
	this->g = g;
	cudaError_t cudastat = cudaSuccess;

	MonteCarlo_Base::DoMC(g,n, mu_a_start, mu_a_end, mu_a_step, mu_s_start,mu_s_end, mu_s_step);
	
	dim3 dimBlock(NUM_THREADS_PER_BLOCK);
	dim3 dimGrid(NUM_BLOCKS);
	do {
		mu_s = mu_s_start;
		do {	
			MonteCarlo_Base::DoMC( mu_a,mu_s,g,n );
			ResetMemory();
			clock_t time = clock();
		
			cudaThreadSynchronize();

			MCd <<<dimGrid, dimBlock >>>(pThisCudaCopy);
			cudaMemcpy(_stored_result.get(), _cuda_buffer, _buffer_size * sizeof(u32), cudaMemcpyDeviceToHost);

			cudaThreadSynchronize();

			cudastat = cudaGetLastError();
			

			printf("cuda error no:%d (%s.) \nTime : %.2fmin (%.4fh)\n", cudastat, cudaGetErrorString(cudastat), (double( clock() - time ) / 1000 / 60), (double(clock() - time) / 1000 / 60 / 60));
			fflush(stdout);
			assert(0 == cudastat);

			this->ProgressData(&SavePhonen_BIN);
			SavePhoton(_stored_result.get(), time_resolution, spatial_resolution, this);
			SavePhoton2(_stored_result.get(), time_resolution, spatial_resolution, this);
			mu_s += mu_s_step;
		} while ( mu_s_step > 0.0f && mu_s <= mu_s_end );
		mu_a += mu_a_step;
	} while ( mu_a_step > 0.0f && mu_a <= mu_a_end );
}

void MonteCarlo_CUDA::ResetMemory()
{
	assert( _stored_result != NULL && 
			_cuda_buffer != NULL);

	ZeroMemory(_stored_result.get(), _buffer_size * sizeof(u32));
	cudaMemcpy(_cuda_buffer, _stored_result.get(), _buffer_size * sizeof(u32), cudaMemcpyHostToDevice);
	cudaMemcpy(pThisCudaCopy, this, sizeof(MonteCarlo_CUDA), cudaMemcpyHostToDevice);
}


void MonteCarlo_CUDA::LoadParamFile(const string& file_name)
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


__global__ void MCd( const MonteCarlo_CUDA* pMC )
{
	unsigned int* xd = (unsigned int*)pMC->xtest;
	unsigned int* cd = (unsigned int*)pMC->ctest;
	unsigned int* ad = (unsigned int*)pMC->atest;
	unsigned int* histd = (unsigned int*)pMC->_cuda_buffer;

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
	unsigned int ii = 0;
	
	for (ii = 0; ii<NUMSTEPS_GPU; ii++)
	{
		LaunchPhoton(&pos, &dir, &t);//Launch the photon
		while (true) {
			s = __fdividef(-__logf(rand_MWC_oc(&x, &a)), mus_max);//sample step length 														  
																  //Perform boundary crossing check here
			if ((pos.z + dir.z*s) <= 0) {
				//photon crosses boundary within the next step
				flag = Reflect(&dir, &pos, &t, &v, &cos_crit, &n, &x, &a, histd, pMC);
			}

			//Move (we can move the photons that have been terminated above since it improves our performance and does not affect our results)
			pos.x += s*dir.x;
			pos.y += s*dir.y;
			pos.z += s*dir.z;
			t += __fdividef(s, v);

			Spin(&dir, &g, &x, &a);

			if (t >= pMC->time_range_ps || flag >= 1) {//Kill photon and launch a new one
				flag = 0;
				//LaunchPhoton(&pos, &dir, &t);//Launch the photon
				break;
			}
		}
	}
	__syncthreads();//necessary?
}

__device__ float rand_MWC_co(unsigned long long* x,//unsigned int* c,
	unsigned int* a)
{
	*x = (*x & 0xffffffffull)*(*a) + (*x >> 32);
	return((float)((unsigned int)(*x & 0xffffffffull)) / (UINT_MAX));

}

__device__ float rand_MWC_oc(unsigned long long* x,//unsigned int* c,
	unsigned int* a)
{
	//Generate a random number (0,1]
	*x = (*x & 0xffffffffull)*(*a) + (*x >> 32);
	return(1.0f - (float)((unsigned int)(*x & 0xffffffffull)) / (UINT_MAX));
}


__device__ void LaunchPhoton(float3* pos, float3* dir, float* t)
{
	pos->x = 0.0f;
	pos->y = 0.0f;
	pos->z = 0.0f;

	dir->x = 0.0f;
	dir->y = 0.0f;
	dir->z = 1.0f;

	*t = 0.0f;
}

__device__ void Spin(float3* dir, float* g, unsigned long long* x,//unsigned int* c,
	unsigned int* a)
{
	float cost, sint;	// cosine and sine of the 
						// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
						// azimuthal angle psi. 
	float temp;

	float tempdir = dir->x;


	//This is more efficient for g!=0 but of course less efficient for g==0
	temp = __fdividef((1.0f - (*g)*(*g)), (1.0f - (*g) + 2.0f*(*g)*rand_MWC_co(x, a)));//Should be close close????!!!!!
	cost = __fdividef((1.0f + (*g)*(*g) - temp*temp), (2.0f*(*g)));
	if ((*g) == 0.0f)
		cost = 2.0f*rand_MWC_co(x, a) - 1.0f;


	sint = sqrtf(1.0f - cost*cost);
	__sincosf(2.0f*PI*rand_MWC_co(x, a), &cosp, &sinp);
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


__device__ unsigned int Reflect(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, unsigned long long* x,//unsigned int* c,
	unsigned int* a, unsigned int* histd, const MonteCarlo_CUDA* pMC )
{
	float r;

	if (-dir->z <= *cos_crit)
		r = 1.0f; //total internal reflection
	else
	{
		if (-dir->z == 1.0f)//normal incident
		{
			r = __fdividef((1.0f - *n), (1 + *n));
			r *= r;//square
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
	if (r<1.0f)
	{
		if (rand_MWC_co(x/*,c*/, a) <= r)//reflect
			r = 1.0f;
		else//transmitt
		{
			//calculate x and y where the photon escapes the medium
			r = __fdividef(pos->z, -dir->z);//dir->z must be finite since we have a boundary cross!
			pos->x += dir->x*r;
			pos->y += dir->y*r;
			*t += __fdividef(r, *v); //calculate the time when the photon exits

			r = sqrtf(pos->x*pos->x + pos->y*pos->y);

			//check for detection here
			if ( r < pMC->spatial_range_cm && *t < pMC->time_range_ps )
			{
				unsigned int ri = __float2uint_rz(__fdividef(r, pMC->spatial_delta_cm));
				unsigned int ti = __float2uint_rz(__fdividef((*t), pMC->delta_time_ps));
				atomicAdd(histd + ri * pMC->time_resolution + ti, 1);
				//atomicAdd(histd + __float2uint_rz(__fdividef((*t), pMC->delta_time_ps)), 1);//&histd[(unsigned int)floorf(__fdividef((t*),DT))],(unsigned int)1);
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
