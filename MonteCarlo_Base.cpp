#include "MonteCarlo_Base.h"
#pragma warning(disable : 4996)

void MonteCarlo_Base::QuickSort(void* data, size_t ElementSize, size_t Elements, compare_cond cond)
{
	if (Elements <= 1 || ElementSize <= 0 || NULL == cond)return;


	char* d = (char*)data;
	unique_ptr<u8[]> temp(new u8[ElementSize]);
	memcpy(temp.get(), data, ElementSize);
	size_t pos = 0;
	size_t left = ElementSize;
	size_t right = (Elements - 1) * ElementSize;
	bool right2left = true;

	while (right >= left) {
		if (right2left) {
			if (cond(d + right, temp.get())) {
				memcpy(d + pos, d + right, ElementSize);
				right2left = !right2left;
				pos = right;
			}
			right -= ElementSize;
		}
		else {
			if (cond(temp.get(), d + left)) {
				memcpy(d + pos, d + left, ElementSize);
				right2left = !right2left;
				pos = left;
			}
			left += ElementSize;
		}
	}
	memcpy(d + pos, temp.get(), ElementSize);
	QuickSort(d, ElementSize, pos / ElementSize, cond);
	QuickSort(d + pos + ElementSize, ElementSize, Elements - pos / ElementSize - 1, cond);
}

int MonteCarlo_Base::BinarySearch(void* data, size_t ElementSize, size_t Elements, const void* target, search_cond cond )
{
	if (Elements <= 0 || ElementSize <= 0 || NULL == cond)return -1;
	


	char* d = static_cast<char*>(data);
	int rtn = cond(d + (Elements / 2) * ElementSize, target);
	if (rtn == 0) return Elements / 2;
	if (rtn == 1) return  BinarySearch(d, ElementSize, Elements / 2, target, cond);
	return Elements / 2 + BinarySearch(d + (Elements / 2 + 1) * ElementSize, ElementSize, Elements - Elements / 2, target, cond) + 1;
}

MonteCarlo_Base::MonteCarlo_Base(u32 time_range, u32 time_resolution,
	const Triple<u32>& spatial_range,
	const Triple<u32>& spatial_resolution) :
	_cbk(NULL),
	time_range_ps(time_range),
	time_resolution(time_resolution),
	delta_time_ps(double(time_range) / double(time_resolution)),
	radius_delta_mm(spatial_range / spatial_resolution),
	radius_range_mm(spatial_range),
	spatial_resolution(spatial_resolution),
	_stored_result(new u32[1]),
	_buffer_size(0)
{
}

bool MonteCarlo_Base::ReadFileContent(const string& fileName, char*& content, size_t& size)
{
	FILE* fd = fopen(fileName.c_str(), "r");
	if (NULL == fd)return false;

	fseek(fd, 0, SEEK_END);
	size = ftell(fd);
	content = new char[size + 1];
	
	rewind(fd);
	fread_s(content, size, size, 1, fd);
	content[size] = '\0';
	fclose(fd);

	return true;
}

void MonteCarlo_Base::Initialize()
{
	printf("Basic information about Monte Carlo object using: \n"
		"\tstored time range: %dps, time resolution: %.4fps\n"
		"\tspatial range: %dmm, spatial resolution: %.3fmm",
		time_range_ps, delta_time_ps,
		radius_range_mm[0], radius_delta_mm[0]);

	fflush(stdout);
}

void MonteCarlo_Base::DoMC(double mu_a, double mu_s, double g, double n) {
	this->mu_a = mu_a;
	this->mu_s = mu_s;
	this->n = n;
	this->g = g;
	printf("\n\nInformation about this time`s simulation:\n"
		"\t mu_a: %.3f, mu_s: %.2f, g: %.2f, n: %.2f\n\n\n",
		this->mu_a, this->mu_s, this->g, this->n);

	fflush(stdout);
}

void MonteCarlo_Base::DoMC(double g, double n,
	double mu_s_start, double mu_s_end, double mu_s_step,
	double mu_a_start, double mu_a_end, double mu_a_step)
{
	printf("\n\n"
		"\tmu_a range: [%.3f : %.4f : %.3f]\n"
		"\tmu_s range: [%.2f : %.3f : %.3f]\n\n",
		mu_s_start, mu_s_step, mu_s_end,
		mu_a_start, mu_a_step, mu_a_end);

	fflush(stdout);
}

void MonteCarlo_Base::ProgressData(_process_data_cbk _cbk ) {
	if (NULL == _cbk && NULL == this->_cbk)
		return;
	if (NULL != _cbk)
		this->_cbk = _cbk;
	this->_cbk(_stored_result.get(), time_resolution, spatial_resolution, this);
}

void MonteCarlo_Base::LoadParamFile(const string& file_name) {}