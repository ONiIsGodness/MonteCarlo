#ifndef __MONTE_CARLO_BASE_H
#define __MONTE_CARLO_BASE_H

#include <memory>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
using namespace std;


#define SP shared_ptr
#define UP unique_ptr
#ifdef VS_PROJECT

#ifdef TJU_BME_EXPORT 
#define MonteCarlo_API _declspec(dllexport)
#define MonteCarlo_CLASS _declspec(dllexport)	
#else
#pragma comment(lib, "MC_CUDA.lib")
#define  MonteCarlo_API _declspec(dllimport)
#define  MonteCarlo_CLASS _declspec(dllimport)
#endif // TJU_BME_EXPORT

#else

#define MonteCarlo_API  
#define MonteCarlo_CLASS

#endif // VS_PROJECT

typedef size_t			u32;
typedef unsigned char 	u8;

typedef enum _MONTE_CARLO_DATA_TYPE_CODE {
	MC_OUT_PHOTON__mu_a = 0,
	MC_OUT_PHOTON__mu_s,
	MC_OUT_PHOTON__g,
	MC_OUT_PHOTON__n,
	MC_OUT_PHOTON__size,
	MC_OUT_PHOTON__curves = 128,
	MC_OUT_PHOTON__photons,
}DataTypeCode;

template <typename T>
class Triple {
public:
	Triple(const Triple& tmp)
	{
		Set(tmp.number);
	}
	Triple(Triple&& tmp) {
		Set(tmp.number);
	}
	Triple(const vector<T>& v) {
		Set(v);
	}

public:
	void Set(const vector<T>& v) {
		number.clear();
		size_t _dims = v.size() >= 3 ? 3 : v.size();
		for (size_t i = 0; i < _dims; ++i)
			number.push_back(v[i]);
	}
	Triple<T> operator+(const Triple<T>& t) {
		vector<double> tmp;
		for (int i = 0; i < number.size() && i < t.number.size(); ++i) {
			tmp.push_back(number[i] + t.number[i]);
		}
		return Triple<T>(tmp);
	}
	Triple<T> operator-(const Triple<T>& t) {
		vector<double> tmp;
		for (int i = 0; i < number.size() && i < t.number.size(); ++i) {
			tmp.push_back(number[i] - t.number[i]);
		}
		return Triple<T>(tmp);
	}
	Triple<T> operator*(const Triple<T>& t) {
		vector<double> tmp;
		for (int i = 0; i < number.size() && i < t.number.size(); ++i) {
			tmp.push_back(number[i] * t.number[i]);
		}
		return Triple<T>(tmp);
	}
	Triple<double> operator/(const Triple<T> t) {
		vector<double> tmp;
		for (int i = 0; i < number.size() && i < t.number.size(); ++i) {
			tmp.push_back((double)number[i] / (double)t.number[i]);
		}
		return Triple<double>(tmp);
	}

	friend Triple<T> operator+(const Triple<T>& opt1, const Triple<T>& opt2) {
		vector<double> tmp;
		for (int i = 0; i < opt1.number.size() && i < opt2.number.size(); ++i) {
			tmp.push_back(opt1.number[i] + opt2.number[i]);
		}
		return Triple<T>(tmp);
	}
	friend Triple<T> operator-(const Triple<T>& opt1, const Triple<T>& opt2) {
		vector<double> tmp;
		for (int i = 0; i < opt1.number.size() && i < opt2.number.size(); ++i) {
			tmp.push_back(opt1.number[i] - opt2.number[i]);
		}
		return Triple<T>(tmp);
	}
	friend Triple<T> operator*(const Triple<T>& opt1, const Triple<T>& opt2) {
		vector<double> tmp;
		for (int i = 0; i < opt1.number.size() && i < opt2.number.size(); ++i) {
			tmp.push_back(opt1.number[i] * opt2.number[i]);
		}
		return Triple<T>(tmp);
	}
	friend Triple<double> operator/(const Triple<T>& opt1, const Triple<T>& opt2) {
		vector<double> tmp;
		for (int i = 0; i < opt1.number.size() && i < opt2.number.size(); ++i) {
			tmp.push_back((double)opt1.number[i] / (double)opt2.number[i]);
		}
		Triple<double> rtn(tmp);
		return rtn;
	}

	Triple<T>& operator+=(const Triple<T>& t) {
		for (int i = 0; i < number.size() && i < t.number.size(); ++i)
			number[i] += t.number[i];

		return *this;
	};

	Triple<T>& operator-=(const Triple<T>& t) {
		for (int i = 0; i < number.size() && i < t.number.size(); ++i)
			number[i] -= t.number[i];
		return *this;
	}
	Triple<T>& operator*=(const Triple<T>& t) {
		for (int i = 0; i < number.size() && i < t.number.size(); ++i)
			number[i] *= t.number[i];
		return *this;
	}
	Triple<T>& operator/=(const Triple<T>& t) {
		for (int i = 0; i < number.size() && i < t.number.size(); ++i)
			number[i] /= t.number[i];
		return *this;
	}
	Triple<T>& operator=(const Triple<double>& td)
	{
		vector<T> tmp;
		for (int i = 0; i < td.number.size(); ++i)
			tmp.push_back(static_cast<T>(td.number[i]));
		this->Set(tmp);
		return *this;
	}
	T operator[](u32 index) const
	{
		assert(index < number.size());
		return number[index];
	}

	T InnerProduct(const Triple<T>& t) {
		T rtn = 0;
		assert(t.number.size() == number.size());
		for (int i = 0; i < number.size(); ++i)
			rtn += number[i] * t.number[i];
		return rtn;
	}


	T Product()
	{
		T rtn = 1;
		for (int i = 0; i < number.size(); ++i)
			rtn *= number[i];
		return rtn;
	}


private:
	vector<T> number;
};


typedef void(*_process_data_cbk)(void *const data, u32 time_rslv, const Triple<u32>& spatial_rslv, void* parameter);
typedef bool(*compare_cond)(const void* arg1, const void* arg2);
typedef int(*search_cond)(const void* arg1, const void* param);

class MonteCarlo_CLASS MonteCarlo_Base
{
public:
	MonteCarlo_Base(u32 time_range, u32 time_resolution,
		const Triple<u32>& spatial_range,
		const Triple<u32>& spatial_resolution);

	virtual ~MonteCarlo_Base() {}

	/* DO INITIAL ACTION ( malloc memory etc. )*/
	virtual void Initialize();
	

	/* Execute one time of Monte Carlo Simulation.*/
	virtual void DoMC(double mu_a, double mu_s, double g, double n);

	virtual void DoMC(	double g, double n, 
						double mu_s_start, double mu_s_end, double mu_s_step,
						double mu_a_start, double mu_a_end, double mu_a_step);

	void ProgressData(_process_data_cbk _cbk = NULL);
public:

	virtual void ResetMemory() = 0;

	virtual void LoadParamFile(const string& file_name);

	static void QuickSort(void* data, size_t ElementSize, size_t Elements, compare_cond cond);
	/*	Find the offset of first Element which makes the input callback function 'cond' return 0.

		**NOTE**:	The return value means how many elements offset from data origin, 
					so it must be smaller than the input parameter 'Elements'.
					Meanwhile it also can be a negative value which means can`t find the target element under given condition.*/
	static int BinarySearch(void* data, size_t ElementSize, size_t Elements, const void* target, search_cond cond);

	static bool ReadFileContent(const string& fileName, char*& const content, size_t& size);
public:
	u32				time_range_ps;
	u32				time_resolution;
	double			delta_time_ps;

	Triple<u32>		radius_range_mm;
	Triple<u32>		spatial_resolution;
	Triple<double>	radius_delta_mm;
	UP<u32[]>		_stored_result;
	size_t			_buffer_size;
	_process_data_cbk _cbk;

	// Optical Properties
	double mu_a;
	double mu_s;
	double n;
	double g;
};

#endif // __MONTE_CARLO_BASE

