#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric> // Для std::accumulate
template <typename T>
using Vector1D = std::vector<T>;
template <typename T>
using Vector2D = std::vector<std::vector<T>>;
template <typename T>
using Vector3D = std::vector<std::vector<std::vector<T>>>;


#include <array>


template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr)
{
	os << "[";
	for (std::size_t i = 0; i < N; ++i)
	{
		os << arr[i];
		if (i != N - 1)
		{
			os << ", ";
		}
	}
	os << "]";
	os << std::endl;

	return os;
}


template <typename T, std::size_t N>
class Array1D : public std::array<T, N>
{
public:
	void Fill(const T& value)
	{
		std::fill(this->begin(), this->end(), value);
	}

	void Random(T min, T max)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(min, max);

		for (auto& element : *this)
		{
			element = dis(gen);
		}
	}

	void SoftMax()
	{
		T sum = 0;
		for (auto& element : *this)
		{
			element = std::exp(element);
			sum += element;
		}
		for (auto& element : *this)
		{
			element /= sum;
		}
	}
};

template <typename T, std::size_t Rows, std::size_t Cols>
class Array2D : public std::array<std::array<T, Cols>, Rows>
{
public:
	void Fill(const T& value)
	{
		for (auto& row : *this)
		{
			row.fill(value);
		}
	}

	void SoftMax()
	{
		for (auto& row : *this)
		{
			T row_sum = 0;
			for (auto& element : row)
			{
				element = std::exp(element);
				row_sum += element;
			}
			for (auto& element : row)
			{
				element /= row_sum;
			}
		}
	}


	void Random(T min, T max)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(min, max);

		for (auto& row : *this)
		{
			for (auto& element : row)
			{
				element = dis(gen);
			}
		}
	}
};

template <typename T, std::size_t Dim1, std::size_t Dim2, std::size_t Dim3>
class Array3D : public std::array<std::array<std::array<T, Dim3>, Dim2>, Dim1>
{
public:
	void Fill(const T& value)
	{
		for (auto& plane : *this)
		{
			for (auto& row : plane)
			{
				row.fill(value);
			}
		}
	}

	void SoftMax()
	{
		for (auto& plane : *this)
		{
			for (auto& row : plane)
			{
				T row_sum = 0;
				for (auto& element : row)
				{
					element = std::exp(element);
					row_sum += element;
				}
				for (auto& element : row)
				{
					element /= row_sum;
				}
			}
		}
	}

	void Random(T min, T max)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(min, max);

		for (auto& plane : *this)
		{
			for (auto& row : plane)
			{
				for (auto& element : row)
				{
					element = dis(gen);
				}
			}
		}
	}
};

template <typename T>
void PrintVector(const Vector1D<T>& vector)
{
	std::cout << "[";
	for (int i = 0; i < vector.size(); ++i)
	{
		std::cout << vector[i];
		if (i < vector.size() - 1)
		{
			std::cout << " ";
		}
	}
	std::cout << "]";
	std::cout << std::endl;
}

template <typename T>
void PrintVector(const Vector2D<T>& vector)
{
	for (auto data : vector)
	{
		PrintVector(data);
	}
}

template <typename T>
void PrintVector(const Vector3D<T>& vector)
{
	for (auto data : vector)
	{
		PrintVector(data);
	}
}


template <typename T>
T GenerateRandomNumber(T start, T end)
{
	std::random_device rd;
	std::mt19937 rng(rd());
	if constexpr (std::is_integral_v<T>)
	{
		std::uniform_int_distribution<T> distribution(start, end);
		return distribution(rng);
	}
	else if constexpr (std::is_floating_point_v<T>)
	{
		std::uniform_real_distribution<T> distribution(start, end);
		return distribution(rng);
	}
	else
	{
		static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
		              "Invalid type for GenerateRandomNumber");
	}
	return {};
}


template <typename T>
Vector1D<T> GenerateRandomVector(int row, T min, T max)
{
	std::vector<T> tempVector(row);
	std::ranges::generate(tempVector, [=]
	{
		return GenerateRandomNumber(min, max);
	});
	return tempVector;
};

template <typename T>
Vector2D<T> GenerateRandomVector(int row, int column, T min, T max)
{
	Vector2D<T> tempVector(row, Vector1D<T>(column));
	for (auto& vector : tempVector)
	{
		std::ranges::generate(vector, [=]
		{
			return GenerateRandomNumber(min, max);
		});
	}
	return tempVector;
};

template <typename T>
Vector3D<T> GenerateRandomVector(int row, int column, int size_z, T min, T max)
{
	Vector3D<T> tempVector(row, Vector1D<Vector1D<T>>(column, Vector1D<T>(size_z)));
	for (auto& matrix : tempVector)
	{
		for (auto& row : matrix)
		{
			std::ranges::generate(row, [=]()
			{
				return GenerateRandomNumber(min, max);
			});
		}
	}

	return tempVector;
};

template <typename T>
void NormalizeVector(Vector1D<T>* veс)
{
	if (!veс || veс->empty())
	{
		std::cerr << "Error: Invalid vector or empty vector." << std::endl;
		return;
	}

	const double sum = std::accumulate(veс->begin(), veс->end(), 0.0);

	for (double& elem : *veс)
	{
		elem /= sum;
	}
}


template <typename T>
void NormalizeVector(Vector2D<T>* veс)
{
	for (auto& row : *veс)
	{
		NormalizeVector(&row);
	}
}
