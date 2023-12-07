#pragma once
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
template <typename T>
using Vector1D = std::vector<T>;
template <typename T>
using Vector2D = std::vector<std::vector<T>>;
template <typename T>
using Vector3D = std::vector<std::vector<std::vector<T>>>;

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
