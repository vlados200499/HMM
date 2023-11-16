// Импортируем необходимые библиотеки
#pragma once
#include <iostream>
#include <utility>
#include <vector>


template <typename T>
using Vector1D = std::vector<T>;
template <typename T>
using Vector2D = std::vector<std::vector<T>>;
template <typename T>
using Vector3D = std::vector<std::vector<std::vector<T>>>;





class HMM
{
protected:
	Vector1D<int> state;
	Vector1D<int> observation_data;
	Vector1D<double> start_prob; // an initial probability distribution over states. πi is the probability that the Markov chain will start in state i. 
	Vector2D<double> trans_prob; // transition probability matrix ; probability moving from i state -> j state
	Vector2D<double> emission_prob; //a sequence of observation_data likelihoods, also called emission probabilities

public:
	HMM(Vector1D<int> state, Vector1D<double> start_prob, Vector2D<double> trans_prob,
	    Vector2D<double> emission_prob)
	{
		this->state = std::move(state);
		this->start_prob = std::move(start_prob);
		this->trans_prob = std::move(trans_prob);
		this->emission_prob = std::move(emission_prob);
	}


	virtual ~HMM();

	virtual double Forward(const Vector1D<int> observations);

	virtual std::pair<std::vector<int>, std::vector<double>> Viterbi(const Vector1D<int>& observations);

	virtual void BaumWelch(Vector2D<double> observations);

	template <typename T>
	void PrintVector(const Vector1D<T>& vec)
	{
		for (const T& element : vec)
		{
			std::cout << element << " ";
		}
		std::cout << std::endl;
	}

	// Overload for 2D vector
	template <typename T>
	void PrintVector(const Vector2D<T>& vec)
	{
		for (const Vector1D<T>& subVec : vec)
		{
			printVector(subVec);
		}
	}

	// Overload for 3D vector
	template <typename T>
	void PrintVector(const Vector3D<T>& vec)
	{
		for (const Vector2D<T>& subVec : vec)
		{
			printVector(subVec);
		}
	}
};

class PolynomialHMM final : public HMM
{
public:
	PolynomialHMM(const Vector1D<int>& state,
	              const Vector1D<double>& start_prob, const Vector2D<double>& trans_prob,
	              const Vector2D<double>& emission_prob)
		: HMM(state, start_prob, trans_prob, emission_prob)
	{
	}

	double Forward(const Vector1D<int> observations) override;
	std::pair<std::vector<int>, std::vector<double>> Viterbi(const Vector1D<int>& observations) override;
	void BaumWelch(Vector2D<double> observations) override;
};


