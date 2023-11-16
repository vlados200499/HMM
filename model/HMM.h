// Импортируем необходимые библиотеки
#pragma once
#include <iostream>
#include <numeric>
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
	int n_states;
	int n_observations;
	Vector1D<int> states;
	//Vector1D<int> observation_data;
	Vector1D<double> start_prob;
	// an initial probability distribution over states. πi is the probability that the Markov chain will start in states i. 
	Vector2D<double> trans_prob; // transition probability matrix ; probability moving from i states -> j states
	Vector2D<double> emission_prob; //a sequence of observation_data likelihoods, also called emission probabilities
	int states_size;

public:
	HMM(int n_states,int n_observations, const Vector1D<double>& start_prob = {}, const Vector2D<double>& trans_prob = {},
	    const Vector2D<double>& emission_prob = {})
	{
		this->n_states = n_states;
		states.resize(n_states); // Creates vector from 1 to n_states
		std::iota(states.begin(), states.end(), 0);
		this->start_prob = start_prob;
		this->trans_prob = trans_prob;
		this->emission_prob = emission_prob;
		states_size = n_states;
		this->n_observations = n_observations;
	}


	virtual ~HMM();

	virtual double Forward(Vector1D<int> observations);

	virtual std::pair<std::vector<int>, double> Viterbi(const Vector1D<int>& observations);

	virtual void BaumWelch(Vector1D<int>& observations);




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
	PolynomialHMM(int n_states,int n_observations,
	              const Vector1D<double>& start_prob, const Vector2D<double>& trans_prob,
	              const Vector2D<double>& emission_prob)
		: HMM(n_states,n_observations, start_prob, trans_prob, emission_prob)
	{
	}

	double Forward(Vector1D<int> observations) override;
	std::pair<std::vector<int>, double> Viterbi(const Vector1D<int>& observations) override;
	void BaumWelch(Vector1D<int>& observations) override;
	void Train(int maxIterations, double convergenceThreshold, Vector1D<int>& observations);
	void Train(int maxIterations, double convergenceThreshold, const Vector2D<int>& observations);
	void Train(int maxIterations, double convergenceThreshold, const Vector3D<int>& observations);

private:
	double ComputeLogLikelihood(const Vector1D<int>& observations) const;

	auto ForwardPass(const Vector1D<int>& observations) const;
	auto BackwardPass(const Vector1D<int>& observations) const;
	auto CalculateGammas(const Vector2D<double>& forward, const Vector2D<double>& backward, int observation_size) const;
	auto CalculateXis(const Vector1D<int>& observations, const Vector2D<double>& forward,const Vector2D<double>& backward, int observation_size) const;
	void UpdatingObservationProbabilityMatrix(const Vector1D<int>& observations, const Vector2D<double>& gamma,int observation_size);
	void UpdatingTransitionMatrix(const Vector2D<double>& gamma, const Vector3D<double>& xi, int observation_size);
	void UpdatingInitialProbabilities(const Vector2D<double>& gamma);


	

public:
};
