#pragma once
#include <numeric>
#include <Vector.h>

namespace HMMlearn
{
	class HMM;


	// Model For 1D Observations Train !!!
	class HMM final
	{
	public:
		int n_observations;
		Vector1D<int> states;
		Vector1D<double> start_prob;
		Vector2D<double> trans_prob;
		Vector2D<double> emission_prob;
		int states_size;
		double epsilon = 1e-3;


		HMM(const int n_states, const int n_observations,
		    const Vector1D<double>& start_prob = {}, const Vector2D<double>& trans_prob = {},
		    const Vector2D<double>& emission_prob = {})
		{
			this->states_size = n_states;
			this->n_observations = n_observations;
			states.resize(n_states); // Creates vector from 1 to n_states
			std::iota(states.begin(), states.end(), 0);

			if (!start_prob.data())
			{
				this->start_prob = GenerateRandomVector(n_states, 0.0, 1.0);
				NormalizeVector(&this->start_prob);
			}
			else
			{
				this->start_prob = start_prob;
			}
			if (!trans_prob.data())
			{
				this->trans_prob = GenerateRandomVector(n_states, n_states, 0.0, 1.0);
				NormalizeVector(&this->trans_prob);
			}
			else
			{
				this->trans_prob = trans_prob;
			}
			if (!emission_prob.data())
			{
				this->emission_prob = GenerateRandomVector(n_states, n_observations, 0.0, 1.0);
				NormalizeVector(&this->emission_prob);
			}
			else
			{
				this->emission_prob = emission_prob;
			}
		}


		std::pair<std::vector<int>, double> Viterbi(const Vector1D<int>& observations) const;
		void BaumWelch(const Vector1D<int>& observations);
		void Train(int maxIterations, double convergenceThreshold, const Vector1D<int>& observations);
		double Loss(const Vector1D<int>& observations) const;

	private:
		Vector2D<double> ForwardPass(const Vector1D<int>& observations) const;
		Vector2D<double> BackwardPass(const Vector1D<int>& observations) const;
		Vector2D<double> CalculateGammas(const Vector2D<double>& forward, const Vector2D<double>& backward,
		                                 int observation_size) const;
		Vector3D<double> CalculateXis(const Vector1D<int>& observations, const Vector2D<double>& forward,
		                              const Vector2D<double>& backward, int observation_size) const;
		void UpdatingObservationProbabilityMatrix(const Vector1D<int>& observations, const Vector2D<double>& gamma,
		                                          int observation_size);
		void UpdateModelParameters(const Vector1D<int>& observations, int size, const Vector2D<double>& gamma,
		                           const Vector3D<double>& xi);
		void UpdatingTransitionMatrix(const Vector2D<double>& gamma, const Vector3D<double>& xi,
		                              int observation_size);
		void UpdatingInitialProbabilities(const Vector2D<double>& gamma);
	};
}
