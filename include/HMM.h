#pragma once
#include <numeric>
#include <Vector.h>


namespace vladosHMM
{
	class HMM;

	class VIRTUAL_HMM

	{
	public:
		int n_observations;
		Vector1D<int> states;
		Vector1D<double> start_prob;
		Vector2D<double> trans_prob;
		Vector2D<double> emission_prob;
		int states_size;
		int EPSILON = 1e-9;

	public:
		VIRTUAL_HMM(int n_states, int n_observations, const Vector1D<double>& start_prob = {},
		            const Vector2D<double>& trans_prob = {},
		            const Vector2D<double>& emission_prob = {})
		{
			this->states_size = n_states;
			this->n_observations = n_observations;
			states.resize(n_states); // Creates vector from 1 to n_states
			std::iota(states.begin(), states.end(), 0);

			if (!start_prob.data())
			{
				this->start_prob = GenerateRandomVector(n_states, 0.0+EPSILON, 1.0-EPSILON);
			}
			else
			{
				this->start_prob = start_prob;
			}
			if (!trans_prob.data())
			{
				this->trans_prob = GenerateRandomVector(n_states,n_states, 0.0+EPSILON, 1.0-EPSILON);
			}
			else
			{
				this->trans_prob = trans_prob;
			}
			if (!emission_prob.data())
			{
				this->emission_prob = GenerateRandomVector(n_states,n_observations, 0.0+EPSILON, 1.0-EPSILON);
			}
			else
			{
				this->emission_prob = emission_prob;
			}
		}


		virtual ~VIRTUAL_HMM() = default;


		virtual double Forward(Vector1D<int> observations) = 0;
		virtual std::pair<std::vector<int>, double> Viterbi(const Vector1D<int>& observations) =0;
		virtual void BaumWelch(Vector1D<int>& observations) = 0;
		virtual void Train(const int maxIterations, const double convergenceThreshold, Vector1D<int>& observations) = 0;

	private:
		friend class HMM;
		virtual void UpdatingObservationProbabilityMatrix(const Vector1D<int>& observations,
		                                                  const Vector2D<double>& gamma,
		                                                  int observation_size) = 0;
		virtual void UpdateModelParameters(Vector1D<int>& observations, int size, Vector2D<double> gamma,
		                                   Vector3D<double> xi) = 0;
		virtual void UpdatingTransitionMatrix(const Vector2D<double>& gamma, const Vector3D<double>& xi,
		                                      int observation_size) = 0;

		virtual void UpdatingInitialProbabilities(const Vector2D<double>& gamma) = 0;


		virtual Vector2D<double> ForwardPass(const Vector1D<int>& observations) = 0;
		virtual Vector2D<double> BackwardPass(const Vector1D<int>& observations) = 0;
		virtual Vector2D<double> CalculateGammas(const Vector2D<double>& forward, const Vector2D<double>& backward,
		                                         int observation_size) = 0;
		virtual Vector3D<double> CalculateXis(const Vector1D<int>& observations, const Vector2D<double>& forward,
		                                      const Vector2D<double>& backward, int observation_size) = 0;

		virtual double ComputeLogLikelihood(const Vector1D<int>& observations) = 0;
	};

	// Model For 1D Observations Train !!!
	class HMM final : public VIRTUAL_HMM
	{
	public:
		HMM(const int n_states, const int n_observations,
		    const Vector1D<double>& start_prob = {}, const Vector2D<double>& trans_prob = {},
		    const Vector2D<double>& emission_prob = {})
			: VIRTUAL_HMM(n_states, n_observations, start_prob, trans_prob, emission_prob)
		{
		}

		double Forward(Vector1D<int> observations) override;
		std::pair<std::vector<int>, double> Viterbi(const Vector1D<int>& observations) override;
		void BaumWelch(Vector1D<int>& observations) override;
		void Train(const int maxIterations, const double convergenceThreshold, Vector1D<int>& observations) override;
		double ComputeLogLikelihood(const Vector1D<int>& observations) override;

	private:
		Vector2D<double> ForwardPass(const Vector1D<int>& observations) override;
		Vector2D<double> BackwardPass(const Vector1D<int>& observations) override;
		Vector2D<double> CalculateGammas(const Vector2D<double>& forward, const Vector2D<double>& backward,
		                                 int observation_size) override;
		Vector3D<double> CalculateXis(const Vector1D<int>& observations, const Vector2D<double>& forward,
		                              const Vector2D<double>& backward, int observation_size) override;
		void UpdatingObservationProbabilityMatrix(const Vector1D<int>& observations, const Vector2D<double>& gamma,
		                                          int observation_size) override;
		void UpdateModelParameters(Vector1D<int>& observations, int size, Vector2D<double> gamma,
		                           Vector3D<double> xi) override;
		void UpdatingTransitionMatrix(const Vector2D<double>& gamma, const Vector3D<double>& xi,
		                              int observation_size) override;
		void UpdatingInitialProbabilities(const Vector2D<double>& gamma) override;
	};
}
