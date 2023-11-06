#include "HMM.h"

#include <numeric>


HMM::~HMM()
{
}

double HMM::Forward(const Vector1D<int> observations)
{
	return 0;
}

std::pair<std::vector<int>, std::vector<double>> HMM::Viterbi(const Vector1D<int>& observations)
{
	return {};
}

void HMM::BaumWelch(Vector1D<double> observations)
{

}

double PolynomialHMM::Forward(const Vector1D<int> observations)
{
	const int observation_size = observations.size();
	const int state_size = state.size();


	Vector2D<double> forward(state_size, std::vector<double>(observation_size, 0.0));
	//probability matrix  [STATE/OBSERVATION] = PROBABILITY


	for (int i = 0; i < state_size; ++i)
	{
		forward[i][0] = start_prob[i] * emission_prob[i][observations[0]];
	}


	for (int t = 1; t < observation_size; ++t)
	{
		// Target state 
		for (int s = 0; s < state_size; ++s)
		{
			// int i is - all states in previous layer
			for (int i = 0; i < state_size; ++i)
			{
				forward[s][t] += trans_prob[i][s] * emission_prob[s][observations[t]] * forward[i][t - 1];
			}
		}
	}
	double sum = 0.0;
	for (const auto& value : forward)
	{
		for (const double value1 : value)
		{
			sum += value1;
		}
	}
	return sum;
}


std::pair<std::vector<int>, std::vector<double>> PolynomialHMM::Viterbi(const Vector1D<int>& observations)
{
	const int observation_size = observations.size();
	const int state_size = state.size();

	Vector2D<double> viterbi(state_size, Vector1D<double>(observation_size, -1));
	Vector2D<int> viterbi_backtrace(state_size, Vector1D<int>(observation_size, -1));


	for (int s = 0; s < state_size; ++s)
	{
		viterbi[s][0] = start_prob[s] * emission_prob[s][observations[0]];
		viterbi_backtrace[s][0] = 0;
	}

	for (int t = 1; t < observation_size; ++t)
	{
		for (int s = 0; s < state_size; ++s)
		{
			Vector1D<double> prob_temp;
			for (int i = 0; i < state_size; ++i)
			{
				prob_temp.push_back(trans_prob[i][s] * emission_prob[s][observations[t]] * viterbi[i][t - 1]);
			}

			// Find max in all probabilities
			auto max = std::ranges::max_element(prob_temp);
			viterbi[s][t] = *max;

			// Best previous state with maximum possibility 
			viterbi_backtrace[s][t] = static_cast<int>(std::distance(prob_temp.begin(), max));
		}
	}

	const auto best_path_pointer = std::ranges::max_element(viterbi[0]);
	Vector1D<double> best_path_prob(observation_size,0);

	Vector1D<int> best_path(observation_size,0);

	// Last element of best path . We must use best_path_pointer to get best and last state.
	best_path[observation_size-1] = static_cast<int>(std::distance(viterbi[0].begin(), best_path_pointer));
	//best_path_prob[observation_size-1] = *best_path_pointer;

	// Back in time . 
	for (int t = observation_size-2; t >= 0; --t)
	{
		best_path[t] = viterbi_backtrace[best_path[t+1]][t+1];
		//best_path_prob[t] = viterbi[t];
	}

	return std::make_pair(best_path , best_path_prob);
}
