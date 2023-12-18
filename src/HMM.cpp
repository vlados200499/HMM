#include "HMM.h"


std::pair<std::vector<int>, double> HMMlearn::HMM::Viterbi(const Vector1D<int>& observations) const
{
	const int observation_size = observations.size();

	Vector2D<double> viterbi(states_size, Vector1D<double>(observation_size, -1));
	Vector2D<int> viterbi_backtrace(states_size, Vector1D<int>(observation_size, -1));


	for (auto s = 0; s < states_size; ++s)
	{
		viterbi[s][0] = start_prob[s] * emission_prob[s][observations[0]];
		viterbi_backtrace[s][0] = 0;
	}

	for (auto t = 1; t < observation_size; ++t)
	{
		for (auto s = 0; s < states_size; ++s)
		{
			Vector1D<double> prob_temp;
			for (auto i = 0; i < states_size; ++i)
			{
				prob_temp.push_back(trans_prob[i][s] * emission_prob[s][observations[t]] * viterbi[i][t - 1]);
			}

			// Find max in all probabilities
			auto max = std::ranges::max_element(prob_temp);
			viterbi[s][t] = *max;

			// Best previous states with maximum possibility 
			viterbi_backtrace[s][t] = static_cast<int>(std::distance(prob_temp.begin(), max));
		}
	}

	const auto best_path_pointer = std::ranges::max_element(viterbi[0]);

	Vector1D<int> best_path(observation_size, 0);

	// Last element of best path . We must use best_path_pointer to get best and last states.
	best_path[observation_size - 1] = static_cast<int>(std::distance(viterbi[0].begin(), best_path_pointer));

	// Back in time. 
	for (int t = observation_size - 2; t >= 0; --t)
	{
		best_path[t] = viterbi_backtrace[best_path[t + 1]][t + 1];
	}

	return std::make_pair(best_path, 0.002);
}


// Execute E-step
Vector2D<double> HMMlearn::HMM::ForwardPass(const Vector1D<int>& observations) const
{
	const int observation_size = observations.size();

	// Forward
	Vector2D<double> forward(observation_size, Vector1D<double>(states_size, 0.0));
	for (auto i = 0; i < states_size; ++i)
	{
		forward[0][i] = start_prob[i] * emission_prob[i][observations[0]];
	}


	for (auto t = 1; t < observation_size; ++t)
	{
		// Target states 
		for (auto i = 0; i < states_size; ++i)
		{
			// int i is - all states in previous layer
			for (auto j = 0; j < states_size; ++j)
			{
				forward[t][i] += trans_prob[i][j] * forward[t - 1][j];
			}
			forward[t][i] *= emission_prob[i][observations[t]];
		}
	}
	return forward;
}


Vector2D<double> HMMlearn::HMM::BackwardPass(const Vector1D<int>& observations) const
{
	const int observation_size = observations.size();

	Vector2D<double> backward(observation_size, Vector1D<double>(states_size, 1e-9));
	for (auto i = 0; i < states_size; ++i)
	{
		backward[observation_size - 1][i] = 1;
	}

	for (int t = observation_size - 2; t >= 0; --t)
	{
		for (auto i = 0; i < states_size; ++i)
		{
			double sum = 0;
			for (auto j = 0; j < states_size; ++j)
			{
				sum += trans_prob[i][j] * emission_prob[j][observations[t + 1]] * backward[t + 1][j];
			}
			backward[t][i] += sum;
		}
	}
	return backward;
}

Vector2D<double> HMMlearn::HMM::CalculateGammas(const Vector2D<double>& forward, const Vector2D<double>& backward,
                                                const int observation_size) const
{
	Vector2D<double> gamma(observation_size, Vector1D<double>(states_size, 1e-9));


	for (auto t = 0; t < observation_size; ++t)
	{
		auto denominator = 0.0;

		for (auto i = 0; i < states_size; ++i)
		{
			gamma[t][i] = forward[t][i] * backward[t][i];
			denominator += forward[t][i] * backward[t][i];
		}
		if (denominator == 0.0)
		{
			throw;
		}
		for (int i = 0; i < states_size; ++i)
		{
			gamma[t][i] /= denominator;
		}
	}
	return gamma;
}

Vector3D<double> HMMlearn::HMM::CalculateXis(const Vector1D<int>& observations, const Vector2D<double>& forward,
                                             const Vector2D<double>& backward, const int observation_size) const
{
	Vector3D<double> xi(observation_size - 1, Vector2D<double>(states_size, Vector1D<double>(states_size, 1e-9)));
	for (auto t = 0; t < observation_size - 1; ++t)
	{
		auto denominator = 0.0;
		for (auto i = 0; i < states_size; ++i)
		{
			for (auto j = 0; j < states_size; ++j)
			{
				denominator += forward[t][i] * trans_prob[i][j] * emission_prob[j][observations[t + 1]] * backward[t +
					1][j];
			}
		}
		if (denominator == 0.0)
		{
			throw;
		}
		for (int i = 0; i < states_size; ++i)
		{
			for (auto j = 0; j < states_size; ++j)
			{
				xi[t][i][j] = forward[t][i] * trans_prob[i][j] * emission_prob[j][observations[t + 1]] * backward[t + 1]
					[j] / denominator;
			}
		}
	}
	return xi;
}


// Execute M-step
void HMMlearn::HMM::UpdatingInitialProbabilities(const Vector2D<double>& gamma)
{
	for (auto i = 0; i < states_size; ++i)
	{
		start_prob[i] = gamma[1][i];
	}
}

void HMMlearn::HMM::UpdatingTransitionMatrix(const Vector2D<double>& gamma, const Vector3D<double>& xi,
                                             const int observation_size)
{
	for (auto i = 0; i < states_size; ++i)
	{
		for (auto j = 0; j < states_size; ++j)
		{
			auto denominator = 0.0;
			auto numerator = 0.0;
			for (auto t = 0; t < observation_size - 1; ++t)
			{
				denominator += gamma[t][i];
				numerator += xi[t][i][j];
			}
			if (denominator == 0.0 || numerator == 0.0 )
			{
					throw;
			}
			trans_prob[i][j] += numerator / denominator;
		}
	}
	//NormalizeVector(&trans_prob);
}

void HMMlearn::HMM::UpdatingObservationProbabilityMatrix(const Vector1D<int>& observations,
                                                         const Vector2D<double>& gamma, const int observation_size)
{
	for (auto i = 0; i < states_size; ++i)
	{
		for (auto j = 0; j < n_observations; ++j)
		{
			auto numerator = 0.0;
			auto denominator = 0.0;
			for (auto t = 0; t < observation_size; ++t)
			{
				if (observations[t] == j)
				{
					numerator += gamma[t][i];
				}
				denominator += gamma[t][i];
				if (denominator == 0.0)
				{
					throw;
				}
			}
			emission_prob[i][j] = numerator / denominator;
		}
	}
}

void HMMlearn::HMM::UpdateModelParameters(const Vector1D<int>& observations, const int size, const Vector2D<double>&
                                          gamma,
                                          const Vector3D<double>& xi)
{
	// Updating Initial Probabilities
	UpdatingInitialProbabilities(gamma);

	// Updating the transition matrix
	UpdatingTransitionMatrix(gamma, xi, size);

	//Updating the Observation Probability Matrix
	UpdatingObservationProbabilityMatrix(observations, gamma, size);
}


// Training
void HMMlearn::HMM::BaumWelch(const Vector1D<int>& observations)
{
	const auto size = observations.size();

	// Forward
	auto forward = ForwardPass(observations);

	// Backward
	NormalizeVector(&forward);


	auto backward = BackwardPass(observations);
	//NormalizeVector(&backward);

	//Calculation of posterior probabilities
	auto gamma = CalculateGammas(forward, backward, size);
	//NormalizeVector(&gamma);
	// Xi
	const auto xi = CalculateXis(observations, forward, backward, size);

	UpdateModelParameters(observations, size, gamma, xi);
}


double HMMlearn::HMM::Loss(const Vector1D<int>& observations) const
{
	const auto observation_size = observations.size();
	const auto forward = ForwardPass(observations);

	double log_likelihood = 0.0;
	for (int state = 0; state < states_size; ++state)
	{
		log_likelihood += forward[observation_size - 1][state];
	}

	return -log(log_likelihood) / n_observations;
}

void HMMlearn::HMM::Train(const int maxIterations, const double convergenceThreshold, const Vector1D<int>& observations)
{
	auto iteration = 0;
	double prevLoss = std::numeric_limits<double>::infinity();

	while (iteration < maxIterations)
	{
		// Perform one iteration of training using the Baum-Welch algorithm
		BaumWelch(observations);

		// Compute the negative log-likelihood (loss)
		const double loss = Loss(observations);
		std::cout << " Loss: " << loss << std::endl;

		// Check for convergence
		if (iteration > 0 && std::abs(prevLoss - loss) < convergenceThreshold)
		{
			std::cout << "Converged after " << iteration + 1 << " iterations.\n";
			break;
		}

		// Update the previous loss for the next iteration
		prevLoss = loss;

		//Increase iteration counter
		++iteration;
	}
}
