#include "HMM.h"

#include <algorithm>
#include <numeric>
#include <random>

HMM::~HMM()
{
}

double HMM::Forward(const Vector1D<int> observations)
{
	return 0;
}

std::pair<std::vector<int>, double> HMM::Viterbi(const Vector1D<int>& observations)
{
	return {};
}

void HMM::BaumWelch(Vector1D<int>& observations)
{
}


double PolynomialHMM::Forward(const Vector1D<int> observations)
{
	const int observation_size = observations.size();


	Vector2D<double> forward(observation_size, std::vector<double>(states_size, 0.0));
	//probability matrix  [STATE/OBSERVATION] = PROBABILITY


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
	auto sum = 0.0;
	for (const auto& value : forward)
	{
		for (const double value1 : value)
		{
			sum += value1;
		}
	}
	return sum;
}

std::pair<std::vector<int>, double> PolynomialHMM::Viterbi(const Vector1D<int>& observations)
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
auto PolynomialHMM::ForwardPass(const Vector1D<int>& observations) const
{
	const int observation_size = observations.size();

	// Forward
	Vector2D<double> forward(observation_size, Vector1D<double>(states_size, 0.0));
	for (int i = 0; i < states_size; ++i)
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


auto PolynomialHMM::BackwardPass(const Vector1D<int>& observations) const
{
	const int observation_size = observations.size();

	Vector2D<double> backward(observation_size, Vector1D<double>(states_size, 0.0));
	for (auto i = 0; i < states_size; ++i)
	{
		backward[observation_size - 1][i] = 1;
	}

	for (int t = observation_size - 2; t >= 0; --t)
	{
		for (auto i = 0; i < states_size; ++i)
		{
			Vector1D<double> temp_probability_on_;
			for (auto j = 0; j < states_size; ++j)
			{
				backward[t][i] += trans_prob[i][j] * emission_prob[i][observations[t + 1]] * backward[t + 1][i];
			}
		}
	}
	return backward;
}

auto PolynomialHMM::CalculateGammas(const Vector2D<double>& forward, const Vector2D<double>& backward,
                                    const int observation_size) const
{
	Vector2D<double> gamma(observation_size, Vector1D<double>(states_size, -1.0));


	for (auto t = 0; t < observation_size; ++t)
	{
		for (int i = 0; i < states_size; ++i)
		{
			gamma[t][i] = forward[t][i] * backward[t][i];

			double denominator = 0.0;
			for (auto j = 0; j < states_size; ++j)
			{
				denominator += forward[t][j] * backward[t][j];
			}

			gamma[t][i] /= denominator;
		}
	}
	return gamma;
}

auto PolynomialHMM::CalculateXis(const Vector1D<int>& observations, const Vector2D<double>& forward,
                                 const Vector2D<double>& backward, const int observation_size) const
{
	Vector3D<double> xi(observation_size, Vector2D<double>(states_size, Vector1D<double>(states_size, 0.0)));
	for (int t = 0; t < observation_size - 1; ++t)
	{
		double denominator = 0.0;
		for (int i = 0; i < states_size; ++i)
		{
			for (int j = 0; j < states_size; ++j)
			{
				denominator = forward[t][i] * trans_prob[i][j] * emission_prob[j][observations[t + 1]] * backward[t + 1]
					[j];
			}
		}
		for (int i = 0; i < states_size; ++i)
		{
			for (int j = 0; j < states_size; ++j)
			{
				xi[t][i][j] = forward[t][i] * trans_prob[i][j] * emission_prob[j][observations[t + 1]] * backward[t + 1]
					[j] / denominator;
			}
		}
	}
	return xi;
}


// Execute M-step
void PolynomialHMM::UpdatingInitialProbabilities(const Vector2D<double>& gamma)
{
	for (int i = 0; i < states_size; ++i)
	{
		start_prob[i] = gamma[1][i];
	}
}

void PolynomialHMM::UpdatingTransitionMatrix(const Vector2D<double>& gamma, const Vector3D<double>& xi,
                                             const int observation_size)
{
	for (int i = 0; i < states_size; ++i)
	{
		double denominator = 0.0;
		for (int t = 0; t < observation_size - 1; ++t)
		{
			for (int k = 0; k < states_size; ++k)
			{
				denominator += xi[t][i][k];
			}
		}
		for (int j = 0; j < states_size; ++j)
		{
			double numerator = 0.0;
			for (int t = 0; t < observation_size - 1; ++t)
			{
				numerator += xi[t][i][j];
			}
			trans_prob[i][j] = numerator / denominator;
		}
	}
}

void PolynomialHMM::UpdatingObservationProbabilityMatrix(const Vector1D<int>& observations,
                                                         const Vector2D<double>& gamma, const int observation_size)
{
	for (int i = 0; i < states_size; ++i)
	{
		for (int j = 0; j < n_observations; ++j)
		{
			double numerator = 0.0;
			double denominator = 0.0;
			for (int t = 0; t < observation_size; ++t)
			{
				if (i == j)
				{
					numerator += gamma[t][i];
				}
				denominator += gamma[t][i];
			}
			emission_prob[i][j] = numerator / denominator;
		}
	}
}

// Training
void PolynomialHMM::BaumWelch(Vector1D<int>& observations)
{
	const auto size = static_cast<int>(observations.size());

	// Forward
	const auto forward = ForwardPass(observations);

	// Backward
	const auto backward = BackwardPass(observations);

	//Calculation of posterior probabilities
	const auto gamma = CalculateGammas(forward, backward, size);

	// Xi
	const auto xi = CalculateXis(observations, forward, backward, size);

	// Updating Initial Probabilities
	UpdatingInitialProbabilities(gamma);

	// Updating the transition matrix
	UpdatingTransitionMatrix(gamma, xi, size);

	//Updating the Observation Probability Matrix
	//UpdatingObservationProbabilityMatrix(observations, gamma, size);
}

double PolynomialHMM::ComputeLogLikelihood(const Vector1D<int>& observations) const
{
	auto observation_size = observations.size();
	auto forward = ForwardPass(observations);

	// Последний временной шаг
	int lastTimeStep = observation_size - 1;

	// Вычисление логарифма правдоподобия
	double logLikelihood = -std::log(std::accumulate(forward[lastTimeStep].begin(), forward[lastTimeStep].end(), 0.0));

	return logLikelihood;
}


void PolynomialHMM::Train(int maxIterations, double convergenceThreshold, Vector1D<int>& observations)
{
	int iteration = 0;
	//double prevLogLikelihood = -std::numeric_limits<double>::infinity();

	double prevLogLikelihood = -std::numeric_limits<double>::infinity();

	while (iteration < maxIterations)
	{
		// Выполнение одной итерации обучения
		BaumWelch(observations);

		// Проверка на сходимость
		const double logLikelihood = ComputeLogLikelihood(observations); // Ваш метод для вычисления логарифма правдоподобия
		std::cout << logLikelihood << std::endl;
		if (std::abs(logLikelihood - prevLogLikelihood) < convergenceThreshold)
		{
			std::cout << "Converged after " << iteration + 1 << " iterations.\n";
			break;
		}

		// Обновление предыдущего логарифма правдоподобия
		prevLogLikelihood = logLikelihood;

		// Увеличение счетчика итераций
		++iteration;
	}
}

void PolynomialHMM::Train(int maxIterations, double convergenceThreshold, const Vector2D<int>& observations)
{
	int iteration = 0;
	//double prevLogLikelihood = -std::numeric_limits<double>::infinity();

	double prevLogLikelihood = -std::numeric_limits<double>::infinity();
	bool doing  = true;
	while (iteration < maxIterations && doing)
	{
		for (auto observation : observations)
		{
			BaumWelch(observation);
			double logLikelihood = ComputeLogLikelihood(observation);
			// Ваш метод для вычисления логарифма правдоподобия
			std::cout << logLikelihood << std::endl;
			if (std::abs(logLikelihood - prevLogLikelihood) < convergenceThreshold)
			{
				std::cout << "Converged after " << iteration + 1 << " iterations.\n";
				doing = false;
				break;
			}

			// Обновление предыдущего логарифма правдоподобия
			prevLogLikelihood = logLikelihood;

			// Увеличение счетчика итераций
			++iteration;
		}
		
		// Проверка на сходимость
	}
}

void PolynomialHMM::Train(int maxIterations, double convergenceThreshold, const Vector3D<int>& observations)
{
	int iteration = 0;
	//double prevLogLikelihood = -std::numeric_limits<double>::infinity();

	double prevLogLikelihood = -std::numeric_limits<double>::infinity();

	while (iteration < maxIterations)
	{
		// Выполнение одной итерации обучения
		for (const auto& observation2D : observations)
		{
			for (auto observation : observation2D)
			{
				BaumWelch(observation);

				// Проверка на сходимость
				double logLikelihood = ComputeLogLikelihood(observation);
				// Ваш метод для вычисления логарифма правдоподобия
				std::cout << logLikelihood << std::endl;
				if (std::abs(logLikelihood - prevLogLikelihood) < convergenceThreshold)
				{
					std::cout << "Converged after " << iteration + 1 << " iterations.\n";
					break;
				}

				// Обновление предыдущего логарифма правдоподобия
				prevLogLikelihood = logLikelihood;

				// Увеличение счетчика итераций
				++iteration;
			}
		}
	}
}
