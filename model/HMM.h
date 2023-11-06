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

	virtual void BaumWelch(Vector1D<double> observations);


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
};


//// Определяем подкласс DiscreteHMM, который наследует от класса HMM и реализует дискретную HMM модель
//class DiscreteHMM : public HMM
//{
//public:
//	// Конструктор класса DiscreteHMM
//	DiscreteHMM(int n_states, int n_symbols, vector<double> start_prob, vector<vector<double>> trans_prob,
//	            vector<vector<double>> emis_prob) : HMM(n_states, n_symbols, start_prob, trans_prob, emis_prob)
//	{
//		
//	}
//
//	// Деструктор класса DiscreteHMM
//	~DiscreteHMM() override
//	{
//	}
//
//	// Метод для генерации случайной последовательности наблюдений и соответствующей последовательности состояний
//	pair<vector<int>, vector<int>> sample(int length) override
//	{
//		// Инициализируем вектора для хранения наблюдений и состояний
//		vector<int> observations;
//		vector<int> states;
//
//		// Выбираем начальное состояние согласно начальным вероятностям
//		int state = 0;
//		double r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//		double p = 0.0;
//		for (int i = 0; i < n_states; i++)
//		{
//			p += start_prob[i];
//			if (r < p)
//			{
//				state = i;
//				break;
//			}
//		}
//
//		// Добавляем начальное состояние в вектор состояний
//		states.push_back(state);
//
//		// Генерируем наблюдение согласно эмиссионным вероятностям
//		int observation_data = 0;
//		r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//		p = 0.0;
//		for (int i = 0; i < n_symbols; i++)
//		{
//			p += emis_prob[state][i];
//			if (r < p)
//			{
//				observation_data = i;
//				break;
//			}
//		}
//
//		// Добавляем наблюдение в вектор наблюдений
//		observations.push_back(observation_data);
//
//		// Повторяем процесс для оставшихся элементов последовательности
//		for (int t = 1; t < length; t++)
//		{
//			// Переходим в новое состояние согласно переходным вероятностям
//			int new_state = 0;
//			r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//			p = 0.0;
//			for (int i = 0; i < n_states; i++)
//			{
//				p += trans_prob[state][i];
//				if (r < p)
//				{
//					new_state = i;
//					break;
//				}
//			}
//
//			// Добавляем новое состояние в вектор состояний
//			states.push_back(new_state);
//
//			// Генерируем наблюдение согласно эмиссионным вероятностям
//			int new_observation = 0;
//			r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//			p = 0.0;
//			for (int i = 0; i < n_symbols; i++)
//			{
//				p += emis_prob[new_state][i];
//				if (r < p)
//				{
//					new_observation = i;
//					break;
//				}
//			}
//
//			// Добавляем наблюдение в вектор наблюдений
//			observations.push_back(new_observation);
//
//			// Обновляем текущее состояние
//			state = new_state;
//		}
//
//		// Возвращаем пару из векторов наблюдений и состояний
//		return make_pair(observations, states);
//	}
//
//	// Метод для вычисления вероятности наблюдений с помощью алгоритма Forward-Backward
//	double forward_backward(vector<int> observations) override
//	{
//		// Получаем длину последовательности наблюдений
//		int length = observations.size();
//
//		// Инициализируем матрицы для хранения прямых и обратных вероятностей
//		vector forward_prob(n_states, vector<double>(length));
//		vector backward_prob(n_states, vector<double>(length));
//
//		// Вычисляем прямые вероятности для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//		for (int t = 0; t < length; t++)
//		{
//			for (int i = 0; i < n_states; i++)
//			{
//				if (t == 0)
//				{
//					// Инициализируем прямые вероятности для первого момента времени согласно начальным вероятностям и эмиссионным вероятностям
//					forward_prob[i][t] = start_prob[i] * emis_prob[i][observations[t]];
//				}
//				else
//				{
//					// Вычисляем прямые вероятности для остальных моментов времени с учетом предыдущих прямых вероятностей и переходных вероятностей
//					double sum = 0.0;
//					for (int j = 0; j < n_states; j++)
//					{
//						sum += forward_prob[j][t - 1] * trans_prob[j][i];
//					}
//					forward_prob[i][t] = sum * emis_prob[i][observations[t]];
//				}
//			}
//		}
//
//		// Вычисляем обратные вероятности для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//		for (int t = length - 1; t >= 0; t--)
//		{
//			for (int i = 0; i < n_states; i++)
//			{
//				if (t == length - 1)
//				{
//					// Инициализируем обратные вероятности для последнего момента времени равными единице
//					backward_prob[i][t] = 1.0;
//				}
//				else
//				{
//					// Вычисляем обратные вероятности для остальных моментов времени с учетом следующих обратных вероятностей и переходных вероятностей
//					double sum = 0.0;
//					for (int j = 0; j < n_states; j++)
//					{
//						sum += backward_prob[j][t + 1] * trans_prob[i][j] * emis_prob[j][observations[t + 1]];
//					}
//					backward_prob[i][t] = sum;
//				}
//			}
//		}
//
//		// Вычисляем вероятность наблюдений как сумму произведений прямых и обратных вероятностей по всем состояниям в любой момент времени
//		double prob = 0.0;
//		for (int i = 0; i < n_states; i++)
//		{
//			prob += forward_prob[i][0] * backward_prob[i][0];
//		}
//
//		// Возвращаем вероятность наблюдений
//		return prob;
//	}
//
//	// Функция для проверки сходимости параметров модели
//
//	// Метод для нахождения наиболее вероятной последовательности состояний с помощью алгоритма Витерби
//	vector<int> viterbi(vector<int> observations) override
//	{
//		// Получаем длину последовательности наблюдений
//		int length = observations.size();
//
//		// Инициализируем матрицу для хранения оптимальных вероятностей
//		vector opt_prob(n_states, vector<double>(length));
//
//		// Инициализируем матрицу для хранения оптимальных предшественников
//		vector opt_pred(n_states, vector<int>(length));
//
//		// Вычисляем оптимальные вероятности и предшественники для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//		for (int t = 0; t < length; t++)
//		{
//			for (int i = 0; i < n_states; i++)
//			{
//				if (t == 0)
//				{
//					// Инициализируем оптимальные вероятности и предшественники для первого момента времени согласно начальным вероятностям и эмиссионным вероятностям
//					opt_prob[i][t] = start_prob[i] * emis_prob[i][observations[t]];
//					opt_pred[i][t] = -1; // Отсутствие предшественника
//				}
//				else
//				{
//					// Вычисляем оптимальные вероятности и предшественники для остальных моментов времени с учетом предыдущих оптимальных вероятностей и переходных вероятностей
//					double max_prob = 0.0;
//					int max_pred = -1;
//					for (int j = 0; j < n_states; j++)
//					{
//						double prob = opt_prob[j][t - 1] * trans_prob[j][i] * emis_prob[i][observations[t]];
//						if (prob > max_prob)
//						{
//							max_prob = prob;
//							max_pred = j;
//						}
//					}
//					opt_prob[i][t] = max_prob;
//					opt_pred[i][t] = max_pred;
//				}
//			}
//		}
//
//		// Инициализируем вектор для хранения наиболее вероятной последовательности состояний
//		vector<int> states;
//
//		// Находим конечное состояние с максимальной оптимальной вероятностью
//		int state = 0;
//		double max_prob = 0.0;
//		for (int i = 0; i < n_states; i++)
//		{
//			if (opt_prob[i][length - 1] > max_prob)
//			{
//				max_prob = opt_prob[i][length - 1];
//				state = i;
//			}
//		}
//
//		// Добавляем конечное состояние в вектор состояний
//		states.push_back(state);
//
//		// Обратно восстанавливаем наиболее вероятную последовательность состояний с помощью матрицы оптимальных предшественников
//		for (int t = length - 2; t >= 0; t--)
//		{
//			state = opt_pred[state][t + 1];
//			states.push_back(state);
//		}
//
//		// Разворачиваем вектор состояний, чтобы получить правильный порядок
//		reverse(states.begin(), states.end());
//
//		// Возвращаем вектор состояний
//		return states;
//	}
//
//	bool check_convergence(vector<vector<double>> forward_prob, vector<vector<double>> trans_prob_new,
//	                       vector<vector<double>> emis_prob_new)
//	{
//		// Задаем некоторый порог для сходимости
//		double threshold = 0.001;
//
//		// Вычисляем разницу между старыми и новыми параметрами модели
//		double diff = 0.0;
//		for (int i = 0; i < n_states; i++)
//		{
//			diff += abs(start_prob[i] - forward_prob[0][i]);
//			for (int j = 0; j < n_states; j++)
//			{
//				diff += abs(trans_prob[i][j] - trans_prob_new[i][j]);
//			}
//			for (int k = 0; k < n_symbols; k++)
//			{
//				diff += abs(emis_prob[i][k] - emis_prob_new[i][k]);
//			}
//		}
//
//		// Сравниваем разницу с порогом
//		if (diff < threshold)
//		{
//			return true; // Параметры модели сходятся
//		}
//		return false; // Параметры модели не сходятся
//	}
//
//	// Метод для обучения параметров модели с помощью алгоритма Baum-Welch
//	void baum_welch(vector<int> observations) override
//	{
//		// Получаем длину последовательности наблюдений
//		int length = observations.size();
//
//		// Инициализируем матрицы для хранения прямых и обратных вероятностей
//		vector forward_prob(n_states, vector<double>(length));
//		vector backward_prob(n_states, vector<double>(length));
//
//		// Инициализируем матрицы для хранения вероятностей переходов и эмиссий
//		vector trans_prob_new(n_states, vector<double>(n_states));
//		vector emis_prob_new(n_states, vector<double>(n_symbols));
//
//		// Повторяем процесс до сходимости параметров модели
//		bool converged = false;
//		while (!converged)
//		{
//			// Вычисляем прямые вероятности для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//			for (int t = 0; t < length; t++)
//			{
//				for (int i = 0; i < n_states; i++)
//				{
//					if (t == 0)
//					{
//						// Инициализируем прямые вероятности для первого момента времени согласно начальным вероятностям и эмиссионным вероятностям
//						forward_prob[i][t] = start_prob[i] * emis_prob[i][observations[t]];
//					}
//					else
//					{
//						// Вычисляем прямые вероятности для остальных моментов времени с учетом предыдущих прямых вероятностей и переходных вероятностей
//						double sum = 0.0;
//						for (int j = 0; j < n_states; j++)
//						{
//							sum += forward_prob[j][t - 1] * trans_prob[j][i];
//						}
//						forward_prob[i][t] = sum * emis_prob[i][observations[t]];
//					}
//				}
//			}
//
//			// Вычисляем обратные вероятности для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//			for (int t = length - 1; t >= 0; t--)
//			{
//				for (int i = 0; i < n_states; i++)
//				{
//					if (t == length - 1)
//					{
//						// Инициализируем обратные вероятности для последнего момента времени равными единице
//						backward_prob[i][t] = 1.0;
//					}
//					else
//					{
//						// Вычисляем обратные вероятности для остальных моментов времени с учетом следующих обратных вероятностей и переходных вероятностей
//						double sum = 0.0;
//						for (int j = 0; j < n_states; j++)
//						{
//							sum += backward_prob[j][t + 1] * trans_prob[i][j] * emis_prob[j][observations[t + 1]];
//						}
//						backward_prob[i][t] = sum;
//					}
//				}
//			}
//
//			// Вычисляем вероятность наблюдений как сумму произведений прямых и обратных вероятностей по всем состояниям в любой момент времени
//			double prob = 0.0;
//			for (int i = 0; i < n_states; i++)
//			{
//				prob += forward_prob[i][0] * backward_prob[i][0];
//			}
//
//			// Вычисляем ожидаемое количество переходов из каждого состояния в каждое другое состояние с учетом наблюдений и текущих параметров модели
//			for (int i = 0; i < n_states; i++)
//			{
//				for (int j = 0; j < n_states; j++)
//				{
//					double num = 0.0;
//					double den = 0.0;
//					for (int t = 0; t < length - 1; t++)
//					{
//						num += forward_prob[i][t] * trans_prob[i][j] * emis_prob[j][observations[t + 1]] * backward_prob
//							[j][t + 1];
//						den += forward_prob[i][t] * backward_prob[i][t];
//					}
//					trans_prob_new[i][j] = num / den;
//				}
//			}
//
//			// Вычисляем ожидаемое количество эмиссий каждого символа из каждого состояния с учетом наблюдений и текущих параметров модели
//			for (int i = 0; i < n_states; i++)
//			{
//				for (int k = 0; k < n_symbols; k++)
//				{
//					double num = 0.0;
//					double den = 0.0;
//					for (int t = 0; t < length; t++)
//					{
//						if (observations[t] == k)
//						{
//							num += forward_prob[i][t] * backward_prob[i][t];
//						}
//						den += forward_prob[i][t] * backward_prob[i][t];
//					}
//					emis_prob_new[i][k] = num / den;
//				}
//			}
//
//			// Обновляем параметры модели согласно вычисленным ожиданиям
//			start_prob = forward_prob[0];
//			trans_prob = trans_prob_new;
//			emis_prob = emis_prob_new;
//
//			// Проверяем сходимость параметров модели с помощью некоторого критерия остановки
//			converged = check_convergence(forward_prob, trans_prob_new, emis_prob_new);
//			// Функция для проверки сходимости
//		}
//	}
//
//	DiscreteHMM() = default;
//};
//
//class PolynomialHMM  : public DiscreteHMM
//{
//public:
//	PolynomialHMM(int n_states, int n_symbols, vector<double> start_prob, vector<vector<double>> trans_prob,
//	              vector<vector<double>> emis_prob) : DiscreteHMM(n_states, n_symbols, start_prob, trans_prob,
//	                                                              emis_prob)
//	{
//	}
//
//	pair<vector<int>, vector<int>> sample(int length) override
//	{
//		// Инициализируем вектора для хранения наблюдений и состояний
//		vector<int> observations;
//		vector<int> states;
//
//		// Выбираем начальное состояние согласно начальным вероятностям
//		int state = 0;
//		double r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//		double p = 0.0;
//		for (int i = 0; i < n_states; i++)
//		{
//			p += start_prob[i];
//			if (r < p)
//			{
//				state = i;
//				break;
//			}
//		}
//
//		// Добавляем начальное состояние в вектор состояний
//		states.push_back(state);
//
//		// Генерируем наблюдение согласно полиномиальному распределению
//		int observation_data = 0;
//		r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//		p = 0.0;
//		for (int i = 0; i < n_symbols; i++)
//		{
//			p += pow(emis_prob[state][i], observations[i]);
//			if (r < p)
//			{
//				observation_data = i;
//				break;
//			}
//		}
//
//		// Добавляем наблюдение в вектор наблюдений
//		observations.push_back(observation_data);
//
//		// Повторяем процесс для оставшихся элементов последовательности
//		for (int t = 1; t < length; t++)
//		{
//			// Переходим в новое состояние согласно переходным вероятностям
//			int new_state = 0;
//			r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//			p = 0.0;
//			for (int i = 0; i < n_states; i++)
//			{
//				p += trans_prob[state][i];
//				if (r < p)
//				{
//					new_state = i;
//					break;
//				}
//			}
//
//			// Добавляем новое состояние в вектор состояний
//			states.push_back(new_state);
//
//			// Генерируем наблюдение согласно полиномиальному распределению
//			int new_observation = 0;
//			r = static_cast<double>(rand()) / RAND_MAX; // Случайное число от 0 до 1
//			p = 0.0;
//			for (int i = 0; i < n_symbols; i++)
//			{
//				p += pow(emis_prob[new_state][i], observations[i]);
//				if (r < p)
//				{
//					new_observation = i;
//					break;
//				}
//			}
//
//			// Добавляем наблюдение в вектор наблюдений
//			observations.push_back(new_observation);
//
//			// Обновляем текущее состояние
//			state = new_state;
//		}
//
//		// Возвращаем пару из векторов наблюдений и состояний
//		return make_pair(observations, states);
//	}
//
//	double forward_backward(vector<int> observations) override
//	{
//		int length = observations.size();
//
//		// Инициализируем матрицы для хранения прямых и обратных вероятностей
//		vector<vector<double>> forward_prob(n_states, vector<double>(length));
//		vector<vector<double>> backward_prob(n_states, vector<double>(length));
//
//		// Вычисляем прямые вероятности для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//		for (int t = 0; t < length; t++)
//		{
//			for (int i = 0; i < n_states; i++)
//			{
//				if (t == 0)
//				{
//					// Инициализируем прямые вероятности для первого момента времени согласно начальным вероятностям и полиномиальному распределению
//					forward_prob[i][t] = start_prob[i] * pow(emis_prob[i][observations[t]], observations[t]);
//				}
//				else
//				{
//					// Вычисляем прямые вероятности для остальных моментов времени с учетом предыдущих прямых вероятностей и переходных вероя
//					// Вычисляем обратные вероятности для каждого состояния и каждого момента времени с помощью рекуррентной формулы
//					for (int t = length - 1; t >= 0; t--)
//					{
//						for (int i = 0; i < n_states; i++)
//						{
//							if (t == length - 1)
//							{
//								// Инициализируем обратные вероятности для последнего момента времени равными единице
//								backward_prob[i][t] = 1.0;
//							}
//							else
//							{
//								// Вычисляем обратные вероятности для остальных моментов времени с учетом следующих обратных вероятностей и переходных вероятностей
//								double sum = 0.0;
//								for (int j = 0; j < n_states; j++)
//								{
//									sum += backward_prob[j][t + 1] * trans_prob[i][j] * pow(
//										emis_prob[j][observations[t + 1]], observations[t + 1]);
//								}
//								backward_prob[i][t] = sum;
//							}
//						}
//					}
//				}
//			}
//		}
//		return 0;
//	}
//	
//};
