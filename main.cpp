#include "Tester.h"


int main(int argc, char* argv[])
{
	Tester test;
	test.Run(argc, argv);

	HMM model(2, 5);

	Vector2D<int> observations = {
		{0, 1, 2, 3, 2, 3, 2, 1, 1, 2, 1, 2, 3, 4, 3, 2, 1, 0, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 4, 3, 2, 1, 2, 1},
		{0, 1, 3, 3, 2, 3, 2, 1, 1, 2, 1, 2, 3, 4, 3, 2, 1, 0, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 4, 3, 2, 1, 2, 1}

	};

	std::cout << "Print emission prob" << std::endl;
	PrintVector(model.emission_prob);
	std::cout << "Print trans prob" << std::endl;
	PrintVector(model.trans_prob);
	for (auto i = 0; i < observations.size(); i++)
	{
		std::cout << "Start " << i << std::endl;
		model.Train(100, 0.00001, observations[i]);
		std::cout << "Print prob after training!" << std::endl;
		std::cout << "Print emission prob" << std::endl;
		PrintVector(model.emission_prob);
		std::cout << "Print trans prob" << std::endl;
		PrintVector(model.trans_prob);
	}
}
