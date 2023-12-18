#include "Tester.h"
#

int main(const int argc, char* argv[])
{
	Tester test;
	test.Run(argc, argv);

	HMM model(2, 300);

	std::ifstream f("C:/Users/Vlad/Desktop/data.json");
	json data;

	f>>data;

	//std::cout << data.dump(4) << std::endl;

	const Vector1D<int> data_test =  data["data"];

	/*Array2D<double,3,3> arra;
	arra.Random(0.0,1);
	std::cout << arra;
	arra.SoftMax();
	std::cout<< arra;*/


	PrintVector(model.emission_prob);
	model.Train(100, 0.001, data_test);
	PrintVector(model.emission_prob);
	/*auto res = model.Viterbi({0,1,2,3});
	for (int first : res.first)
	{
		std::cout << first<< " ";
	}
	std::cout << res.second;*/


	/*std::cout << "Print emission prob" << std::endl;
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

	auto out = model.Viterbi({1,2,3,4});
	std::cout << out.second<<std::endl;
	PrintVector(out.first);*/
}
