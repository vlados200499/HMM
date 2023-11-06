#include <map>

#include "model/HMM.h"


int main()
{
	//                      HOT / COLD
	std::vector<int> state = {0, 1};

	std::vector<double> start_prob = {0.5, 0.5};

	std::vector<std::vector<double>> trans_prob = {
		{0.5, 0.5}, // HOT
		{0.5, 0.5}, // COLD
	}; 
	std::vector<std::vector<double>> emission_prob = {
		{0.0, 0.2, 0.7, 0.9, 0.3} , // HOT
		{0.9, 0.8, 0.6, 0.3, 0.0},  // COLD
	}; 

	std::vector<std::string> states = {"HOT", "COLD"};


	PolynomialHMM model(state, start_prob, trans_prob, emission_prob);


	//const double forward_pr = model.Forward({0,3,3,3,0,3,0,1});
	//std::cout << "Total Probability: " << forward_pr << std::endl;

	const auto result = model.Viterbi({0,3,2,0});

	std::cout << "Best Path: ";
	for (const int state1 : result.first)
	{
		std::cout << states[state1] << " ";
	}
	for (const auto & prop : result.second)
	{
		std::cout << "\nBest Path Probability: " << prop << std::endl;
		
	}
}
