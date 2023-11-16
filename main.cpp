#include <map>

#include "model/HMM.h"




int main()
{
	

	std::vector start_prob = {0.5, 0.5};

	std::vector<std::vector<double>> trans_prob = {
		{0.001, 0.45}, // HOT
		{0.94, 0.935}, // COLD
	};
	std::vector<std::vector<double>> emission_prob = {
		{0.995, 0.057, 0.45, 0.235, 0.135}, // HOT
		{0.035, 0.45675, 0.45, 0.425, 0.24435}, // COLD
	};


	




	




	const Vector1D<int> observation_data = {0, 1, 2, 3, 4};
	std::vector<std::string> states = {"HOT", "COLD"};

	PolynomialHMM model(2,4, start_prob, trans_prob, emission_prob);
	
	


	Vector2D<int> observationDataJdj = {{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,0,2,1,2,2,1,3,2,3,2},{0,0,2,1,2,2,1,3,2,3,2},{0,0,2,1,2,2,1,3,2,3,2}
	,{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},{0,1,2,1,2,2,1,3,2,3,2},};
	//model.BaumWelch(observation_data_jdj);
	//double logLikelihood = model.ComputeLogLikelihood(observation_data_jdj);
    
    //std::cout << "Log Likelihood after training: " << logLikelihood << std::endl;
	model.Train(1000,0.000001,observationDataJdj);
	const double forward_pr = model.Forward({0, 2, 1,4,0,0,0,0});
	std::cout << "Total Probability: " << forward_pr << std::endl;


	//ogLikelihood = model.ComputeLogLikelihood(observation_data_jdj);
	//std::cout << "Log Likelihood after training: " << logLikelihood << std::endl;
}
