#pragma once
#include <cxxopts.hpp>
#include <tuple>
#include <functional>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <HMM.h>
using json = nlohmann::json;
using HMM = HMMlearn::HMM;

class JSONParser
{
public:
	struct Data
	{
		int n_states;
		int n_observations;
		Vector1D<double> start_prob;
		Vector2D<double> trans_prob;
		Vector2D<double> emission_prob;
	};

	explicit JSONParser(std::string file_path) : file_path_(std::move(file_path))
	{
	}

	bool Parse()
	{
		std::ifstream input_file(file_path_);
		if (!input_file.is_open())
		{
			std::cerr << "Failed to open the file.\n";
			return false;
		}

		json j;
		input_file >> j;


		Data entry;
		entry.n_states = j["n_states"];
		entry.n_observations = j["n_observations"];
		entry.start_prob = j["start_prob"].get<Vector1D<double>>();
		entry.trans_prob = j["trans_prob"].get<Vector2D<double>>();
		entry.emission_prob = j["emission_prob"].get<Vector2D<double>>();
		data_vector_ = entry;


		return true;
	}


	static json ParseTrain(const std::string& file)
	{
		json j;
		std::ifstream f(file);
		return json::parse(f);
	}


	Data data_vector_;

private:
	std::string file_path_;
};

class Tester
{
public:
	void Run(int argc, char* argv[])
	{
		cxxopts::Options options("HMM Tester", "Command-line options for Hidden Markov Model testing");

		// Define command-line options
		options.add_options()
			("t,train_data", "Path to the JSON file containing training data", cxxopts::value<std::string>())
			("e,test_data", "Path to the JSON file containing test data", cxxopts::value<std::string>())
			("O,output_model", "Path to save the trained HMM model in JSON format", cxxopts::value<std::string>())
			("s,num_states", "Number of hidden states in the HMM", cxxopts::value<int>())
			("o,num_symbols", "Number of observation symbols in the HMM", cxxopts::value<int>())
			("i,num_iterations", "Number of iterations for training using the Baum-Welch algorithm",
			 cxxopts::value<int>())
			("p,initial_model", "Path to an existing HMM model in JSON format for initialization",
			 cxxopts::value<std::string>())
			("d,decode_output", "Path to save the decoded output of the test data in JSON format",
			 cxxopts::value<std::string>())
			("v,viterbi", "Include this flag to perform decoding using the Viterbi algorithm")
			("T,time", "Measure and display the execution time");


		try
		{
			const auto result = options.parse(argc, argv);

			// Check if training data is provided
			if (result.count("train_data"))
			{
				TrainHMM(result, options);
			}

			// Check if test data is provided
			if (result.count("test_data"))
			{
				TestHMM(result, options);
			}

			// If neither training nor testing options are provided, display help
			if (!result.count("train_data") && !result.count("test_data"))
			{
				PrintHelp(options);
			}
		}
		catch (const cxxopts::OptionException& e)
		{
			std::cerr << "Error parsing command line options: " << e.what() << std::endl;
			PrintHelp(options);
		}
	}

private:
	void TrainHMM(const cxxopts::ParseResult& result, const cxxopts::Options& opt)
	{
		auto file_out = result["output_model"].as<std::string>();
		if (result.count("output_model") && result.count("num_iterations") && result.count("initial_model"))
		{
			std::cout << "Starting Hidden Markov Model training..." << std::endl;

			// Check if an existing model is provided for initialization

			std::cout << "Initializing training with an existing HMM model." << std::endl;


			// Set up model
			JSONParser parser(result["initial_model"].as<std::string>());
			parser.Parse();
			HMM model(parser.data_vector_.n_states, parser.data_vector_.n_observations,
			          parser.data_vector_.start_prob, parser.data_vector_.trans_prob,
			          parser.data_vector_.emission_prob);

			// Train model
			auto data = JSONParser::ParseTrain(result["train_data"].as<std::string>())["data"].get<Vector1D<int>>();

			if (result.count("time"))
			{
				MeasureTime([&data, &model, result]
				{
					model.Train(result["num_iterations"].as<int>(), 0.01, data);
				}, "Training");
				ExportDataToJson(file_out, model.n_observations, model.states_size, model.start_prob, model.trans_prob,
				                 model.emission_prob);
			}
			else
			{
				model.Train(result["num_iterations"].as<int>(), 0.01, data);
				ExportDataToJson(file_out, model.n_observations, model.states_size, model.start_prob, model.trans_prob,
				                 model.emission_prob);
			}


			// Display training completion message
			std::cout << "Training completed successfully. Model saved to: " << result["output_model"].as<std::string>()
				<< std::endl;
		}
		if (result.count("output_model") && result.count("num_iterations") && result.count("num_states") && result.
			count("num_symbols"))
		{
			// Display additional training parameters
			std::cout << "Number of hidden states: " << result["num_states"].as<int>() << std::endl;
			std::cout << "Number of observation symbols: " << result["num_symbols"].as<int>() << std::endl;
			std::cout << "Number of training iterations: " << result["num_iterations"].as<int>() << std::endl;


			HMM model(result["num_states"].as<int>(), result["num_symbols"].as<int>());

			auto data = JSONParser::ParseTrain(result["train_data"].as<std::string>())["data"].get<Vector1D<
				int>>();


			if (result.count("time"))
			{
				MeasureTime([&data, &model, result]
				{
					model.Train(result["num_iterations"].as<int>(), 0.01, data);
				}, "Training");
				ExportDataToJson(file_out, model.n_observations, model.states_size, model.start_prob, model.trans_prob,
				                 model.emission_prob);
			}
			else
			{
				model.Train(result["num_iterations"].as<int>(), 0.01, data);
				ExportDataToJson(file_out, model.n_observations, model.states_size, model.start_prob, model.trans_prob,
				                 model.emission_prob);
			}
		}
		else
		{
			std::cerr << "Error: Insufficient options provided for training." << std::endl;
			PrintHelp(opt);
		}
	}

	void TestHMM(const cxxopts::ParseResult& result, const cxxopts::Options& opt)
	{
		// Check if necessary options for testing are provided
		if (result.count("initial_model") && result.count("decode_output") && result.count("viterbi"))
		{
			// Display information about the testing process
			std::cout << "Starting Hidden Markov Model testing..." << std::endl;

			// Check if Viterbi decoding is automatically used during testing
			std::cout << "Using Viterbi decoding during testing." << std::endl;


			// Display additional testing parameters
			std::cout << "Path to test data: " << result["test_data"].as<std::string>() << std::endl;

			// Check if an existing model is provided for testing
			std::cout << "Using an existing HMM model for testing." << std::endl;

			JSONParser parser(result["initial_model"].as<std::string>());
			parser.Parse();
			HMM model(parser.data_vector_.n_states, parser.data_vector_.n_observations,
			          parser.data_vector_.start_prob, parser.data_vector_.trans_prob,
			          parser.data_vector_.emission_prob);

			const auto data = JSONParser::ParseTrain(result["test_data"].as<std::string>())["data"].get<Vector1D<
				int>>();

			if (result.count("time")) MeasureTime([&data, &model] { model.Viterbi(data); }, "Viterbi");
			else model.Viterbi(data);

			// Display testing completion message
			std::cout << "Testing completed successfully. Decoded output saved to: " << result["decode_output"].as<
				std::string>() << std::endl;
		}
		else
		{
			std::cerr << "Error: Insufficient options provided for testing." << std::endl;
			PrintHelp(opt);
		}
	}

	static void PrintHelp(const cxxopts::Options& options)
	{
		std::cout << options.help() << std::endl;
	}

	static void ExportDataToJson(std::string file,
	                             int n_observations,
	                             int n_states,
	                             const Vector1D<double>& start_prob,
	                             const Vector2D<double>& trans_prob,
	                             const Vector2D<double>& emission_prob
	)
	{
		// Create a JSON object to hold the data
		json jsonData;

		// Add individual elements to the JSON object
		jsonData["states"] = n_states;
		jsonData["n_observations"] = n_observations;
		jsonData["start_prob"] = start_prob;
		jsonData["trans_prob"] = trans_prob;
		jsonData["emission_prob"] = emission_prob;

		// Write JSON to a file
		std::ofstream outFile(file);
		if (outFile.is_open())
		{
			outFile << std::setw(4) << jsonData << std::endl;
			std::cout << "JSON file generated successfully: exported_data.json" << std::endl;
			outFile.close();
		}
		else
		{
			std::cerr << "Unable to open file for writing." << std::endl;
		}
	}


	template <typename Function, typename... Args>
	void MeasureTime(Function&& action, const std::string& actionName, Args&&... args)
	{
		const auto start_time = std::chrono::steady_clock::now();

		// Execute the provided function with arguments
		std::invoke(std::forward<Function>(action), std::forward<Args>(args)...);

		const auto end_time = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << actionName << " completed in " << duration << " milliseconds." << std::endl;
	}
};
