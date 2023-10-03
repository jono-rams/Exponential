#pragma once
#ifndef JONATHAN_RAMPERSAD_EXPONENTIAL_H_
#define JONATHAN_RAMPERSAD_EXPONENTIAL_H_

#include <ostream>
#include <vector>
#include <float.h>
#include <random>
#include <algorithm>
#include <execution>
#include <exception>

namespace JRAMPERSAD
{
	namespace EXPONENTIAL
	{
		/**
		* \brief Structure for options to be used when running one of the two genetic algorithms in a Function object
		*
		*/
		struct GA_Options
		{
			/** \brief Minimum value you believe the answer can be */
			double min_range = -100;
			/** \brief Maximum value you believe the answer can be */
			double max_range = 100;
			/** \brief Number of times you'd like to run the algorithm  (increasing this value causes the algorithm to take longer) */
			unsigned int num_of_generations = 10;
			/** \brief Amount of approximate solutions you'd like to be returned */
			unsigned int sample_size = 1000;
			/** \brief Amount of solutions you'd like the algorithm to generate (increasing this value causes the algorithm to take longer) */
			unsigned int data_size = 100000;
			/** \brief How much you'd like the algorithm to mutate solutions (Leave this as default in most cases) */
			double mutation_percentage = 0.01;
		};

		namespace detail
		{
			template<typename T>
			[[nodiscard("MATH::ABS(T) returns a value of type T")]] T ABS(const T& n) noexcept
			{
				return n < 0 ? n * -1 : n;
			}

			template<typename T>
			[[nodiscard("MATH::NEGATE(T) returns a value of type T")]] T NEGATE(const T& n) noexcept
			{
				return n * -1;
			}

			template<typename T>
			[[nodiscard("MATH::POW(T, int) returns a value of type T")]] T POW(const T& n, const int& exp) noexcept
			{
				if (exp == 0)
					return 1;

				T res = n;
				for (int i = 1; i < exp; i++)
				{
					res *= n;
				}

				return res;
			}

			template<typename T>
			[[nodiscard("MATH::SUM(std::vector<T>) returns a value of type T")]] T SUM(const std::vector<T>& vec) noexcept
			{
				T res{};
				for (auto& val : vec)
					res += val;
				return res;
			}

			template<typename T>
			[[nodiscard]] T MEDIAN(std::vector<T> vec) noexcept
			{
				std::sort(
					vec.begin(),
					vec.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs < rhs;
					});

				return vec[vec.size() / 2];
			}

			template<typename T>
			[[nodiscard]] double MEAN(const std::vector<T>& vec) noexcept
			{
				return SUM(vec) / vec.size();
			}

			template<typename T>
			[[noreturn]] void SortASC(std::vector<T>& vec)
			{
				std::sort(
					std::execution::par,
					vec.begin(), vec.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs < rhs;
					});
			}

			template<typename T>
			[[noreturn]] void SortDESC(std::vector<T>& vec)
			{
				std::sort(
					std::execution::par,
					vec.begin(), vec.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs > rhs;
					});
			}

			template <int lrgst_expo> // Genetic Algorithm helper struct
			struct GA_Solution
			{
				double rank, x, y_val;
				bool ranked;

				GA_Solution() : rank(0), x(0), y_val(0), ranked(false) {}
				GA_Solution(double Rank, double x_val, double y = 0) : rank(Rank), x(x_val), y_val(y), ranked(false) {}
				virtual ~GA_Solution() = default;

				void fitness(const std::vector<int>& constants)
				{
					double ans = 0;
					for (int i = lrgst_expo; i >= 0; i--)
						ans += constants[i] * POW(x, (lrgst_expo - i));

					ans -= y_val;
					rank = (ans == 0) ? DBL_MAX : ABS(1 / ans);
				}
			};
		}

		using namespace detail;
		/**
		* \brief A class representing an Exponential Function (e.g 2x^2 + 4x - 1),
		* \tparam lrgst_expo The largest exponent in the function (e.g 2 means largest exponent is x^2)
		*/
		template <int lrgst_expo>
		class Function
		{
		private:
			std::vector<int> constants;

		public:
			// Speicialty function to get the real roots of a Quadratic Function without relying on a Genetic Algorithm to approximate
			friend std::vector<double> QuadraticSolve(const Function<2>& f);

		public:
			/**
			* \brief Constructor for Function class
			* \param constnts An array with the constants for the function (e.g 2, 1, 3 = 2x^2 + 1x - 3) size of array MUST be lrgst_expo + 1
			*/
			Function(const std::vector<int>& constnts);
			/**
			* \brief Constructor for Function class
			* \param constnts An array with the constants for the function (e.g 2, 1, 3 = 2x^2 + 1x - 3) size of array MUST be lrgst_expo + 1
			*/
			Function(std::vector<int>&& constnts);
			Function(const Function& other) = default;
			Function(Function&& other) noexcept = default;
			virtual ~Function();

			Function& operator=(const Function& other) = default;
			Function& operator=(Function&& other) noexcept = default;

			// Operator function to display function object in a human readable format
			friend std::ostream& operator<<(std::ostream& os, const Function<lrgst_expo> func)
			{
				if (lrgst_expo == 0)
				{
					os << func.constants[0];
					return os;
				}

				if (func.constants[0] == 1)
					os << "x";
				else if (func.constants[0] == -1)
					os << "-x";
				else
					os << func.constants[0] << "x";

				if (lrgst_expo != 1)
					os << "^" << lrgst_expo;

				for (int i = lrgst_expo - 1; i > 0; i--)
				{
					int n = func.constants[lrgst_expo - i];
					if (n == 0) continue;

					auto s = n > 0 ? " + " : " - ";

					if (n != 1)
						os << s << ABS(n) << "x";
					else
						os << s << "x";

					if (i != 1)
						os << "^" << i;
				}

				int n = func.constants[lrgst_expo];
				if (n == 0) return os;

				auto s = n > 0 ? " + " : " - ";
				os << s;

				os << ABS(n);

				return os;
			}

			template<int e1, int e2, int r>
			friend Function<r> operator+(const Function<e1>& f1, const Function<e2>& f2); // Operator to add two functions
			template<int e1, int e2, int r>
			friend Function<r> operator-(const Function<e1>& f1, const Function<e2>& f2); // Operator to subtract two functions

			// Operators to multiply a function by a constant (Scaling it)
			friend Function<lrgst_expo> operator*(const Function<lrgst_expo>& f, const int& c)
			{
				if (c == 1) return f;
				if (c == 0) throw std::logic_error("Cannot multiply a function by 0");

				std::vector<int> res;
				for (auto& val : f.constants)
					res.push_back(c * val);

				return Function<lrgst_expo>(res);
			}
			Function<lrgst_expo>& operator*=(const int& c)
			{
				if (c == 1) return *this;
				if (c == 0) throw std::logic_error("Cannot multiply a function by 0");

				for (auto& val : this->constants)
					val *= c;

				return *this;
			}

			/**
			* \brief Calculates the differential (dy/dx) of the function
			* \returns a function representing the differential (dy/dx) of the calling function object
			*/
			[[nodiscard("MATH::EXP::Function::differential() returns the differential, the calling object is not changed")]]
			Function<lrgst_expo - 1> differential() const;

			/**
			* \brief Function that uses a genetic algorithm to find the approximate roots of the function
			* \param options GA_Options object specifying the options to run the algorithm
			* \returns A vector containing a n number of approximate root values (n = sample_size as defined in options)
			*/
			[[nodiscard]] std::vector<double> get_real_roots(const GA_Options& options = GA_Options()) const;

			/**
			* \brief Function that solves for y when x = user value
			* \param x_val the X Value you'd like the function to use
			* \returns the Y value the function returns based on the entered X value
			*/
			[[nodiscard]] double solve_y(const double& x_val) const noexcept;

			/**
			* \brief Function that uses a genetic algorithm to find the values of x where y = user value
			* \param y_val The return value that you would like to find the approximate x values needed to solve when entered into the function
			* \param options GA_Options object specifying the options to run the algorithm
			* \returns A vector containing a n number of x values that cause the function to approximately equal the y_val (n = sample_size as defined in options)
			*/
			[[nodiscard]] std::vector<double> solve_x(const double& y_val, const GA_Options& options = GA_Options()) const;
		};

		/**
		* \brief Uses the quadratic function to solve the roots of an entered quadratic equation
		* \param f Quadratic function you'd like to find the roots of (Quadratic Function object is a Function<2> object
		* \returns a vector containing the roots
		*/
		std::vector<double> QuadraticSolve(const Function<2>& f)
		{
			std::vector<double> res;

			const int& a = f.constants[0];
			const int& b = f.constants[1];
			const int& c = f.constants[2];

			const double sqr_val = static_cast<double>(POW(b, 2) - (4 * a * c));

			if (sqr_val < 0)
			{
				return res;
			}

			res.push_back(((NEGATE(b) + sqrt(sqr_val)) / 2 * a));
			res.push_back(((NEGATE(b) - sqrt(sqr_val)) / 2 * a));
			return res;
		}

		template<int e1, int e2, int r = (e1 > e2 ? e1 : e2)>
			Function<r> operator+(const Function<e1>& f1, const Function<e2>& f2)
		{
			std::vector<int> res;
			if (e1 > e2)
			{
				for (auto& val : f1.constants)
					res.push_back(val);

				int i = e1 - e2;
				for (auto& val : f2.constants)
				{
					res[i] += val;
					i++;
				}
			}
			else
			{
				for (auto& val : f2.constants)
					res.push_back(val);

				int i = e2 - e1;
				for (auto& val : f1.constants)
				{
					res[i] += val;
					i++;
				}
			}

			return Function<r>{res};
		}

		template<int e1, int e2, int r = (e1 > e2 ? e1 : e2)>
			Function<r> operator-(const Function<e1>& f1, const Function<e2>& f2)
		{
			std::vector<int> res;
			if (e1 > e2)
			{
				for (auto& val : f1.constants)
					res.push_back(val);

				int i = e1 - e2;
				for (auto& val : f2.constants)
				{
					res[i] -= val;
					i++;
				}
			}
			else
			{
				for (auto& val : f2.constants)
					res.push_back(val);

				int i = e2 - e1;

				for (int j = 0; j < i; j++)
					res[j] *= -1;

				for (auto& val : f1.constants)
				{
					res[i] = val - res[i];
					i++;
				}
			}

			return Function<r>{res};
		}

		template <int lrgst_expo>
		Function<lrgst_expo>::Function(const std::vector<int>& constnts)
		{
			if (lrgst_expo < 0)
				throw std::logic_error("Function template argument must not be less than 0");

			if (constnts.size() != lrgst_expo + 1)
				throw std::logic_error("Function<n> must be created with (n+1) integers in vector object");

			if (constnts[0] == 0)
				throw std::logic_error("First value should not be 0");

			constants = constnts;
		}

		template<int lrgst_expo>
		Function<lrgst_expo>::Function(std::vector<int>&& constnts)
		{
			if (lrgst_expo < 0)
				throw std::logic_error("Function template argument must not be less than 0");

			if (constnts.size() != lrgst_expo + 1)
				throw std::logic_error("Function<n> must be created with (n+1) integers in vector object");

			if (constnts[0] == 0)
				throw std::logic_error("First value should not be 0");

			constants = std::move(constnts);
		}

		template <int lrgst_expo>
		Function<lrgst_expo>::~Function()
		{
			constants.clear();
		}

		template <int lrgst_expo>
		Function<lrgst_expo - 1> Function<lrgst_expo>::differential() const
		{
			if (lrgst_expo == 0)
				throw std::logic_error("Cannot differentiate a number (Function<0>)");

			std::vector<int> result;
			for (int i = 0; i < lrgst_expo; i++)
			{
				result.push_back(constants[i] * (lrgst_expo - i));
			}

			return Function<lrgst_expo - 1>{result};
		}

		template<int lrgst_expo>
		std::vector<double> Function<lrgst_expo>::get_real_roots(const GA_Options& options) const
		{
			// Create initial random solutions
			std::random_device device;
			std::uniform_real_distribution<double> unif(options.min_range, options.max_range);
			std::vector<GA_Solution<lrgst_expo>> solutions;

			solutions.resize(options.data_size);
			for (unsigned int i = 0; i < options.sample_size; i++)
				solutions[i] = (GA_Solution<lrgst_expo>{0, unif(device)});

			float timer{ 0 };

			for (unsigned int count = 0; count < options.num_of_generations; count++)
			{
				std::generate(std::execution::par, solutions.begin() + options.sample_size, solutions.end(), [&unif, &device]() {
					return GA_Solution<lrgst_expo>{0, unif(device)};
					});

				// Run our fitness function
				for (auto& s : solutions) { s.fitness(constants); }

				// Sort our solutions by rank
				std::sort(std::execution::par, solutions.begin(), solutions.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs.rank > rhs.rank;
					});

				// Take top solutions
				std::vector<GA_Solution<lrgst_expo>> sample;
				std::copy(
					solutions.begin(),
					solutions.begin() + options.sample_size,
					std::back_inserter(sample)
				);
				solutions.clear();

				if (count + 1 == options.num_of_generations)
				{
					std::copy(
						sample.begin(),
						sample.end(),
						std::back_inserter(solutions)
					);
					sample.clear();
					break;
				}

				// Mutate the top solutions by %
				std::uniform_real_distribution<double> m((1 - options.mutation_percentage), (1 + options.mutation_percentage));
				std::for_each(sample.begin(), sample.end(), [&m, &device](auto& s) {
					s.x *= m(device);
					});

				// Cross over not needed as it's only one value

				std::copy(
					sample.begin(),
					sample.end(),
					std::back_inserter(solutions)
				);
				sample.clear();
				solutions.resize(options.data_size);
			}

			std::sort(solutions.begin(), solutions.end(),
				[](const auto& lhs, const auto& rhs) {
					return lhs.x < rhs.x;
				});

			std::vector<double> ans;
			for (auto& s : solutions)
			{
				ans.push_back(s.x);
			}
			return ans;
		}

		template<int lrgst_expo>
		double Function<lrgst_expo>::solve_y(const double& x_val) const noexcept
		{
			std::vector<bool> exceptions;

			for (int i : constants)
				exceptions.push_back(i != 0);

			double ans{ 0 };
			for (int i = lrgst_expo; i >= 0; i--)
			{
				if (exceptions[i])
					ans += constants[i] * POW(x_val, (lrgst_expo - i));
			}

			return ans;
		}

		template<int lrgst_expo>
		inline std::vector<double> Function<lrgst_expo>::solve_x(const double& y_val, const GA_Options& options) const
		{
			// Create initial random solutions
			std::random_device device;
			std::uniform_real_distribution<double> unif(options.min_range, options.max_range);
			std::vector<GA_Solution<lrgst_expo>> solutions;

			solutions.resize(options.data_size);
			for (unsigned int i = 0; i < options.sample_size; i++)
				solutions[i] = (GA_Solution<lrgst_expo>{0, unif(device), y_val});

			for (unsigned int count = 0; count < options.num_of_generations; count++)
			{
				std::generate(std::execution::par, solutions.begin() + options.sample_size, solutions.end(), [&unif, &device, &y_val]() {
					return GA_Solution<lrgst_expo>{0, unif(device), y_val};
					});


				// Run our fitness function
				for (auto& s : solutions) { s.fitness(constants); }

				// Sort our solutions by rank
				std::sort(std::execution::par, solutions.begin(), solutions.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs.rank > rhs.rank;
					});

				// Take top solutions
				std::vector<GA_Solution<lrgst_expo>> sample;
				std::copy(
					solutions.begin(),
					solutions.begin() + options.sample_size,
					std::back_inserter(sample)
				);
				solutions.clear();

				if (count + 1 == options.num_of_generations)
				{
					std::copy(
						sample.begin(),
						sample.end(),
						std::back_inserter(solutions)
					);
					sample.clear();
					break;
				}

				// Mutate the top solutions by %
				std::uniform_real_distribution<double> m((1 - options.mutation_percentage), (1 + options.mutation_percentage));
				std::for_each(sample.begin(), sample.end(), [&m, &device](auto& s) {
					s.x *= m(device);
					});

				// Cross over not needed as it's only one value

				std::copy(
					sample.begin(),
					sample.end(),
					std::back_inserter(solutions)
				);
				sample.clear();
				solutions.resize(options.data_size);
			}

			std::sort(solutions.begin(), solutions.end(),
				[](const auto& lhs, const auto& rhs) {
					return lhs.x < rhs.x;
				});

			std::vector<double> ans;
			for (auto& s : solutions)
			{
				ans.push_back(s.x);
			}
			return ans;
		}
	}
}

#endif // !JONATHAN_RAMPERSAD_EXPONENTIAL_H_