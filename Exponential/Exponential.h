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
#include <type_traits>

namespace JRAMPERSAD
{
	namespace EXPONENTIAL
	{
		/**
		* \brief Structure for options to be used when running one of the two genetic algorithms in a Function object
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
				static_assert(std::is_arithmetic<T>::value, "Arithmetic type required.");
				return n < 0 ? n * -1 : n;
			}

			template<typename T>
			[[nodiscard("MATH::NEGATE(T) returns a value of type T")]] T NEGATE(const T& n) noexcept
			{
				static_assert(std::is_arithmetic<T>::value, "Arithmetic type required.");
				return n * -1;
			}

			template<typename T>
			[[nodiscard("MATH::POW(T, int) returns a value of type T")]] T POW(const T& n, const int& exp) noexcept
			{
				static_assert(std::is_arithmetic<T>::value, "Arithmetic type required.");
				if (exp == 0)
					return 1;

				T res = n;
				for (int i = 1; i < exp; i++)
				{
					res *= n;
				}

				return res;
			}

			// Genetic Algorithm helper struct
			struct GA_Solution
			{
				unsigned short lrgst_expo;
				double rank, x, y_val;

				GA_Solution() : lrgst_expo(0), rank(0), x(0), y_val(0) {}
				GA_Solution(unsigned short Lrgst_expo, double Rank, double x_val, double y = 0) : lrgst_expo(Lrgst_expo), rank(Rank), x(x_val), y_val(y) {}
				virtual ~GA_Solution() = default;

				void fitness(const std::vector<int64_t>& constants)
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
		* \brief class representing an Exponential Function (e.g 2x^2 + 4x - 1)
		*/
		class Function
		{
		private:
			const unsigned short lrgst_expo; /**< lrgst_expo The largest exponent in the function (e.g 2 means largest exponent is x^2) */
			std::vector<int64_t> constants;

			bool bInitialized;

			inline void CanPerform() const { if (!bInitialized) throw std::logic_error("Function object not initialized fully! Please call .SetConstants() to initialize"); }

		public:
			// Speicialty function to get the real roots of a Quadratic Function without relying on a Genetic Algorithm to approximate
			friend std::vector<double> QuadraticSolve(const Function& f);

		public:
			/**
			* \brief Constructor for Function class
			* \param Lrgst_expo The largest exponent in the function (e.g 2 means largest exponent is x^2)
			*/
			Function(const unsigned short& Lrgst_expo) : lrgst_expo(Lrgst_expo), bInitialized(false)
			{
				if (lrgst_expo < 0)
					throw std::logic_error("Function template argument must not be less than 0"); 
				constants.reserve(Lrgst_expo); 
			}
			/** \brief Destructor */
			virtual ~Function();
			/** \brief Copy Constructor */
			Function(const Function& other) = default;
			/** \brief Move Constructor */
			Function(Function&& other) noexcept = default;
			/** \brief Copy Assignment operator */
			Function& operator=(const Function& other) = default;
			/** \brief Move Assignment operator */
			Function& operator=(Function&& other) noexcept = default;

			/**
			* \brief Sets the constants of the function
			* \param constnts An array with the constants for the function (e.g 2, 1, 3 = 2x^2 + 1x - 3) size of array MUST be lrgst_expo + 1
			*/
			void SetConstants(const std::vector<int64_t>& constnts);
			/**
			* \brief Sets the constants of the function
			* \param constnts An array with the constants for the function (e.g 2, 1, 3 = 2x^2 + 1x - 3) size of array MUST be lrgst_expo + 1
			*/
			void SetConstants(std::vector<int64_t>&& constnts);

			friend std::ostream& operator<<(std::ostream& os, const Function func);
		
			friend Function operator+(const Function& f1, const Function& f2);
			friend Function operator-(const Function& f1, const Function& f2);

			friend Function operator*(const Function& f, const int64_t& c);
			Function& operator*=(const int64_t& c);

			/**
			* \brief Calculates the differential (dy/dx) of the Function
			* \returns a Function representing the differential (dy/dx) of the calling function object
			*/
			[[nodiscard("MATH::EXP::Function::differential() returns the differential, the calling object is not changed")]]
			Function differential() const;

			/**
			* \brief Uses a genetic algorithm to find the approximate roots of the function
			* \param options GA_Options object specifying the options to run the algorithm
			* \returns A vector containing a n number of approximate root values (n = sample_size as defined in options)
			*/
			[[nodiscard]] std::vector<double> get_real_roots(const GA_Options& options = GA_Options()) const;

			/**
			* \brief Solves for y when x = user value
			* \param x_val the X Value you'd like the function to use
			* \returns the Y value the function returns based on the entered X value
			*/
			[[nodiscard]] double solve_y(const double& x_val) const;

			/**
			* \brief Uses a genetic algorithm to find the values of x where y = user value
			* \param y_val The return value that you would like to find the approximate x values needed to solve when entered into the function
			* \param options GA_Options object specifying the options to run the algorithm
			* \returns A vector containing a n number of x values that cause the function to approximately equal the y_val (n = sample_size as defined in options)
			*/
			[[nodiscard]] std::vector<double> solve_x(const double& y_val, const GA_Options& options = GA_Options()) const;

			/** \returns lrgst_expo */
			[[nodiscard]] auto GetWhatIsTheLargestExponent() const { return lrgst_expo; }
		};

		/**
		* \brief Uses the quadratic function to solve the roots of an entered quadratic equation
		* \param f Quadratic function you'd like to find the roots of (Quadratic Function object is a Function object who's lrgst_expo value = 2
		* \returns a vector containing the roots
		*/
		std::vector<double> QuadraticSolve(const Function& f)
		{
			try
			{
				if (f.lrgst_expo != 2) throw std::logic_error("Function f is not a quadratic function");
				f.CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			std::vector<double> res;

			const auto& a = f.constants[0];
			const auto& b = f.constants[1];
			const auto& c = f.constants[2];

			const double sqr_val = static_cast<double>(POW(b, 2) - (4 * a * c));

			if (sqr_val < 0)
			{
				return res;
			}

			res.push_back(((NEGATE(b) + sqrt(sqr_val)) / 2 * a));
			res.push_back(((NEGATE(b) - sqrt(sqr_val)) / 2 * a));
			return res;
		}
	
		Function::~Function()
		{
			constants.clear();
		}

		void Function::SetConstants(const std::vector<int64_t>& constnts)
		{
			if (constnts.size() != lrgst_expo + 1)
				throw std::logic_error("Function<n> must be created with (n+1) integers in vector object");

			if (constnts[0] == 0)
				throw std::logic_error("First value should not be 0");

			constants = constnts;
			bInitialized = true;
		}

		void Function::SetConstants(std::vector<int64_t>&& constnts)
		{
			if (constnts.size() != lrgst_expo + 1)
				throw std::logic_error("Function<n> must be created with (n+1) integers in vector object");

			if (constnts[0] == 0)
				throw std::logic_error("First value should not be 0");

			constants = std::move(constnts);
			bInitialized = true;
		}

		/** Operator function to display function object in a human readable format */
		std::ostream& operator<<(std::ostream& os, const Function func)
		{
			try
			{
				func.CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			if (func.lrgst_expo == 0)
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

			if (func.lrgst_expo != 1)
				os << "^" << func.lrgst_expo;

			for (auto i = func.lrgst_expo - 1; i > 0; i--)
			{
				auto n = func.constants[func.lrgst_expo - i];
				if (n == 0) continue;

				auto s = n > 0 ? " + " : " - ";

				if (n != 1)
					os << s << ABS(n) << "x";
				else
					os << s << "x";

				if (i != 1)
					os << "^" << i;
			}

			auto n = func.constants[func.lrgst_expo];
			if (n == 0) return os;

			auto s = n > 0 ? " + " : " - ";
			os << s;

			os << ABS(n);

			return os;
		}

		/** Operator to add two functions */
		Function operator+(const Function& f1, const Function& f2)
		{
			try
			{
				f1.CanPerform();
				f2.CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			auto e1 = f1.lrgst_expo;
			auto e2 = f2.lrgst_expo;
			auto r = e1 > e2 ? e1 : e2;

			std::vector<int64_t> res;
			if (e1 > e2)
			{
				for (auto& val : f1.constants)
					res.push_back(val);

				auto i = e1 - e2;
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

			Function f(r);
			f.SetConstants(res);
			return f;
		}
	
		/** Operator to subtract two functions */
		Function operator-(const Function& f1, const Function& f2)
		{
			try
			{
				f1.CanPerform();
				f2.CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			auto e1 = f1.lrgst_expo;
			auto e2 = f2.lrgst_expo;
			auto r = e1 > e2 ? e1 : e2;

			std::vector<int64_t> res;
			if (e1 > e2)
			{
				for (auto& val : f1.constants)
					res.push_back(val);

				auto i = e1 - e2;
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

			Function f(r);
			f.SetConstants(res);
			return f;
		}

		/** Operator to multiply a function by a constant (Scaling it) */
		Function operator*(const Function& f, const int64_t& c)
		{
			try
			{
				f.CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			if (c == 1) return f;
			if (c == 0) throw std::logic_error("Cannot multiply a function by 0");

			std::vector<int64_t> res;
			for (auto& val : f.constants)
				res.push_back(c * val);

			Function f_res(f.lrgst_expo);
			f_res.SetConstants(res);

			return f_res;
		}

		/** Operator to multiply a function by a constant (Scaling it) */
		Function& Function::operator*=(const int64_t& c)
		{
			try
			{
				this->CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			if (c == 1) return *this;
			if (c == 0) throw std::logic_error("Cannot multiply a function by 0");

			for (auto& val : this->constants)
				val *= c;

			return *this;
		}
		
		Function Function::differential() const
		{
			try
			{
				this->CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			if (lrgst_expo == 0)
				throw std::logic_error("Cannot differentiate a number (Function<0>)");

			std::vector<int64_t> result;
			for (int i = 0; i < lrgst_expo; i++)
			{
				result.push_back(constants[i] * (lrgst_expo - i));
			}

			Function f{ (unsigned short)(lrgst_expo - 1) };
			f.SetConstants(result);

			return f;
		}

		std::vector<double> Function::get_real_roots(const GA_Options& options) const
		{
			try
			{
				this->CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			return solve_x(0, options);
		}

		double Function::solve_y(const double& x_val) const
		{
			try
			{
				this->CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			double ans{ 0 };
			for (int i = lrgst_expo; i >= 0; i--)
			{
				ans += constants[i] * POW(x_val, (lrgst_expo - i));
			}
			return ans;
		}

		inline std::vector<double> Function::solve_x(const double& y_val, const GA_Options& options) const
		{
			try
			{
				this->CanPerform();
			}
			catch (const std::exception& e)
			{
				throw e;
			}

			// Create initial random solutions
			std::random_device device;
			std::uniform_real_distribution<double> unif(options.min_range, options.max_range);
			std::vector<GA_Solution> solutions;

			solutions.resize(options.data_size);
			for (unsigned int i = 0; i < options.sample_size; i++)
				solutions[i] = (GA_Solution{lrgst_expo, 0, unif(device), y_val});

			for (unsigned int count = 0; count < options.num_of_generations; count++)
			{
				std::generate(std::execution::par, solutions.begin() + options.sample_size, solutions.end(), [this, &unif, &device, &y_val]() {
					return GA_Solution{lrgst_expo, 0, unif(device), y_val};
					});


				// Run our fitness function
				for (auto& s : solutions) { s.fitness(constants); }

				// Sort our solutions by rank
				std::sort(std::execution::par, solutions.begin(), solutions.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs.rank > rhs.rank;
					});

				// Take top solutions
				std::vector<GA_Solution> sample;
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

#define INITIALIZE_EXPO_FUNCTION(func, ...) \
func.SetConstants(__VA_ARGS__)

#endif // !JONATHAN_RAMPERSAD_EXPONENTIAL_H_