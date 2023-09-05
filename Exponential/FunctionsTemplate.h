#pragma once
#include <ostream>
#include <vector>
#include <float.h>
#include <random>
#include <algorithm>
#include <exception>

namespace MATH
{
	constexpr double GA_DEFAULT_MIN_RANGE = -100;
	constexpr double GA_DEFAULT_MAX_RANGE = 100;
	constexpr int GA_DEFAULT_NUM_OF_GENERATIONS = 100;
	constexpr int GA_DEFAULT_SAMPLE_SIZE = 1000;
	constexpr int GA_DEFAULT_DATA_SIZE = 100000;
	constexpr double GA_DEFAULT_MUTATION_PERCENTAGE = 0.01;

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
	[[nodiscard]] T SUM(const std::vector<T>& vec) noexcept
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
			[](const auto& lhs, const auto& rhs){
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
	void SortASC(std::vector<T>& vec)
	{
		std::sort(
			vec.begin(), vec.end(), 
			[](const auto& lhs, const auto& rhs) {
				return lhs < rhs;
			});
	}

	template<typename T>
	void SortDESC(std::vector<T>& vec)
	{
		std::sort(
			vec.begin(), vec.end(),
			[](const auto& lhs, const auto& rhs) {
				return lhs > rhs;
			});
	}

	class Coordinate2D
	{
	private:
		double X, Y;

	public:
		Coordinate2D() : X(0), Y(0) {}
		Coordinate2D(double x) : X(x), Y(x) {}
		Coordinate2D(double x, double y) : X(x), Y(y) {}
		virtual ~Coordinate2D() = default;

		inline void set_x(const double val) noexcept { X = val; }
		inline void set_y(const double val) noexcept { Y = val; }

		[[nodiscard]] inline double get_x() const noexcept { return X; }
		[[nodiscard]] inline double get_y() const noexcept { return Y; }

		friend std::ostream& operator<<(std::ostream& os, const Coordinate2D& coord)
		{
			os << '(' << coord.X << ", " << coord.Y << ") ";
			return os;
		}
	};

	namespace INTERNAL
	{
		template <int lrgst_exp> // Genetic Algorithm helper struct
		struct GA_Solution
		{
			double rank, x;

			GA_Solution(double Rank, double x_val) : rank(Rank), x(x_val) {}
			virtual ~GA_Solution() = default;

			void fitness(const std::vector<int>& constants)
			{
				std::vector<bool> exceptions;
				for (int i : constants)
					exceptions.push_back(i != 0);

				double ans = 0;
				for (int i = lrgst_exp; i >= 0; i--)
				{
					if (exceptions[i])
					{
						ans += constants[i] * POW(x, (lrgst_exp - i));
					}
				}
				rank = (ans == 0) ? DBL_MAX : ABS(1 / ans);
			}
		};
	}

	namespace EXP
	{
		template <int lrgst_exp>
		class Function
		{
		private:
			std::vector<int> constants;

		public:
			// Speicialty function to get the real roots of a Quadratic Function without relying on a Genetic Algorithm to approximate
			friend std::vector<double> QuadraticSolve(const Function<2>& f);

		public:
			Function(const std::vector<int>& constnts);
			Function(std::vector<int>&& constnts);
			Function(const Function& other) = default;
			Function(Function&& other) noexcept = default;
			virtual ~Function();

			Function& operator=(const Function& other) = default;
			Function& operator=(Function&& other) noexcept = default;

			// Operator function to display function object in a human readable format
			friend std::ostream& operator<<(std::ostream& os, const Function<lrgst_exp> func)
			{
				if (lrgst_exp == 0)
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

				if (lrgst_exp != 1)
					os << "^" << lrgst_exp;

				for (int i = lrgst_exp - 1; i > 0; i--)
				{
					int n = func.constants[lrgst_exp - i];
					if (n == 0) continue;
					
					auto s = n > 0 ? " + " : " - ";

					if (n != 1)
						os << s << ABS(n) << "x";
					else
						os << s << "x";

					if (i != 1)
						os << "^" << i;
				}

				int n = func.constants[lrgst_exp];
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
			friend Function<lrgst_exp> operator*(const Function<lrgst_exp>& f, const int& c)
			{
				if (c == 1) return f;
				if (c == 0) throw std::logic_error("Cannot multiply a function by 0");

				std::vector<int> res;
				for (auto& val : f.constants)
					res.push_back(c * val);

				return Function<lrgst_exp>(res);
			}		
			Function<lrgst_exp>& operator*=(const int& c)
			{
				if (c == 1) return *this;
				if (c == 0) throw std::logic_error("Cannot multiply a function by 0");

				for (auto& val : this->constants)
					val *= c;

				return *this;
			}

			[[nodiscard("MATH::EXP::Function::differential() returns the differential, the calling object is not changed")]]
			Function<lrgst_exp - 1> differential() const; // This function returns the differential (dy/dx) of the Function object

			// Function that uses a genetic algorithm to find the approximate roots of the function
			[[nodiscard]] std::vector<double> get_real_roots_ga(
				const double& min_range = GA_DEFAULT_MIN_RANGE,
				const double& max_range = GA_DEFAULT_MAX_RANGE,
				const int& num_of_generations = GA_DEFAULT_NUM_OF_GENERATIONS,
				const int& sample_size = GA_DEFAULT_SAMPLE_SIZE,
				const int& data_size = GA_DEFAULT_DATA_SIZE,
				const double& mutation_percentage = GA_DEFAULT_MUTATION_PERCENTAGE) const;

			// Function that returns the y-intercept of the function i.e. where x = 0
			[[nodiscard]] inline Coordinate2D get_y_intrcpt() const noexcept { return Coordinate2D{ 0, (double)constants[lrgst_exp] }; }

			[[nodiscard]] double solve_y(const double& x_val) const noexcept;

		};

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

			res.push_back(	((NEGATE(b) + sqrt(sqr_val)) / 2 * a)	);
			res.push_back(	((NEGATE(b) - sqrt(sqr_val)) / 2 * a)	);
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

		template <int lrgst_exp>
		Function<lrgst_exp>::Function(const std::vector<int>& constnts)
		{
			if (lrgst_exp < 0)
				throw std::logic_error("Function template argument must not be less than 0");

			if (constnts.size() != lrgst_exp + 1)
				throw std::logic_error("Function<n> must be created with (n+1) integers in vector object");

			if (constnts[0] == 0)
				throw std::logic_error("First value should not be 0");

			constants = constnts;
		}

		template<int lrgst_exp>
		Function<lrgst_exp>::Function(std::vector<int>&& constnts)
		{
			if (lrgst_exp < 0)
				throw std::logic_error("Function template argument must not be less than 0");

			if (constnts.size() != lrgst_exp + 1)
				throw std::logic_error("Function<n> must be created with (n+1) integers in vector object");

			if (constnts[0] == 0)
				throw std::logic_error("First value should not be 0");

			constants = std::move(constnts);
		}

		template <int lrgst_exp>
		Function<lrgst_exp>::~Function()
		{
			constants.clear();
		}

		template <int lrgst_exp>
		Function<lrgst_exp - 1> Function<lrgst_exp>::differential() const
		{
			if (lrgst_exp == 0)
				throw std::logic_error("Cannot differentiate a number (Function<0>)");

			std::vector<int> result;
			for (int i = 0; i < lrgst_exp; i++)
			{
				result.push_back(constants[i] * (lrgst_exp - i));
			}

			return Function<lrgst_exp - 1>{result};
		}

		template<int lrgst_exp>
		std::vector<double> Function<lrgst_exp>::get_real_roots_ga(
			const double& min_range, const double& max_range, 
			const int& num_of_generations, 
			const int& sample_size, const int& data_size, 
			const double& mutation_percentage) const
		{
			// Create initial random solutions
			std::random_device device;
			std::uniform_real_distribution<double> unif(static_cast<double>(min_range), static_cast<double>(max_range));
			std::vector<INTERNAL::GA_Solution<lrgst_exp>> solutions;

			for (int i = 0; i < sample_size; i++)
				solutions.push_back(INTERNAL::GA_Solution<lrgst_exp>{0, unif(device)});

			for(int count = 0; count < num_of_generations; count++)
			{ 
				for (int i = sample_size; i < data_size; i++)
					solutions.push_back(INTERNAL::GA_Solution<lrgst_exp>{0, unif(device)});

				// Run our fitness function
				for (auto& s : solutions) { s.fitness(constants); }

				// Sort our solutions by rank
				std::sort(solutions.begin(), solutions.end(),
					[](const auto& lhs, const auto& rhs) {
						return lhs.rank > rhs.rank;
					});

				// Take top solutions
				std::vector<INTERNAL::GA_Solution<lrgst_exp>> sample;
				std::copy(
					solutions.begin(),
					solutions.begin() + sample_size,
					std::back_inserter(sample)
				);
				solutions.clear();

				if (count + 1 == num_of_generations)
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
				std::uniform_real_distribution<double> m((1 - mutation_percentage), (1 + mutation_percentage));
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

		template<int lrgst_exp>
		double Function<lrgst_exp>::solve_y(const double& x_val) const noexcept
		{
			std::vector<bool> exceptions;

			for (int i : constants)
				exceptions.push_back(i != 0);

			double ans{ 0 };
			for (int i = lrgst_exp; i >= 0; i--)
			{
				if (exceptions[i])
					ans += constants[i] * POW(x_val, (lrgst_exp - i));
			}

			return ans;
		}
	}
}