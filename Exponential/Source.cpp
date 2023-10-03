#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include "Exponential.h"
#include "Timer.h"

using namespace JRAMPERSAD;

template <int n>
using Function = EXPONENTIAL::Function<n>;

typedef TIMER::Timer timer;

template<int exp>
void CalcRoots(std::mutex& m, const Function<exp>& func, EXPONENTIAL::GA_Options options)
{
	m.lock();
	std::cout << "Starting calculation...\n";
	m.unlock();

	timer t;
	auto gr = func.get_real_roots(options);
	t.SetEnd();

	m.lock();
	std::cout << "Time took to calculate approx root values: " << t.GetTimeInS() << "s\n";
	std::cout << "Approximate values of x where y = 0 are: \n";
	std::for_each(gr.begin(), gr.end(),
		[](const auto& val) {
			std::cout << "x:" << val << '\n';
		});
	m.unlock();
}

template<int exp>
void SolveX(std::mutex& m, const Function<exp>& func, EXPONENTIAL::GA_Options options, const double& y)
{
	timer t;
	auto res = func.solve_x(y, options);
	t.SetEnd();

	m.lock();
	std::cout << "Time took to calculate approx x values: " << t.GetTimeInS() << "s\n";
	std::cout << "Approximate values of x where y = " << y << " are: \n";
	std::for_each(res.begin(), res.end(),
		[](const auto& val) {
			std::cout << "x:" << val << '\n';
		});
	m.unlock();
}

int main()
{
	std::vector<int> vec{ 1, 5, 4 };
	Function<2> f{ vec };
	Function<3> g{ { 1, -6, 11, -6 } };

	EXPONENTIAL::GA_Options options;
	options.mutation_percentage = 0.005;
	options.num_of_generations = 10;
	options.sample_size = 1000;
	options.data_size = 100000;
	options.min_range = 4.9;
	options.max_range = 5;

	std::mutex m;
	std::thread th(CalcRoots<3>, std::ref(m), std::cref(g), options);
	//std::thread th1(SolveX<3>, std::ref(m), std::cref(g), options, 5);
	//std::thread th2(SolveX<3>, std::ref(m), std::cref(g), options, 23);

	//CalcRoots<3>(m, g);

	m.lock();
	std::cout << g << " when x = 4.961015\n" << "y = " << g.solve_y(4.961015) << "\n\n";
	//std::cout << g << " when x = 4.30891\n" << "y = " << g.solve_y(4.30891) << "\n\n";
	//std::cout << g << " when x = 2\n" << "y = " << g.solve_y(2) << "\n\n";
	//std::cout << g << " when x = 3\n" << "y = " << g.solve_y(3) << "\n\n";

	//std::cout << "Median: " << MATH::MEDIAN(gr) << '\n';
	//std::cout << "Mean: " << MATH::MEAN(gr) << '\n';

	//std::cout << "Calculating Roots for function f(x) = " << g << '\n';
	//std::cout << "The y-intercept of the function f(x) is " << g.solve_y(0) << '\n';
	std::cout << "dy/dx of f(x) is " << g.differential() << '\n';
	m.unlock();

	th.join();
	//th1.join();
	//th2.join();
	return 0;
}