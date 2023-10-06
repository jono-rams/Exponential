#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include "Exponential.h"
#include "Timer.h"

using namespace JRAMPERSAD;

using EXPONENTIAL::Function;

typedef TIMER::Timer timer;

void CalcRoots(std::mutex& m, const Function& func, EXPONENTIAL::GA_Options options)
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

void SolveX(std::mutex& m, const Function& func, EXPONENTIAL::GA_Options options, const double& y)
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
	std::vector<int64_t> vec{ 1, 5, 4 };
	Function f{2};
	INITIALIZE_EXPO_FUNCTION(f, vec);
	Function g{3};
	INITIALIZE_EXPO_FUNCTION(g, { 1, -6, 11, -6 });

	EXPONENTIAL::GA_Options options;
	options.mutation_percentage = 0.005;
	options.num_of_generations = 1;
	options.sample_size = 1;
	options.data_size = 2;
	options.min_range = 0.13;
	options.max_range = 0.14;

	auto res = (f + g).get_real_roots(options);
	std::for_each(res.begin(), res.end(),
		[](const auto& val) {
			std::cout << "x:" << val << '\n';
		});

	std::cout << (f + g) << " when x = 0.13056\n" << (f + g).solve_y(0.13056);

	std::mutex m;
	//std::thread th(CalcRoots, std::ref(m), std::cref(g), options);
	//std::thread th1(SolveX, std::ref(m), std::cref(g), options, 5);
	//std::thread th2(SolveX, std::ref(m), std::cref(g), options, 23);

	//CalcRoots(m, g);

	m.lock();
	//std::cout << g << " when x = 4.961015\n" << "y = " << g.solve_y(4.961015) << "\n\n";
	//std::cout << g << " when x = 4.30891\n" << "y = " << g.solve_y(4.30891) << "\n\n";
	//std::cout << g << " when x = 2\n" << "y = " << g.solve_y(2) << "\n\n";
	//std::cout << g << " when x = 3\n" << "y = " << g.solve_y(3) << "\n\n";

	//std::cout << "Median: " << MATH::MEDIAN(gr) << '\n';
	//std::cout << "Mean: " << MATH::MEAN(gr) << '\n';

	//std::cout << "Calculating Roots for function f(x) = " << g << '\n';
	//std::cout << "The y-intercept of the function f(x) is " << g.solve_y(0) << '\n';
	//std::cout << "dy/dx of f(x) is " << g.differential() << '\n';
	//std::cout << "f(x) = " << f << std::endl;
	//std::cout << "g(x) = " << g << std::endl;
	//std::cout << "f(x) + g(x) = " << f + g << std::endl;
	m.unlock();

	//th.join();
	//th1.join();
	//th2.join();
	return 0;
}