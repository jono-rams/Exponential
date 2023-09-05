#include <iostream>
#include <memory>
#include "FunctionsTemplate.h"
#include "Timer.h"

template <int n>
using Function = MATH::EXP::Function<n>;

typedef TIMER::Timer timer;

int main()
{
	std::vector<int> vec{ 1, 5, 4 };
	Function<2> f{ vec };
	Function<3> g{ { 1, -6, 11, -6 } };

	timer t;

	for (int i = 1; i < 2; i++)
	{
		t.Reset();
		auto gr = g.get_real_roots_ga(-100, 100, i, 1000, 100000, 0.005);
		t.SetEnd();

		std::cout << "Time took: " << t.GetTimeInS() << "s\n";


		std::for_each(gr.begin(), gr.end(),
			[](const auto& val) {
				std::cout << "x:" << val << '\n';
			});
	}

	std::cout << g << " when x = 1\n" << "y = " << g.solve_y(1) << "\n\n";
	std::cout << g << " when x = 2\n" << "y = " << g.solve_y(2) << "\n\n";
	std::cout << g << " when x = 3\n" << "y = " << g.solve_y(3) << "\n\n";

	//std::cout << "Median: " << MATH::MEDIAN(gr) << '\n';
	//std::cout << "Mean: " << MATH::MEAN(gr) << '\n';

	//std::cout << g << '\n';
	//std::cout << fr[0] << ", " << fr[1] << '\n';
	//std::cout << f.get_y_intrcpt() << '\n';
	//std::cout << f.differential() << '\n';

	return 0;
}