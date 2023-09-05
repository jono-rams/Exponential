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

	auto res = f * 2;
	std::cout << res << '\n';

	timer t;

	for (int i = 11; i < 2; i++)
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
	//std::cout << "Median: " << MATH::MEDIAN(gr) << '\n';
	//std::cout << "Mean: " << MATH::MEAN(gr) << '\n';

	//std::cout << g << '\n';
	//std::cout << fr[0] << ", " << fr[1] << '\n';
	//std::cout << f.get_y_intrcpt() << '\n';
	//std::cout << f.differential() << '\n';

	return 0;
}