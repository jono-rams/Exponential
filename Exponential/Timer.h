#pragma once
#include <ostream>
#include <chrono>

namespace TIMER{
	struct Timer
	{
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<float> duration;

		Timer()
		{
			Reset();
		}

		~Timer()
		{
			
		}

		inline void Reset() noexcept
		{
			start = std::chrono::high_resolution_clock::now();
		}

		void SetEnd() noexcept
		{
			end = std::chrono::high_resolution_clock::now();
			duration = end - start;
		}

		inline float GetTimeInMS() const noexcept
		{
			return float(duration.count() * 1000.f);
		}

		inline float GetTimeInS() const noexcept
		{
			return float(duration.count());
		}
	};
}