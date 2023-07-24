#include "neural.h"

float empty_activate_func(float value) { return value; }

float empty_activate_func_d(float value) { return 1.0f; }

float target_func(float x) { return 1 * x * x + 2 * x + 3; }

int main() {
  Neural<float, 3, 1> neural{0.01f, empty_activate_func, empty_activate_func_d};
  float xs[] = {-2, -1, 0, 1, 2};
  neural.train(xs | map([](float x) {
                 return std::make_tuple(std::valarray<float>{x * x, x, 1},
                                        std::valarray<float>{target_func(x)});
               }),
               100, 5);
}