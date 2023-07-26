#pragma once

#include <execution>
#include <random>

#include "tools.h"

template <class DataType, std::size_t... EachLayerSize>
class Neural {
 public:
  static_assert(sizeof...(EachLayerSize) > 1);
  static constexpr std::array each_layer_size = {EachLayerSize...};
  static std::uniform_real_distribution<DataType> urd;

  std::array<std::valarray<DataType>, each_layer_size.size()> layers;
  std::array<std::valarray<DataType>, each_layer_size.size()> grads;
  std::array<std::valarray<DataType>, each_layer_size.size() - 1> weights;

  DataType learning_rate;
  DataType (*activate_func)(DataType);
  DataType (*activate_func_d)(DataType);

  Neural(DataType learning_rate, DataType activate_func(DataType),
         DataType activate_func_d(DataType))
      : layers{std::valarray<DataType>(EachLayerSize)...},
        grads{std::valarray<DataType>(EachLayerSize)...},
        weights{std::apply(
            [](auto &&...weight) {
              return std::array{std::valarray<DataType>(weight.size())...};
            },
            typename multi_array<DataType, EachLayerSize...>::type())},
        learning_rate{learning_rate},
        activate_func{activate_func},
        activate_func_d{activate_func_d} {
    std::default_random_engine gen{std::random_device{}()};
    for (auto &weight : weights) {
      std::generate_n(std::execution::par, std::begin(weight), weight.size(),
                      [&]() { return 0.01f * urd(gen); });
    }
  }

  void forward(std::valarray<DataType> input_data) {
    std::fill(layers.begin(), layers.end(), DataType{});
    layers[0] = input_data;
#pragma unroll
    for (auto [layer_left, layer_right, weight] :
         zip(layers, layers | drop<1>(), weights)) {
      for (auto i : iota(layer_right.size())) {
        std::valarray<DataType> wl{
            weight[std::slice(i * layer_left.size(), layer_left.size(), 1)]};
        layer_right[i] = activate_func((wl * layer_left).sum());
      }
    }
  }

  void backward(std::valarray<DataType> ideal_output) {
    std::fill(grads.begin(), grads.end(), DataType{});
    grads.back() =
        2.0 / each_layer_size.back() * (layers.back() - ideal_output);
#pragma unroll
    for (auto [layer_left, layer_right, grad_left, grad_right, weight] :
         zip(layers | reverse | drop<1>(), layers | reverse,
             grads | reverse | drop<1>(), grads | reverse, weights | reverse)) {
      for (auto i : iota(grad_left.size())) {
        std::valarray<DataType> trans_wl{
            weight[std::slice(i, layer_right.size(), layer_left.size())]};
        grad_left[i] =
            (layer_right.apply(activate_func_d) * grad_right * trans_wl).sum();
      }
      for (auto [i, lr, gr] :
           zip(iota(grad_right.size()), layer_right, grad_right)) {
        weight[std::slice(i * layer_left.size(), layer_left.size(), 1)] -=
            learning_rate * layer_left *
            std::valarray<DataType>(lr, layer_left.size())
                .apply(activate_func_d) *
            std::valarray<DataType>(gr, layer_left.size());
      }
    }
  }

  template <class DataSet>
  void train(range<DataSet> data, std::size_t times,
             std::size_t log_times = 1) {
    std::size_t gap = log_times <= 0      ? times
                      : log_times > times ? 1
                                          : times / log_times;
    for (auto i : iota(times)) {
      for (auto [input_data, ideal_output] : data) {
        forward(input_data);
        backward(ideal_output);
      }
      if ((i + 1) % gap == 0)
        std::cout << "loss: " << std::sqrt(std::pow(grads.back(), 2).sum())
                  << ' ' << "weights: " << weights << '\n';
    }
  }
};

template <class DataType, std::size_t... EachLayerSize>
std::uniform_real_distribution<DataType>
    Neural<DataType, EachLayerSize...>::urd{-1, 1};
