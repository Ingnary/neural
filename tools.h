#pragma once

#include <array>
#include <iostream>
#include <tuple>
#include <valarray>

#pragma region multi_array
template <class DataType, int... Is>
struct multi_array;

template <class DataType, int First, int Second, int... Rest>
struct multi_array<DataType, First, Second, Rest...> {
  using type = decltype(std::tuple_cat(
      std::tuple<std::array<DataType, First * Second>>(),
      typename multi_array<DataType, Second, Rest...>::type()));
};

template <class DataType, int Last>
struct multi_array<DataType, Last> {
  using type = std::tuple<>;
};
#pragma endregion multi_array

#pragma region tuple
template <class Tuple1, class Tuple2, std::size_t... Is>
constexpr bool tuple_compare(Tuple1 const& tuple1, Tuple2 const& tuple2,
                             std::index_sequence<Is...>) {
  return ((std::get<Is>(tuple1) != std::get<Is>(tuple2)) && ...);
}
#pragma endregion tuple

#pragma region range
template <class It>
struct range {
  It m_begin;
  It m_end;

  constexpr range(It begin, It end)
      : m_begin{std::move(begin)}, m_end{std::move(end)} {}

  constexpr It begin() const { return m_begin; }
  constexpr It end() const { return m_end; }
};
template <class It>
range(It, It) -> range<It>;

template <class F>
struct pipable {
  F m_f;

  constexpr pipable(F f) : m_f{std::move(f)} {}

  template <class... Rs>
  constexpr decltype(auto) operator()(Rs&&... rs) const {
    return m_f(range{std::begin(rs), std::end(rs)}...);
  }

  template <class R>
  friend constexpr decltype(auto) operator|(R&& r, pipable const& self) {
    return self(std::forward<decltype(r)>(r));
  }
};

static constexpr auto reverse = pipable{[](auto&& r) {
  return range{std::make_reverse_iterator(std::end(r)),
               std::make_reverse_iterator(std::begin(r))};
}};

template <std::size_t Count>
static constexpr auto drop() {
  return pipable{[=](auto&& r) {
    return range{std::next(std::begin(r), Count), std::end(r)};
  }};
}

struct iota_iterator {
  std::size_t index{};

  constexpr decltype(auto) operator*() const { return index; }

  constexpr iota_iterator& operator++() {
    ++index;
    return *this;
  }

  constexpr bool operator!=(iota_iterator const& that) const {
    return index != that.index;
  }

  constexpr bool operator==(iota_iterator const& that) const {
    return index == that.index;
  }
};
static constexpr auto iota = [](std::size_t stop) {
  return range{iota_iterator{0}, iota_iterator{stop}};
};

template <class Base>
struct enumerate_iterator {
  Base m_it;
  std::size_t m_index{};

  constexpr decltype(auto) operator*() const {
    return std::pair<std::size_t, decltype(*m_it)>(m_index, *m_it);
  }

  constexpr enumerate_iterator& operator++() {
    ++m_it;
    ++m_index;
    return *this;
  }

  constexpr bool operator!=(enumerate_iterator const& that) const {
    return m_it != that.m_it;
  }
};
template <class Base>
enumerate_iterator(Base) -> enumerate_iterator<Base>;
static constexpr auto enumerate = pipable([](auto&& r) {
  return range(enumerate_iterator{std::begin(r)},
               enumerate_iterator{std::end(r)});
});

template <class Base, class Func>
struct map_iterator {
  Base m_it;
  Func m_func;

  constexpr decltype(auto) operator*() const { return m_func(*m_it); }

  constexpr map_iterator& operator++() {
    ++m_it;
    return *this;
  }

  constexpr bool operator!=(map_iterator const& that) const {
    return m_it != that.m_it;
  }
};
template <class Func, class Base>
map_iterator(Base, Func) -> map_iterator<Base, Func>;
template <class Func>
static constexpr auto map(Func&& func) {
  return pipable{[=](auto&& r) {
    return range{map_iterator{std::begin(r), func},
                 map_iterator{std::end(r), func}};
  }};
}

template <class... Bases>
struct zip_iterator {
  std::tuple<Bases...> m_it;

  constexpr decltype(auto) operator*() const {
    return std::apply(
        [](auto&&... it) { return std::tuple<decltype(*it)...>{*it...}; },
        m_it);
  }

  constexpr zip_iterator& operator++() {
    std::apply([](auto&&... it) { (++it, ...); }, m_it);
    return *this;
  }

  constexpr bool operator!=(zip_iterator const& that) const {
    static_assert(sizeof...(Bases) == std::tuple_size_v<decltype(that.m_it)>);
    return tuple_compare(m_it, that.m_it,
                         std::make_index_sequence<sizeof...(Bases)>{});
  }
};
template <class... Bases>
zip_iterator(std::tuple<Bases...>) -> zip_iterator<Bases...>;
static constexpr auto zip = pipable{[](auto&&... r) {
  return range(zip_iterator{std::make_tuple(std::begin(r)...)},
               zip_iterator{std::make_tuple(std::end(r)...)});
}};

template <class T>
struct is_range {
  template <class U>
  static constexpr auto test(U* u)
      -> decltype(std::begin(*u), std::end(*u), u->size(), std::true_type{});

  template <class U>
  static constexpr auto test(...) -> std::false_type;

  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <class, template <class...> class>
struct is_instance : std::false_type {};
template <class... T, template <class...> class U>
struct is_instance<U<T...>, U> : std::true_type {};

template <class Range>
constexpr auto operator<<(std::ostream& os, Range const& r)
    -> std::enable_if_t<is_range<Range>::value &&
                            !is_instance<Range, std::basic_string>::value,
                        std::ostream&> {
  os << '{';
  for (auto [i, element] : enumerate(r))
    os << element << (i + 1 == r.size() ? "" : ", ");
  os << '}';
  return os;
}
#pragma endregion range
