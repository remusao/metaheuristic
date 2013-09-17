#ifndef RANDOM_HH_
# define RANDOM_HH_

# include <type_traits>
# include <random>

namespace rnd
{
    namespace
    {
        template <bool Integral, typename Scalar>
        struct RandomSelector;

        template <typename Scalar>
        struct RandomSelector<true, Scalar>
        {
            typedef std::uniform_int_distribution<Scalar> type;
        };

        template <typename Scalar>
        struct RandomSelector<false, Scalar>
        {
            typedef std::uniform_real_distribution<Scalar> type;
        };
    }

    template <typename Scalar>
    struct Random
    {
        typedef typename RandomSelector<std::is_integral<Scalar>::value, Scalar>::type type;
    };
}

#endif /* !RANDOM_HH_ */
