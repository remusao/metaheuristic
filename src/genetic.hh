#ifndef GENETIC_HH_
# define GENETIC_HH_

# include <limits>
# include <random>
# include <assert.h>

# include <armadillo>

namespace mh
{
    template <uint64_t Size, typename Scalar>
    class Dna
    {
        public:
            typedef arma::Col<Scalar> vector_type;

        public:
            // Random init constructor
            Dna()
                : code_(vector_type(Size))
            {
                // Randomly init the code
                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_real_distribution<Scalar> distr(std::numeric_limits<Scalar>::min(), std::numeric_limits<Scalar>::max());
                code_.imbue([&]() { return distr(engine); });
            }

            // Cross-over init constructor
            Dna(const std::vector<Dna>& parents)
                : code_(vector_type(Size))
            {
                assert(parents.size() > 0);
                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_int_distribution<> distr(0, parents.size());
                for (uint64_t i = 0; i < Size; ++i)
                {
                    code_(i) = parents[distr(engine)](i);
                }
            }


            void mutate(double rate)
            {
                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_real_distribution<> mutation(0.0, 1.0);
                std::uniform_real_distribution<Scalar> distr(std::numeric_limits<Scalar>::min(), std::numeric_limits<Scalar>::max());

                for (uint64_t i = 0; i < Size; ++i)
                {
                    if (mutation(engine) <= rate)
                    {
                        code_(i) = distr(engine);
                    }
                }
            }

        private:
            vector_type code_;
            uint64_t    score_;
    };

    class 
    template <typename EvaluationFunction, int range_start, int range_end>
    arma::uvec run_genetic(const arma::uvec& dna, const EvaluationFunction& eval);
}

#endif /* !GENETIC_HH_ */
