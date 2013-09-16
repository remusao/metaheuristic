#ifndef GENETIC_HH_
# define GENETIC_HH_

# include <assert.h>
# include <limits>
# include <random>
# include <vector>
# include <algorithm>

# include <armadillo>

namespace mh
{
    template <typename Scalar,
              Scalar   begin_range = std::numeric_limits<Scalar>::min(),
              Scalar   end_range = std::numeric_limits<Scalar>::max()>
    class Dna
    {
        public:
            typedef arma::Col<Scalar> vector_type;

        public:
            // Random init constructor
            Dna(uint64_t size)
                : code_(vector_type(size)),
                  size_(size),
                  score_(0.0)
            {
                // Randomly init the dna code
                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_int_distribution<Scalar> distr(begin_range, end_range);
                code_.imbue([&]() { return distr(engine); });
            }

            // Cross-over init constructor
            Dna(uint64_t size, const std::vector<Dna>& parents)
                : code_(vector_type(size)),
                  size_(size),
                  score_(0.0)
            {
                assert(parents.size() > 0);
                assert(parents[0].size_ == size);

                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_int_distribution<> distr(0, parents.size());
                for (uint64_t i = 0; i < size; ++i)
                {
                    code_(i) = parents[distr(engine)](i);
                }
            }

            // Copy
            Dna(const Dna&) = default;
            Dna& operator=(const Dna&) = default;

            // Move
            Dna(Dna&&) = default;
            Dna& operator=(Dna&&) = default;

            // Methods
            void mutate(double rate)
            {
                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_real_distribution<double> mutation(0.0, 1.0);
                std::uniform_real_distribution<Scalar> distr(begin_range, end_range);

                for (uint64_t i = 0; i < size_; ++i)
                {
                    if (mutation(engine) <= rate)
                    {
                        code_(i) = distr(engine);
                    }
                }
            }

            // Accessors
            const vector_type& getCode() const { return code_; }
            uint64_t getSize() const { return size_; }
            double getScore() const { return score_; }

            void setScore(double score) { score_ = score; }

        private:
            vector_type code_;
            uint64_t    size_;
            double      score_;
    };


    template <typename DNA>
    class GeneticAlgorithm
    {
        public:
            GeneticAlgorithm(
                uint64_t pop_size,
                uint64_t dna_size,
                uint64_t selection_amount,
                double mutation_rate)
                : population_(std::vector<DNA>()),
                  dna_size_(dna_size),
                  selection_(selection_amount),
                  mutation_rate_(mutation_rate),
                  best_(DNA(dna_size))
            {
		population_.reserve(pop_size);
                std::generate_n(population_.begin(), pop_size, [dna_size]()
                {
                    return DNA(dna_size);
                });
            }

            // Copy
            GeneticAlgorithm(const GeneticAlgorithm&) = delete;
            GeneticAlgorithm& operator=(const GeneticAlgorithm&) = delete;

            // Move
            GeneticAlgorithm(GeneticAlgorithm&&) = default;
            GeneticAlgorithm& operator=(GeneticAlgorithm&&) = default;

            // Methods
            template <typename FitnessFunction>
            void step(FitnessFunction fitness)
            {
                // Compute fitness
                for (auto& dna: population_)
                {
                    dna.setScore(fitness(dna));
                }

                // Sort by score
                std::sort(population_.begin(), population_.end(),
                [](const DNA& a, const DNA& b)
                {
                    return a.getScore() < b.getScore();
                });

                // Update the best dna
                if (population_[0].getScore() >= best_.getScore())
                {
                    best_ = population_[0];
                }

                // Creation of new generation
                std::mt19937 engine;  // Mersenne twister random number engine
                std::uniform_int_distribution<> distr(0, selection_);
                std::uniform_real_distribution<> selection(0.0, 1.0);
                std::vector<DNA> new_pop(population_.size());

                for (int i = 0; i < population_.size(); ++i)
                {
                    // Cross-over
                    if (selection(engine) > 0.5)
                    {
                        new_pop.emplace_back(DNA({population_[distr(engine)], population_[distr(engine)]}));
                    }
                    else // Mutation
                    {
                        new_pop.emplace_back(std::move(population_[distr(engine)]));
                        new_pop.back().mutate(mutation_rate_);
                    }
                }

                population_ = std::move(new_pop);
            }


            // Accessors
            uint64_t getDnaSize() const { return dna_size_; }
            uint64_t getSelectionAmount() const { return selection_; }
            double   getMutationRate() const { return mutation_rate_; }
            const DNA& getBest() const { return best_; }
            double getBestScore() const { return best_.getScore(); }

        private:
            std::vector<DNA>    population_;
            uint64_t            dna_size_;
            uint64_t            selection_;
            double              mutation_rate_;
            DNA                 best_;
    };
}

#endif /* !GENETIC_HH_ */
