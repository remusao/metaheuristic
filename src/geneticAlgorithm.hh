#ifndef GENETIC_ALGORITHM_HH_
# define GENETIC_ALGORITHM_HH_

# include <assert.h>
# include <limits>
# include <vector>
# include <algorithm>

# include "tbb/tbb.h"
# include "random.hh"

# include <armadillo>

namespace mh
{
    // ----------------------
    // GeneticAlgorithm class
    // ----------------------

    template <typename Scalar>
    class GeneticAlgorithm
    {
        private:

            //
            // Dna type
            //

            struct Dna
            {
                public:
                    typedef arma::Col<Scalar> vector_type;

                public:
                    // Default constructor
                    Dna() = default;

                    // Random init constructor
                    template <typename Random>
                    Dna(Random& engine, Scalar begin, Scalar end, uint64_t size);

                    // Cross-over init constructor
                    template <typename Random>
                    Dna(Random& engine, uint64_t size, const Dna& p1, const Dna& p2);

                    // Copy
                    Dna(const Dna&) = default;
                    Dna& operator=(const Dna&) = default;

                    // Move
                    Dna(Dna&&) = default;
                    Dna& operator=(Dna&&) = default;

                    // Accessors
                    const Scalar& operator[](int i) const { return code_[i]; }
                    Scalar& operator[](int i) { return code_[i]; }

                    uint64_t getSize() const { return size_; }
                    double getScore() const { return score_; }
                    void setScore(double score) { score_ = score; }

                    // Get internal representation
                    vector_type& getCode() { return code_; }
                    const vector_type& getCode() const { return code_; }

                private:
                    vector_type code_;
                    uint64_t    size_;
                    double      score_;
            };

        public:

            //
            // Option Type
            //

            class Options
            {
                public:
                    // Default
                    Options();

                    void setPopSize(unsigned long long);
                    void setSelectionAmount(unsigned long long);
                    void setMaxEpoc(unsigned long long);
                    void setMutationRate(long double);
                    void setBeginRange(Scalar);
                    void setEndRange(Scalar);

                private:

                    friend GeneticAlgorithm;

                    unsigned long long  popSize_;
                    unsigned long long  selectionAmount_;
                    unsigned long long  maxEpoc_;
                    long double         mutationRate_;
                    Scalar              beginRange_;
                    Scalar              endRange_;
            };

            typedef Dna dna_type;

        public:
            GeneticAlgorithm(const Options& options, uint64_t dna_size);

            // Copy
            GeneticAlgorithm(const GeneticAlgorithm&) = delete;
            GeneticAlgorithm& operator=(const GeneticAlgorithm&) = delete;

            // Move
            GeneticAlgorithm(GeneticAlgorithm&&) = default;
            GeneticAlgorithm& operator=(GeneticAlgorithm&&) = default;

            // Methods
            template <typename FitnessFunction, typename TargetFunction>
            typename dna_type::vector_type
            run(FitnessFunction& fitness, TargetFunction& target);

            // Accessors
            uint64_t getDnaSize() const { return dnaSize_; }
            const dna_type& getBest() const { return best_; }
            double getBestScore() const { return best_.getScore(); }

        private:
            // One step of the genetic algorithm
            template <typename FitnessFunction>
            void step(FitnessFunction& fitness);

            // Mutate dna
            void mutate(Dna& dna);

            // Attributes
            std::vector<dna_type>   population_;
            uint64_t                dnaSize_;
            Options                 options_;
            dna_type                best_;
            std::mt19937_64         engine_;
    };
}

// Implementation
# include "geneticAlgorithm.hxx"

#endif /* !GENETIC_ALGORITHM_HH_ */
