
namespace mh
{

    // ---------------------
    // Implementation of DNA
    // ---------------------


    template <typename Scalar>
    template <typename Random>
    GeneticAlgorithm<Scalar>::Dna::Dna(Random& engine, Scalar begin, Scalar end, uint64_t size)
        : code_(vector_type(size)),
          size_(size),
          score_(0.0)
    {
        // Randomly init the dna code
        typename rnd::Random<Scalar>::type distr(begin, end);
        code_.imbue([&]() { return distr(engine); });
    }


    template <typename Scalar>
    template <typename Random>
    GeneticAlgorithm<Scalar>::Dna::Dna(Random& engine, uint64_t size, const Dna& p1, const Dna& p2)
        : code_(vector_type(size)),
          size_(size),
          score_(0.0)
    {
        assert(p1.size_ == size);
        assert(p2.size_ == size);

        std::uniform_int_distribution<> distr(0, 1);
        for (uint64_t i = 0; i < size; ++i)
        {
            if (distr(engine))
            {
                code_(i) = p1[i];
            }
            else
            {
                code_(i) = p2[i];
            }
        }
    }


    // -------------------------
    // Implementation of Options
    // -------------------------


    template <typename Scalar>
    GeneticAlgorithm<Scalar>::Options::Options()
        : popSize_(1000),
          selectionAmount_(100),
          maxEpoc_(1000),
          mutationRate_(0.7),
          beginRange_(std::numeric_limits<Scalar>::min()),
          endRange_(std::numeric_limits<Scalar>::max())
    {
    }

    template <typename Scalar>
    void
    GeneticAlgorithm<Scalar>::Options::setPopSize(unsigned long long v)
    {
        popSize_ = v;
    }

    template <typename Scalar>
    void
    GeneticAlgorithm<Scalar>::Options::setSelectionAmount(unsigned long long v)
    {
        selectionAmount_ = v;
    }

    template <typename Scalar>
    void
    GeneticAlgorithm<Scalar>::Options::setMaxEpoc(unsigned long long v)
    {
        maxEpoc_ = v;
    }

    template <typename Scalar>
    void
    GeneticAlgorithm<Scalar>::Options::setMutationRate(long double v)
    {
        mutationRate_ = v;
    }

    template <typename Scalar>
    void
    GeneticAlgorithm<Scalar>::Options::setBeginRange(Scalar v)
    {
        beginRange_ = v;
    }

    template <typename Scalar>
    void
    GeneticAlgorithm<Scalar>::Options::setEndRange(Scalar v)
    {
        endRange_ = v;
    }


    // ----------------------------------
    // Implementation of GeneticAlgorithm
    // ----------------------------------


    template <typename Scalar>
    GeneticAlgorithm<Scalar>::GeneticAlgorithm(const Options& opt, uint64_t dnaSize)
        : population_(std::vector<dna_type>()),
          dnaSize_(dnaSize),
          options_(opt)
    {
        // Assert on forbiden values
        assert(dnaSize != 0);

        best_ = dna_type(engine_, options_.beginRange_, options_.endRange_, dnaSize);

        // Randomly init population
        population_.reserve(options_.popSize_);
        for (uint64_t i = 0; i < options_.popSize_; ++i)
        {
            population_.emplace_back(dna_type(engine_, options_.beginRange_, options_.endRange_, dnaSize));
        }
    }


    template <typename Scalar>
    template <typename FitnessFunction>
    uint64_t GeneticAlgorithm<Scalar>::run(FitnessFunction& fitness, double target)
    {
        uint64_t epoc = 0;

        while (epoc < options_.maxEpoc_ && best_.getScore() < target)
        {
            std::cout << "Epoc: " << epoc << " - Best: " << best_.getScore() << std::endl;
            step(fitness);
            ++epoc;
        }

        return epoc;
    }


    template <typename Scalar>
    template <typename FitnessFunction>
    void GeneticAlgorithm<Scalar>::step(FitnessFunction& fitness)
    {
        if (population_.empty())
        {
            return;
        }

        // Parallel computation of fitness score
        auto computeFitness = [this, &fitness](const tbb::blocked_range<uint64_t>& r)
        {
            for (uint64_t i = r.begin(); i != r.end(); ++i)
            {
                population_[i].setScore(fitness(population_[i]));
            }
        };
        tbb::parallel_for(tbb::blocked_range<uint64_t>(0, population_.size()), computeFitness);

        // Sort by descending score
        std::sort(population_.begin(), population_.end(),
        [](const dna_type& a, const dna_type& b)
        {
            return a.getScore() > b.getScore();
        });

        // Update the best dna
        if (population_[0].getScore() >= best_.getScore())
        {
            best_ = population_[0];
        }

        // Creation of new generation
        std::uniform_int_distribution<> distr(0, options_.selectionAmount_);
        std::uniform_real_distribution<> selection(0.0, 1.0);
        std::vector<dna_type> new_pop;

        new_pop.reserve(population_.size());

        for (uint64_t i = 0; i < population_.size(); ++i)
        {
            // Cross-over
            new_pop.emplace_back(dna_type(engine_, dnaSize_, population_[distr(engine_)], population_[distr(engine_)]));

            // Mutation
            mutate(new_pop.back());
        }

        population_ = std::move(new_pop);
    }


    template <typename Scalar>
    void GeneticAlgorithm<Scalar>::mutate(Dna& dna)
    {
        std::uniform_real_distribution<double> mutation(0.0, 1.0);
        typename rnd::Random<Scalar>::type distr(options_.beginRange_, options_.endRange_);

        for (uint64_t i = 0; i < dna.getSize(); ++i)
        {
            if (mutation(engine_) <= options_.mutationRate_)
            {
                dna[i] = distr(engine_);
            }
        }
    }
}
