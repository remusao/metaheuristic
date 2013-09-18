#include <iostream>
#include <metaheuristic.hh>


int main()
{
    using GA = mh::GeneticAlgorithm<double>;

    // Options
    GA::Options options;
    options.setBeginRange(-2.48);
    options.setEndRange(2.48);
    options.setPopSize(10000);
    options.setSelectionAmount(100);
    options.setMutationRate(0.2);

    uint64_t n = 20;

    // Genetic Algorithm
    GA ga(options, n);
    // Optional init of tbb scheduler
    tbb::task_scheduler_init init(8);

    // De Jong function
    auto fitness = [](const typename GA::dna_type& dna)
    {
        return (1 - arma::dot(dna.getCode(), dna.getCode()));
    };

    // Rotated hyper-ellipsoid function
    auto fitness2 = [](const typename GA::dna_type& dna)
    {
        double res = 0;
        for (uint64_t i = 0; i < dna.getSize(); ++i)
        {
            for (uint64_t j = 0; j < i; ++j)
            {
                res += dna[j] * dna[j];
            }
        }
        return (1 - res);
    };

    // Rosenbrock's valley
    auto fitness3 = [](const typename GA::dna_type& dna)
    {
        double res = 0;
        for (uint64_t i = 0; i < (dna.getSize() - 1); ++i)
        {
            auto t1 = 100 * (dna[i + 1] - dna[i] * dna[i]);
            auto t2 = 1 - dna[i];
            res += t1 * t1 + t2 * t2;
        }
        return (1 - res);
    };

    // Rastrigin's function
    auto fitness4 = [](const typename GA::dna_type& dna)
    {
        return 1 - (10 * dna.getSize() + arma::dot(dna.getCode(), dna.getCode()) - 10 * arma::accu(arma::cos(2 * arma::datum::pi * dna.getCode())));
    };

    auto target = [](double score) -> bool
    {
        return (int)score == 0;
    };

    std::cout << "Dna:\n" << ga.run(fitness4, target) << std::endl;

    return 0;
}
