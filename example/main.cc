#include <iostream>
#include <metaheuristic.hh>


int main()
{
    using GA = mh::GeneticAlgorithm<unsigned>;

    // Options
    GA::Options options;
    options.setBeginRange(0);
    options.setEndRange(200);
    options.setPopSize(10000);
    options.setSelectionAmount(50);
    options.setMutationRate(0.1);

    // Genetic Algorithm
    GA ga(options, 100);
    // Optional init of tbb scheduler
    tbb::task_scheduler_init init(8);

    auto fitness = [](const typename GA::dna_type& dna)
    {
        double res = 0.0;
        for (uint64_t i = 0; i < dna.getSize(); ++i)
        {
            if (dna[i] == 42)
            {
                res += 1;
            }
        }

        return res;
    };

    std::cout << "Epocs: " << ga.run(fitness, 90.0) << std::endl;

    return 0;
}
