#include <metaheuristic.hh>


int main()
{
    mh::GeneticAlgorithm<mh::Dna<unsigned>> ga(1000, 5, 100, 0.5);
    auto fitness = [](const mh::Dna<unsigned>& dna)
    {
        double res;
        for (uint64_t i = 0; i < dna.getSize(); ++i)
        {
            if (dna.getCode()(i) == 42)
            {
                res += 1;
            }
        }

        return res;
    };

    while (ga.getBestScore() < 4.0)
    {
        ga.step(fitness);
    }

    return 0;
}
