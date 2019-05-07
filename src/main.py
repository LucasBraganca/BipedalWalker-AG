import argparse
import heapq
import pickle
import random
from copy import deepcopy

import gym

from src.Individual import Individual


def evaluate(individual, epsodio_size=500, is_render=False):
    env = gym.make('BipedalWalker-v2')
    env.reset()
    score = 0
    steps = 0
    size = len(individual.loop_actions)

    for a in individual.init_actions:
        observation, reward, done, info = env.step(a)
        if is_render:
            env.render()
        score += reward
        steps += 1

    for i in range(epsodio_size):
        idx = (i % size)
        observation, reward, done, info = env.step(individual.loop_actions[idx])
        if is_render:
            env.render()
        score += reward
        steps += 1
        if score < -100:
            break

    env.close()
    return score, steps


def get_k_best(population, k):
    best = heapq.nlargest(k, population)
    return best


def selection(population, n, tournsize):
    chosen = []
    for i in range(n):
        aspirants = [random.choice(population) for _ in range(tournsize)]
        chosen.append(max(aspirants))

    return chosen


def crossover_uniform(ind1, ind2, pc):
    for i in range(len(ind1.init_actions)):
        if random.random() < pc:
            ind1.init_actions[i], ind2.init_actions[i] = ind2.init_actions[i], ind1.init_actions[i]

    for i in range(len(ind1.loop_actions)):
        if random.random() < pc:
            ind1.loop_actions[i], ind2.loop_actions[i] = ind2.loop_actions[i], ind1.loop_actions[i]

    return ind1, ind2


def crossover_1P(ind1, ind2):
    size = min(len(ind1.init_actions), len(ind2.init_actions))
    cxpoint = random.randint(1, size - 1)
    ind1.init_actions[cxpoint:], ind2.init_actions[cxpoint:] = ind2.init_actions[cxpoint:], ind1.init_actions[cxpoint:]

    size = min(len(ind1.loop_actions), len(ind2.loop_actions))
    cxpoint = random.randint(1, size - 1)
    ind1.loop_actions[cxpoint:], ind2.loop_actions[cxpoint:] = ind2.loop_actions[cxpoint:], ind1.loop_actions[cxpoint:]

    return ind1, ind2


def crossover_2P(ind1, ind2):
    size = min(len(ind1.init_actions), len(ind2.init_actions))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1.init_actions[cxpoint1:cxpoint2], ind2.init_actions[cxpoint1:cxpoint2] \
        = ind2.init_actions[cxpoint1:cxpoint2], ind1.init_actions[cxpoint1:cxpoint2]

    size = min(len(ind1.loop_actions), len(ind2.loop_actions))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1.loop_actions[cxpoint1:cxpoint2], ind2.loop_actions[cxpoint1:cxpoint2] \
        = ind2.loop_actions[cxpoint1:cxpoint2], ind1.loop_actions[cxpoint1:cxpoint2]

    return ind1, ind2


def mutate(individual, pb_mut_action=1.0):
    for i in range(len(individual.init_actions)):
        for j in range(4):
            if random.random() < pb_mut_action:
                individual.init_actions[i][j] += random.uniform(-0.1, 0.1)

    for i in range(len(individual.loop_actions)):
        for j in range(4):
            if random.random() < pb_mut_action:
                individual.loop_actions[i][j] += random.uniform(-0.2, 0.2)

    return individual


def create_population(size):
    return [Individual() for _ in range(size)]


def statistics(generation, population, recal_ind):
    all_fitness_score = [ind.fitness_score for ind in population]
    all_fitness_steps = [ind.fitness_steps for ind in population]

    length = len(population)
    mean1 = sum(all_fitness_score) / length
    sum2 = sum(x * x for x in all_fitness_score)
    std1 = abs(sum2 / length - mean1 ** 2) ** 0.5

    mean2 = sum(all_fitness_steps) / length
    sum2 = sum(x * x for x in all_fitness_steps)
    std2 = abs(sum2 / length - mean2 ** 2) ** 0.5

    best_ind = max(population)
    worst_ind = min(population)

    print("Generation %d:" % (generation + 1))
    print(" Evaluated %i individuals" % recal_ind)
    print(" Best individual is: %s" % (best_ind.fitness_score))
    print(" Worst individual is: %s" % (worst_ind.fitness_score))
    print(" Fitness Min: %s , %s" % (worst_ind.fitness_score, worst_ind.fitness_steps))
    print(" Fitness Max: %s , %s" % (best_ind.fitness_score, best_ind.fitness_steps))
    print(" Fitness Avg: %s , %s" % (mean1, mean2))
    print(" Fitness Std: %s , %s" % (std1, std2))
    print()

    f = open('data_graph.txt', 'a')
    f.write('%d,%f\n' % (generation + 1, best_ind.fitness_score))
    f.close()


def evolution(num_gen=100, pop_size=100, pb_cx=0.4, pb_mut=0.3, print_statistics=True):
    mate = crossover_1P

    population = create_population(pop_size * 4)

    for ind in population:
        score, steps = evaluate(ind)
        ind.fitness_score = score
        ind.fitness_steps = steps

    population = get_k_best(population, pop_size)

    for g in range(num_gen):
        offspring = selection(population, len(population), 5)
        offspring = list(map(deepcopy, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < pb_cx:
                mate(child1, child2)
                child1.fitness_score = None
                child2.fitness_score = None
        for mutant in offspring:
            if random.random() < pb_mut:
                mutate(mutant, 0.6)
                mutant.fitness_score = None

        recal_ind = [ind for ind in offspring if not ind.fitness_score]

        for ind in recal_ind:
            score, steps = evaluate(ind)
            ind.fitness_score = score
            ind.fitness_steps = steps

        population[:] = offspring

        if print_statistics:
            statistics(g, population, len(recal_ind))

    return max(population)


def show_best():
    with open('best.dat', 'rb') as file:
        best = pickle.load(file)
        evaluate(best,is_render=True)
        file.close()


if __name__ == "__main__":
    random.seed(0)
    parser = argparse.ArgumentParser(
        description='Executa algoritmo genético para controle do BipedalWalker do OpenIA/gym.')
    parser.add_argument('--run', required=True)
    args = vars(parser.parse_args())

    if args['run'] == 'ag':
        evolution()
    elif args['run'] == 'show':
        show_best()
    else:
        print('Argumento inválido!')
