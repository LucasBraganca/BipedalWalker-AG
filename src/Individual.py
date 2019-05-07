import random

class Individual:
    def __init__(self, ini_actions_size = 12, loop_actions_size = 14):
        self.init_actions = [self.create_random_action() for _ in range(ini_actions_size)]
        self.loop_actions = [self.create_random_action() for _ in range(loop_actions_size)]
        self.fitness_score = 0.0
        self.fitness_steps = 0

    def __repr__(self):
        return '%s %s %s %s' % (self.fitness_score, self.fitness_steps,self.init_actions,self.loop_actions)

    def __lt__(self, other):
        if round(self.fitness_score) == round(other.fitness_score):
            if self.fitness_score < 0.0:
                return self.fitness_steps > other.fitness_steps
            else:
                return self.fitness_steps < other.fitness_steps

        return self.fitness_score < other.fitness_score

    def __gt__(self, other):
        if round(self.fitness_score) == round(other.fitness_score):
            if self.fitness_score < 0.0:
                return self.fitness_steps < other.fitness_steps
            else:
                return self.fitness_steps > other.fitness_steps

        return self.fitness_score > other.fitness_score

    def __eq__(self, other):
        return self.fitness_score == other.fitness_score

    def create_random_action(self):
        return [random.uniform(-1, 1) for _ in range(4)]
