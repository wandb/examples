import optuna


def objective(trial):
    database = trial.suggest_categorical('database', ['small', 'medium', 'large'])

    randomize = None  # init for printing

    if database in ['small', 'large']:
        batch_size = trial.suggest_int('batch_size', 16, 64)

        # maybe test randomization when the batch_size is small
        randomize = trial.suggest_categorical('randomize', [True, False])
    else:
        batch_size = trial.suggest_int('batch_size', 64, 256)

    print(f"{database=} {batch_size=} {randomize=}")

    return -1


def sampler():
    return optuna.samplers.NSGAIISampler(
        population_size=100,
        crossover_prob=0.2,
        seed=1000000,
    )


def pruner():
    return optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(), patience=1
    )