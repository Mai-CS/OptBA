import time

import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState

import os
import sys

my_lib_path = os.path.abspath("./")
sys.path.append(my_lib_path)

import lstm_module


def objective(trial):
    model = lstm_module.LSTM(trial=trial)
    score = lstm_module.evaluate_model(model)

    return score


start_time = time.time()
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

end_time = time.time()
print("Total time for Optuna: ", end_time - start_time)
