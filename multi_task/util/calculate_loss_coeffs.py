# This was a failed experiment, not actually used
import numpy as np
approximate_avg_loss_per_task = np.array([0.15, 0.4, 0.45, 0.35, 0.05, 0.15, 0.45, 0.4, 0.3, 0.15, 0.1, 0.35, 0.25, 0.15, 0.1,
                                0.05, 0.1, 0.05, 0.25, 0.35, 0.1, 0.25, 0.1, 0.25, 0.15, 0.55, 0.1, 0.5, 0.15, 0.15,
                                0.1, 0.25, 0.35, 0.4, 0.3, 0.05, 0.2, 0.3, 0.15, 0.4])

weights = (1 / approximate_avg_loss_per_task)
weights /= weights.sum()

json_weights = dict(map(lambda kv: (str(kv[0]), kv[1]), enumerate(weights)))
print(json_weights)