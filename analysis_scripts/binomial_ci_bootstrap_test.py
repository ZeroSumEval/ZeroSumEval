import numpy as np

n = 20
k = 9
iters = 1000

observations = np.array([1]*k + [0]*(n-k))

means = []
for _ in range(iters):
    samples = np.random.choice(observations, n, replace=True)
    means.append(np.mean(samples))

means.sort()
print(means[int(0.025 * iters)])
print(means[int(0.500 * iters)])
print(means[int(0.975 * iters)])