import matplotlib.pyplot as plt

# Loading airline data
import numpy as np

data = np.load('airline.npz')
X_train, Y_train = data['X_train'], data['Y_train']
D = Y_train.shape[1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Airline passengers ($10^3$)')
ax.plot(X_train.flatten(), Y_train.flatten(), c='b')
ax.set_xticklabels([1949, 1952, 1955, 1958, 1961, 1964])
plt.tight_layout()

from gpflow.kernels import RBF, Cosine, Linear, Bias, Matern52
from gpflow import transforms
from gpflow.gpr import GPR

Q = 10  # nr of terms in the sum
max_iters = 1000


# Trains a model with a spectral mixture kernel, given an ndarray of 2Q frequencies and lengthscales
def create_model(hypers):
    f = np.clip(hypers[:Q], 0, 5)
    weights = np.ones(Q) / Q
    lengths = hypers[Q:]

    kterms = []
    for i in range(Q):
        rbf = RBF(D, lengthscales=lengths[i], variance=1. / Q)
        rbf.lengthscales.transform = transforms.Exp()
        cos = Cosine(D, lengthscales=f[i])
        kterms.append(rbf * cos)

    k = np.sum(kterms) + Linear(D) + Bias(D)
    return GPR(X_train, Y_train, kern=k)


X_test, X_complete = data['X_test'], data['X_complete']


def plotprediction(m):
    # Perform prediction
    mu, var = m.predict_f(X_complete)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Airline passengers ($10^3$)')
    ax.set_xticklabels([1949, 1952, 1955, 1958, 1961, 1964, 1967, 1970, 1973])
    ax.plot(X_train.flatten(), Y_train.flatten(), c='b')
    ax.plot(X_complete.flatten(), mu.flatten(), c='g')
    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)
    ax.plot(X_complete, upper, 'g--', X_complete, lower, 'g--', lw=1.2)
    ax.fill_between(X_complete.flatten(), lower.flatten(), upper.flatten(),
                    color='g', alpha=.1)
    plt.tight_layout()


m = create_model(np.ones((2 * Q,)))
m.optimize(maxiter=max_iters)
plotprediction(m)

from gpflowopt.domain import ContinuousParameter
from gpflowopt.objective import batch_apply


# Objective function for our optimization
# Input: N x 2Q ndarray, output: N x 1.
# returns the negative log likelihood obtained by training with given frequencies and rbf lengthscales
# Applies some tricks for stability similar to GPy's jitchol
@batch_apply
def objectivefx(freq):
    m = create_model(freq)
    for i in [0] + [10 ** exponent for exponent in range(6, 1, -1)]:
        try:
            mean_diag = np.mean(np.diag(m.kern.compute_K_symm(X_train)))
            m.likelihood.variance = 1 + mean_diag * i
            m.optimize(maxiter=max_iters)
            return -m.compute_log_likelihood()
        except:
            pass
    raise RuntimeError("Frequency combination failed indefinately.")


# Setting up optimization domain.
lower = [0.] * Q
upper = [5.] * int(Q)
df = np.sum([ContinuousParameter('freq{0}'.format(i), l, u) for i, l, u in zip(range(Q), lower, upper)])

lower = [1e-5] * Q
upper = [300] * int(Q)
dl = np.sum([ContinuousParameter('l{0}'.format(i), l, u) for i, l, u in zip(range(Q), lower, upper)])
domain = df + dl

from gpflowopt.design import LatinHyperCube
from gpflowopt.acquisition import ExpectedImprovement
from gpflowopt import optim, BayesianOptimizer

design = LatinHyperCube(6, domain)
X = design.generate()

Y = objectivefx(X)
m = GPR(X, Y, kern=Matern52(domain.size, ARD=False))
ei = ExpectedImprovement(m)
opt = optim.StagedOptimizer([optim.MCOptimizer(domain, 5000), optim.SciPyOptimizer(domain)])
optimizer = BayesianOptimizer(domain, ei, optimizer=opt)
with optimizer.silent():
    result = optimizer.optimize(objectivefx, n_iter=24)

m = create_model(result.x[0, :])
m.optimize()
plotprediction(m)

f, axes = plt.subplots(1, 1, figsize=(7, 5))
f = ei.data[1][:,0]
axes.plot(np.arange(0, ei.data[0].shape[0]), np.minimum.accumulate(f))
axes.set_ylabel('fmin')
axes.set_xlabel('Number of evaluated points')
plt.show()