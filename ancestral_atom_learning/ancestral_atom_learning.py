import numpy
from sklearn import linear_model
import time


class AncestralAtomLearning(object):
    def __init__(self, init_ancestor, extract_operators, sc):
        self._ancestor = numpy.copy(init_ancestor)

        if self._ancestor.ndim == 1:
            self._ancestor = self._ancestor[:, numpy.newaxis]
        self._extract_operators = extract_operators
        self._sc = sc

        # list to store dictionry
        self._ancestor_history = []

        self._error_history = []
        self._last_coef = None

    def _orthogonal_quasi_geodesic(self, G, X, t):
        # alpha is fixed to 2

        U = numpy.concatenate((G, X), axis=1)
        V = numpy.concatenate((X, -G), axis=1)

        size = V.shape[1]
        inv = numpy.linalg.inv(numpy.eye(size) + t/2 * numpy.dot(V.T, U))

        VtX = numpy.dot(V.T, X)
        # Y = X - t U*inv*V.T*X
        Y = X - t * numpy.dot(numpy.dot(U, inv), VtX)

        return Y

    def _gen_D(self, index=None):
        if index is None:
            ancestor = self._ancestor
        else:
            ancestor = self._ancestor_history[index]
        D = numpy.column_stack([numpy.dot(operator, ancestor) for operator in self._extract_operators])

        return D

    def _update_ancestors(self, y, coef, op, diff, stepsize, learning_decay=0.9999,
            orthogonal_constraint=False, decorrelation=False, decorrelation_coefficient=1e-6):
        # shuffle order of sample data
        idx = numpy.arange(y.shape[1])
        numpy.random.shuffle(idx)

        num_ancestor = self._ancestor.shape[1]
        eta = stepsize
        for j in idx:
            gradient = numpy.zeros_like(self._ancestor)
            correlation = numpy.dot(self._ancestor.T, self._ancestor)
            for k in range(self._ancestor.shape[1]):
                index_except_k = [x for x in range(self._ancestor.shape[1]) if x != k]
                operator_weighted_sum = numpy.dot(op.T, coef[k::num_ancestor, j]).reshape(self._extract_operators[0].shape)
                gradient[:, k] = -eta * numpy.dot(operator_weighted_sum.T, diff[:, j])
                if decorrelation:
                    gradient[:, k] += decorrelation_coefficient *\
                        numpy.sum(numpy.sign(correlation[k, index_except_k]) * self._ancestor[:, index_except_k], axis=1)
            # self._ancestor /= numpy.linalg.norm(self._ancestor, 2, axis=0)

            if orthogonal_constraint:
                self._ancestor = self._orthogonal_quasi_geodesic(gradient, self._ancestor, eta)
            else:
                self._ancestor -= gradient
            eta = stepsize * (1 - j / len(idx))
        self._ancestor /= numpy.linalg.norm(self._ancestor, 2, axis=0)


    def fit(self, y, iteration = 20, learning_rate = 1e-2, learning_decay = 0.9999,
            orthogonal_constraint=False, normalize_dict=False,
            decorrelation=False, decorrelation_coefficient=1e-6,
            verbose=False):
        op = numpy.array([x.flatten() for x in self._extract_operators])
        assert (not orthogonal_constraint) or (not decorrelation),\
            print('Either orthogonal constraint or decorrelation can be used same time')
        for i in range(iteration):
            start = time.time()
            tmp_learning_rate = numpy.copy(learning_rate)
            print(i)
            # calculate D from model
            D = self._gen_D()
            if normalize_dict:
                norms = numpy.linalg.norm(D, 2, axis=0)
                D = D / norms
            self._ancestor_history.append(numpy.copy(self._ancestor))

            # fit to training data and calculate error
            self._sc.fit(D, y)
            coef = self._sc.coef_.T
            diff = y - numpy.dot(D, coef)

            if normalize_dict:
                D = D * norms
                coef = coef / norms[:, None]
            # print(diff)
            if verbose:
                print(coef)
                print(numpy.sort(numpy.count_nonzero(coef, axis=0)))
                print(numpy.count_nonzero(coef, axis=1))
            self._error_history.append(numpy.sqrt(numpy.sum(diff**2) / y.shape[1]))
            self._last_coef = coef
            print(self._error_history[-1])
            self._update_ancestors(y, coef, op, diff, tmp_learning_rate, learning_decay,
                    orthogonal_constraint, decorrelation, decorrelation_coefficient)
            D = self._gen_D()
            diff = y - numpy.dot(D, coef)
            self._ancestor /= numpy.linalg.norm(self._ancestor, 2, axis=0)
            # print(numpy.sum(numpy.abs(diff)))
            print('---------')
            print('time', time.time() - start)
            print('---------')
        self._ancestor_history.append(self._ancestor)


    def predict(self, y, normalize_dict=False):
        D = self._gen_D()
        if normalize_dict:
            norms = numpy.linalg.norm(D, 2, axis=0)
            D = D / norms
        self._sc.fit(D, y)
        coef = self._sc.coef_.T

        if normalize_dict:
            D = D * norms
            coef = coef / norms[:, None]
        y_est = numpy.dot(D, coef)

        return y_est, coef

    def save_D_figs(self, patchsize, dirname='./image/'):
        from utils.image_patch_utils import gen_patch_image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pylab as plt
        plt.gray()
        for i in range(len(self._ancestor_history)):
            D = self._gen_D(i)
            patch_img = gen_patch_image(D, patchsize, emphasis=True)
            plt.imshow(patch_img)
            plt.axis('off')
            plt.savefig(dirname + 'bases_' + str(i) + '.png', bbox_inches='tight')
            # plt.close()
