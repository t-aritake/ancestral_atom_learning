import numpy
import scipy.misc
import pickle
import datetime
import os
from sklearn import linear_model
from ancestral_atom_learning import AncestralAtomLearning
# from gen_extract_operators import ExtractOperatorsGenerator
from gen_mean_downsampling_operators import gen_extract_operators
from utils.image_patch_utils import gen_patch_2d, restore_2d


def mexh(t, sigma=1.0):
    return 2 / (numpy.pi**(1/4) * numpy.sqrt(3*sigma)) * (1-(t/sigma)**2) * numpy.exp(-t**2/(2*sigma**2))

# set parameters for ExtractOperatorsGenerator
ancestor_size = numpy.array([64,])
patchsize = numpy.array([64,])
ancestor_shift = numpy.array([8,])

# data_shift = numpy.array((4, 4))
max_level = 3

# learning_parameters
fit_args = {
    'learning_rate': 1e-4,
    'iteration': 30,
    'normalize_dict': False,
    'verbose': True,
    }

# crate instance of generator of the extract operators
# generator = ExtractOperatorsGenerator(2, ancestor_size, patchsize, shift)
# generate the extract operators
downsampled_size = [numpy.array([64 // 2**i,]) for i in range(0, max_level)]
# downsampled_size = [numpy.array([ancestor_size[0]//(2**x), ancestor_size[1]//(2**x)]) for x in range(max_level)]
extract_operators = gen_extract_operators(ancestor_size, downsampled_size, patchsize, ancestor_shift)

ancestor_true = numpy.column_stack([mexh(numpy.linspace(-4, 4, 64)), mexh(numpy.linspace(-4, 4, 64), sigma=2.0), mexh(numpy.linspace(-4, 4, 64), sigma=0.2)])
# ancestor_true = numpy.sin(numpy.linspace(0, 2*numpy.pi, 64))
D_true = numpy.column_stack([numpy.dot(F, ancestor_true) for F in extract_operators])

# p = []
# for l in range(max_level):
#     num = (ancestor_size[0]-downsampled_size[l][0]) // ancestor_shift[0] + 1
#     p += [1/(max_level * num)] * num

l0norm = 5
data_num = 3000
C_true = numpy.zeros(shape=(D_true.shape[1], data_num))
for col in range(data_num):
    rows = numpy.random.choice(C_true.shape[0], l0norm)
    C_true[rows, col] = numpy.random.normal(size=l0norm)

y = numpy.dot(D_true, C_true)
y += numpy.random.normal(scale=0.01, size=y.shape)

# y = gen_patch_2d(image, patchsize, data_shift)
y_mean = numpy.mean(y, axis=0)
y = y - numpy.tile(y_mean, [y.shape[0], 1])

# declare lasso model
lasso = linear_model.Lasso(alpha=1e-3)
# omp =  linear_model.OrthogonalMatchingPursuit(tol=0.1, normalize=False)
omp =  linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=5, normalize=False)

# aal = AncestralAtomLearning(ancestor, extract_operators, omp)

# remember datetime for filename
dtstr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

for num_ancestor in range(1, 3):
    print('---------------------------')
    print('num_ancestor = ', num_ancestor)
    print('---------------------------')
    # set random seed
    # numpy.random.seed(0)

    # initialize ancestor as random vector
    # ancestor = numpy.random.normal(size=(numpy.prod(ancestor_size), num_ancestor))
    ancestor = numpy.random.normal(size=(64, 3))
    ancestor = ancestor - numpy.mean(ancestor)
    ancestor = ancestor / numpy.linalg.norm(ancestor, 2, axis=0)
    # ancestor = numpy.random.normal(size=(64, 64))
    # ancestor, _ = numpy.linalg.qr(ancestor)
    # ancestor = ancestor[:, :num_ancestor]

    # aal = AncestralAtomLearning(ancestor, extract_operators, omp)
    aal = AncestralAtomLearning(ancestor, extract_operators, lasso)
    aal.fit(y, **fit_args)
    # y_est, _ = aal.predict(y, fit_args['normalize_dict'])
    # y_est += y_mean
    # restored_img = restore_2d(y_est, image.shape, patchsize, data_shift)

    dirname = '/home/data/aritaket/aal/all_omp_not_normalize_artificial/'
    imdirname = dirname + 'image' + dtstr + '_' + str(num_ancestor) + '_lasso_not_normalize' + '/'
    if not os.path.isdir(imdirname):
        os.makedirs(imdirname)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(ancestor_true)
    plt.plot(aal._ancestor)
    plt.savefig('ancestor.pdf')
    plt.close()
    # writer = open(
    #         dirname + 'exam_' + dtstr + '_' + str(num_ancestor) + '.pkl', 'wb')
    # pickle.dump({
    #     'aal': aal,
    #     'fit_args': fit_args,
    #     'y' : y,
    #     'y_mean': y_mean,}, writer)
    #     # 'restored_img': restored_img,}, writer)

    # aal.save_D_figs(patchsize, dirname=imdirname)
    # scipy.misc.imsave(imdirname + 'restored_img.png', numpy.uint8(restored_img*255))
