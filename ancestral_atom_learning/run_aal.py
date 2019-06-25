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

# set parameters for ExtractOperatorsGenerator
ancestor_size = numpy.array((8, 8))
patchsize = numpy.array((8, 8))
ancestor_shift = numpy.array((1, 1))
data_shift = numpy.array((4, 4))
max_level = 3

# learning_parameters
fit_args = {
    'learning_rate': 1e-4,
    'iteration': 100,
    'normalize_dict': False,
    'verbose': True,
    }

# crate instance of generator of the extract operators
# generator = ExtractOperatorsGenerator(2, ancestor_size, patchsize, shift)
# generate the extract operators
# downsampled_size = [numpy.array([x, x]) for x in range(8, 1, -1)]
downsampled_size = [numpy.array([ancestor_size[0]//(2**x), ancestor_size[1]//(2**x)]) for x in range(max_level)]
extract_operators = gen_extract_operators(ancestor_size, downsampled_size, patchsize, ancestor_shift)

image = scipy.misc.imread('./lena.png')
image = numpy.array(image) / 255.

y = gen_patch_2d(image, patchsize, data_shift)
# y = y[:, numpy.random.choice(y.shape[1], 3000)]
y_mean = numpy.mean(y, axis=0)
y = y - numpy.tile(y_mean, [y.shape[0], 1])

# declare lasso model
lasso = linear_model.Lasso(alpha=1e-3)
# omp =  linear_model.OrthogonalMatchingPursuit(tol=0.1, normalize=False)
omp =  linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=15, normalize=False)

# aal = AncestralAtomLearning(ancestor, extract_operators, omp)

# remember datetime for filename
dtstr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

init_ancestors = []
for i in range(10):
    theta = numpy.linspace(0, 2*numpy.pi, ancestor_size[0])
    sin_wave = numpy.sin((i+1)*theta)
    ancestor_init = numpy.outer(sin_wave, sin_wave)
    init_ancestors.append(ancestor_init.flatten('F'))
init_ancestors = numpy.array(init_ancestors).T

for num_ancestor in range(1, 10):
    print('---------------------------')
    print('num_ancestor = ', num_ancestor)
    print('---------------------------')
    # set random seed
    # numpy.random.seed(0)

    # initialize ancestor as random vector
    # ancestor = numpy.random.normal(size=(numpy.prod(ancestor_size), num_ancestor))
    ancestor = init_ancestors[:, :num_ancestor]
    ancestor = ancestor - numpy.mean(ancestor)
    ancestor = ancestor / numpy.linalg.norm(ancestor, 2, axis=0)
    # ancestor = numpy.random.normal(size=(64, 64))
    # ancestor, _ = numpy.linalg.qr(ancestor)
    # ancestor = ancestor[:, :num_ancestor]

    aal = AncestralAtomLearning(ancestor, extract_operators, lasso)
    # aal = AncestralAtomLearning(ancestor, extract_operators, lasso)
    aal.fit(y, **fit_args)
    y_est, _ = aal.predict(y, fit_args['normalize_dict'])
    y_est += y_mean
    restored_img = restore_2d(y_est, image.shape, patchsize, data_shift)

    dirname = '/home/data2/aritaket/aal/all_omp_normalize_sin_init/'
    imdirname = dirname + 'image' + dtstr + '_' + str(num_ancestor) + '_lasso_not_normalize' + '/'
    if not os.path.isdir(imdirname):
        os.makedirs(imdirname)
    writer = open(
            dirname + 'exam_' + dtstr + '_' + str(num_ancestor) + '.pkl', 'wb')
    pickle.dump({
        'aal': aal,
        'fit_args': fit_args,
        'y' : y,
        'y_mean': y_mean,
        'restored_img': restored_img,}, writer)

    aal.save_D_figs(patchsize, dirname=imdirname)
    scipy.misc.imsave(imdirname + 'restored_img.png', numpy.uint8(restored_img*255))
