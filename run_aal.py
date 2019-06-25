import numpy
import glob
import imageio
from sklearn import linear_model
from ancestral_atom_learning import ancestral_atom_learning
from ancestral_atom_learning.gen_mean_downsampling_operators import gen_extract_operators
from utils.image_patch_utils import gen_patch_2d, restore_2d

# set parameters for ExtractOperatorsGenerator
ancestor_size = numpy.array((16, 16))
patchsize = numpy.array((16, 16))
ancestor_shift = [numpy.array((2, 2)), numpy.array((2, 2)), numpy.array((2, 2))]
# ancestor_shift = [numpy.array((1, 1)), numpy.array((1, 1)), numpy.array((1, 1))]
overlap = patchsize - ancestor_shift
data_shift = numpy.array((8, 8))
max_level = 3

# learning_parameters
fit_args = {
    'learning_rate': 1e-3,
    'iteration': 2,
    'normalize_dict': True,
    'verbose': True,
    'learning_decay':0.95
    }

# crate instance of generator of the extract operators
# generate the extract operators
downsampled_size = [numpy.array([x, x]) for x in range(16, 7, -2)]
# downsampled_size = [numpy.array([ancestor_size[0]//(2**x), ancestor_size[1]//(2**x)]) for x in range(max_level)]
extract_operators = gen_extract_operators(ancestor_size, downsampled_size, patchsize, ancestor_shift)

patches = []
patches_mean = []
import pdb
for filename in glob.glob('./dataset/*'):
    image = imageio.imread(filename)
    image = numpy.array(image) / 255.

    tmp_patches = gen_patch_2d(image, patchsize, data_shift)
    # y = y[:, numpy.random.choice(y.shape[1], 3000)]
    tmp_patches_mean = numpy.mean(tmp_patches, axis=0)
    patches.append(tmp_patches - tmp_patches_mean)
    patches_mean.append(tmp_patches_mean)

patches = numpy.concatenate(patches, axis=1)
patches_mean = numpy.concatenate(patches_mean)

# declare instance of sparse coding model
lasso = linear_model.Lasso(alpha=1e-3)
# omp =  linear_model.OrthogonalMatchingPursuit(tol=0.1, normalize=False)
# omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=15, normalize=False)

for ancestor_num in range(1, 2):
    print('---------------------------')
    print('num_ancestor = ', ancestor_num)
    print('---------------------------')
    # set random seed
    # numpy.random.seed(0)

    # initialize ancestor as random vector
    # ancestor = numpy.random.normal(size=(numpy.prod(ancestor_size), num_ancestor))
    ancestor = numpy.random.normal(size=(patches.shape[0], ancestor_num))
    ancestor = ancestor - numpy.mean(ancestor)
    ancestor = ancestor / numpy.linalg.norm(ancestor, 2, axis=0)

    aal = ancestral_atom_learning.AncestralAtomLearning(ancestor, extract_operators, lasso)
    aal.fit(patches, **fit_args)

    numpy.save('ancestor_{0}.npy'.format(ancestor_num), {
        'ancestor_size': ancestor_size,
        'overlap': overlap,
        'aal': aal,
        'downsampled_size': downsampled_size})
