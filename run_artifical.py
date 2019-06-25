import numpy
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
data_shift = numpy.array((2, 2))
max_level = 3

# learning_parameters
fit_args = {
    'learning_rate': 1e-4,
    'iteration': 30,
    'normalize_dict': True,
    'verbose': True,
    'learning_decay':0.98
    }

# crate instance of generator of the extract operators
downsampled_size = [numpy.array([x, x]) for x in [16, 14, 12, 10, 8]]
extract_operators = gen_extract_operators(ancestor_size, downsampled_size, patchsize, ancestor_shift)

patches = numpy.load('./artificial_dense.npy')
patches_mean = numpy.mean(patches, axis=0)
patches = patches - patches_mean

# declare lasso model
lasso = linear_model.Lasso(alpha=1e-2)

for ancestor_num in range(3, 4):
    print('---------------------------')
    print('num_ancestor = ', ancestor_num)
    print('---------------------------')
    
    # initialize ancestor as random vector
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
