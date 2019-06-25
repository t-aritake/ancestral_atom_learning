import numpy
import glob
from ancestral_atom_learning.ancestral_atom_learning import AncestralAtomLearning
import utils.image_patch_utils
import imageio

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as plt


def make_image(data, outputname, size=(1, 1), dpi=100):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('gray')
    ax.imshow(data, aspect='equal', interpolation='nearest')
    plt.savefig(outputname, dpi=dpi)
    plt.close()


for filename in glob.glob('ancestor*.npy'):
    print(filename)
    result = numpy.load(filename)
    result = result[None][0]

    aal = result['aal']
    num_ancestor = aal._ancestor.shape[1]
    coef = aal._last_coef

    ancestor = aal._ancestor
    ancestor_init = aal._ancestor_history[0]

    num_ancestor = ancestor.shape[1]

    ancestors_tile = utils.image_patch_utils.gen_patch_image(ancestor, result['downsampled_size'][0], row_max=3)
    ancestors_init_tile = utils.image_patch_utils.gen_patch_image(ancestor_init, result['downsampled_size'][0], row_max=3)

    scale = 4
    # expand image by nearest neighbor
    ancestors_tile = ancestors_tile.repeat(scale, axis=0).repeat(scale, axis=1)
    ancestors_init_tile = ancestors_init_tile.repeat(scale, axis=0).repeat(scale, axis=1)
    
    imageio.imwrite(str(num_ancestor) + '_tile.png', ancestors_tile)
    imageio.imwrite(str(num_ancestor) + '_init_tile.png', ancestors_init_tile)
