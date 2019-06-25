from utils.image_patch_utils import gen_patch_image
import numpy
import matplotlib.pyplot as plt

res = numpy.load('./ancestor_1.npy')
aal = res[None][0]['aal']

patch_size = res[None][0]['ancestor_size']
ancestor_num = aal._ancestor.shape[1]
downsampled_size = res[None][0]['downsampled_size']

D = aal._gen_D()

D2 = numpy.ones(shape=(D.shape[0], 10 * 5))
D2 *= numpy.min(D)
index = list(range(ancestor_num))
D2[:, index] = D[:, index]

count = ancestor_num
for scale in range(1, len(downsampled_size)):
    tmp_index = list(range(count, count+ancestor_num))
    D2[:, 10*scale:10*scale+len(tmp_index)] = D[:, tmp_index]
    index += list(range(count, count+ancestor_num))
    num_shift = numpy.prod((patch_size - downsampled_size[scale]) // 2 + 1)
    count += num_shift * ancestor_num

im = gen_patch_image(D2, patch_size, row_max=5)
plt.gray()
plt.imshow(im)
plt.axis('off')
plt.savefig('ancestor_image1.png', bbox_inches='tight')
