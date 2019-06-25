import numpy
import pdb


def gen_patch_2d(image, patchsize, stride):
    width = image.shape[0]
    height = image.shape[1]

    patches = []
    for i in range((height - patchsize[1]) // stride[1] + 1):
        for j in range((width - patchsize[0]) // stride[0] + 1):
            patch = image[
                j * stride[0]:j * stride[0] + patchsize[0],
                i * stride[1]:i * stride[1] + patchsize[1]]
            patches.append(patch.flatten('F'))
    patches = numpy.column_stack(patches)

    return patches

def gen_patch_image(Y, patchsize, row_max=None, emphasis=False):
    patch_number = Y.shape[1]
    if row_max is None:
        row_max = int(numpy.sqrt(patch_number))
    col_max = patch_number // row_max

    if (patch_number % row_max) > 0:
        col_max += 1

    # create blank image matrix including separating line
    image = numpy.ones(shape=(
        row_max * patchsize[0] + row_max - 1,
        col_max * patchsize[1] + col_max - 1))
    image *= numpy.min(Y) - 0.03

    for k in range(Y.shape[1]):
        i = k % row_max
        j = k // row_max
        patch = numpy.copy(Y[:, k].reshape(patchsize, order='F'))
        if emphasis:
            patch -= (Y[:, k].min()-0.1)
            patch /= (patch.max() + 0.1)
        image[
            i * patchsize[0] + i:(i+1)*patchsize[0] + i,
            j * patchsize[1] + j:(j+1)*patchsize[1] + j] = patch

    return image


def restore_2d(Y, imagesize, patchsize, stride):
    image_array = numpy.zeros(shape=imagesize)
    count_array = numpy.zeros(shape=imagesize)

    horizontal_patch_number = (imagesize[0] - patchsize[0]) // stride[0] + 1
    vertical_patch_number = (imagesize[1] - patchsize[1]) // stride[1] + 1

    for k in range(Y.shape[1]):
        i = k % vertical_patch_number
        j = k // vertical_patch_number

        image_array[
            i * stride[0]:i * stride[0] + patchsize[0],
            j * stride[0]:j * stride[1] + patchsize[1]] += Y[:, k].reshape(patchsize, order='F')
        count_array[
            i * stride[0]:i * stride[0] + patchsize[0],
            j * stride[0]:j * stride[1] + patchsize[1]] += 1
    return image_array / count_array


if __name__ == '__main__':
    import scipy.misc
    import matplotlib.pyplot as plt
    image = scipy.misc.imread('./images/lena.png')
    image = numpy.array(image) / 255.
    plt.gray()

    Y = gen_patch_2d(image, [8, 8], [4, 4])
    idx = numpy.random.choice(Y.shape[1], 128)
    patch_img = gen_patch_image(Y[:, idx], [8, 8], emphasis=True)
    plt.imshow(patch_img)
    plt.show()
    # reconstruct_image = restore_2d(Y, [512, 512], [8, 8], [4, 4])
