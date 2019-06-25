import numpy
import pdb


def gen_patch_1d(data, patchsize, stride):
    patches = []
    for i in range((data.shape[0] - patchsize) // stride + 1):
        patch = data[i * stride:i * stride + patchsize]
        patches.append(patch)
    patches = numpy.column_stack(patches)

    return patches


def restore_1d(Y, sig_length, patchsize, stride):
    sig_array = numpy.zeros(shape=(sig_length,))
    count_array = numpy.zeros(shape=(sig_length,))

    for i in range(Y.shape[1]):
        sig_array[i * stride:i * stride + patchsize] += Y[:, i]
        count_array[i * stride:i * stride + patchsize] += 1

    return sig_array / count_array


if __name__ == '__main__':
    import scipy.misc
    import matplotlib.pyplot as plt
    x = numpy.linspace(0, 8 * numpy.pi, 256)
    sig = numpy.sin(x)

    Y = gen_patch_1d(sig, 32, 4)

    reconst_sig = restore_1d(Y, 256, 32, 4)
