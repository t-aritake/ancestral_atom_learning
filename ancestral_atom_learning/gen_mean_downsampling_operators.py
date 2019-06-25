import numpy
import pdb

def _gen_sum_vector(datasize, filtersize):
    """Generate base down sampling vector (down sampling operator for one element)
    """
    ### for mean down sampling
    # generate 1D zero vector to calculate first element of down sampled ancestor
    base_dsvector = numpy.zeros(shape=numpy.prod(datasize))

    # mean downsampling for single point
    # generate pair of indices for down sampling using numpy.meshgrid
    grid_idx = numpy.meshgrid(*[range(0, filtersize[i]) for i in range(len(filtersize))])
    ds_idx = grid_idx[0].flatten('C')

    # convert pair of indices for each axis to index of 1D array
    # e.g. The indices of 2D downsampling of 4 pixels are (0,0),(1,0),(0,1),(1,1).
    # if ancestor is 32x32px image, these indices are converted to 0, 1, 32, 33
    for i in range(1, len(datasize)):
        ds_idx += datasize[i-1] * grid_idx[i].flatten('C')
    # base_dsvector[ds_idx] = 1 / len(ds_idx)
    base_dsvector[ds_idx] = 1

    return base_dsvector

def _gen_sum_matrix(datasize, filtersize):
    """Generate base down sampling matrix
    (matrix to generate downsampled ancestor without shift)
    by rotating and stacking array
    """
    assert len(datasize) == len(filtersize), "The dimension of datasize and filtersize do not match"
    assert all([datasize[i] % filtersize[i] == 0 for i in range(len(datasize))]),\
            "filtersize must be divisor of corresponding element of datasize"

    base_vector = _gen_sum_vector(datasize, filtersize)

    ### list for rows of matrix
    rows = []
    target_size = [datasize[i] // filtersize[i] for i in range(len(datasize))]
    # generate pair of amount of rotation for each axis (0 to patchsize-1 for each axis)
    # maximum amount of rotation is min(self._patchsize[i], self._ancestor_size[i] // 2**level)
    # if the (downsampled) ancestor is shorter than the patchsize, it is required to restrict the rotation
    patch_rotation_axis = numpy.meshgrid(
        *[range(target_size[i]) for i in range(len(target_size))])

    # convert amount of rotation for each axis to rotation of 1D array
    amount_of_rotate = patch_rotation_axis[0].flatten('C') * filtersize[0]

    # if the ancestor is multidimensional
    for i in range(1, len(datasize)):
        amount_of_rotate +=\
            datasize[i-1] * filtersize[1] * patch_rotation_axis[i].flatten('C')
    # append rotated array to list
    for r in amount_of_rotate:
        rows.append(numpy.roll(base_vector, r))
    # generate downsampling matrix without shift by stacking row vectors
    base_extract_operator = numpy.array(rows)

    return base_extract_operator


def _least_common_multiple(a, b):
    return a * b // _greatest_common_divisor(a, b)


def _greatest_common_divisor(a, b):
    while b > 0:
        a, b = b, a%b
    return a

def _gen_base_ds_matrix(datasize, downsampled_size):
    lcm_list = [_least_common_multiple(datasize[i], downsampled_size[i]) for i in range(datasize.shape[0])]
    filtersize1 = [lcm_list[i] // datasize[i] for i in range(len(lcm_list))]
    filtersize2 = [lcm_list[i] // downsampled_size[i] for i in range(len(lcm_list))]
    expansion_matrix = _gen_sum_matrix(lcm_list, filtersize1).T

    mean_matrix = _gen_sum_matrix(lcm_list, filtersize2)
    mean_matrix /= numpy.sum(mean_matrix, axis=1)[:, None]

    return numpy.dot(mean_matrix, expansion_matrix)


def _gen_shifted_operators1(datasize, downsampled_size, patchsize, shift, base_ds_matrix):
    # calculate the count of the shifts from the ancestor_size, patchsize and shift
    shift_counts = numpy.abs(downsampled_size - patchsize) // shift + 1
    # generate list of count of shifts for each axis using meshgrid
    shift_idx = numpy.meshgrid(*[range(x) for x in shift_counts])
    amount_of_rotate_shift = shift_idx[0].flatten('C') * shift[0]
    for i in range(1, datasize.shape[0]):
        amount_of_rotate_shift +=\
            downsampled_size[i-1] * shift[i] * shift_idx[i].flatten('C')

    operators_list = []
    # generate indices for rows to be selected
    rows_grid = numpy.meshgrid(*[range(x) for x in patchsize])
    rows = rows_grid[0].flatten('C')
    for i in range(1, datasize.shape[0]):
        rows += downsampled_size[i-1] * rows_grid[i].flatten('C')

    for rotate in amount_of_rotate_shift:
        operator = base_ds_matrix[rows+rotate]
        operators_list.append(operator)

    return operators_list


def _gen_shifted_operators2(datasize, downsampled_size, patchsize, shift, base_ds_matrix):
    # calculate the count of the shifts from the ancestor_size, patchsize and shift
    shift_counts = numpy.abs(downsampled_size - patchsize) // shift + 1
    # generate list of count of shifts for each axis using meshgrid
    shift_idx = numpy.meshgrid(*[range(x) for x in shift_counts])
    amount_of_rotate_shift = shift_idx[0].flatten('C') * shift[0]
    for i in range(1, datasize.shape[0]):
        amount_of_rotate_shift +=\
            patchsize[i-1] * shift[i] * shift_idx[i].flatten('C')

    operators_list = []
    # generate indices for rows to be selected
    rows_grid = numpy.meshgrid(*[range(x) for x in downsampled_size])
    rows = rows_grid[0].flatten('C')
    for i in range(1, datasize.shape[0]):
        rows += patchsize[i-1] * rows_grid[i].flatten('C')

    for rotate in amount_of_rotate_shift:
        operator = numpy.zeros(shape=(numpy.prod(patchsize), numpy.prod(datasize)))
        operator[rows+rotate] = base_ds_matrix
        operators_list.append(operator)

    return operators_list


def _gen_shifted_operators(datasize, downsampled_size, patchsize, shift, base_ds_matrix):
    if all(downsampled_size > patchsize):
        return _gen_shifted_operators1(datasize, downsampled_size, patchsize, shift, base_ds_matrix)
    if all(downsampled_size <= patchsize):
        return _gen_shifted_operators2(datasize, downsampled_size, patchsize, shift, base_ds_matrix)

    raise Exception("invalid downsampled_size and patchsize combination")

def gen_extract_operators(datasize, downsampled_sizes, patchsize, shift):
    if type(downsampled_sizes) != list:
        downsampled_sizes = [downsampled_sizes, ]
    if type(shift) != list:
        shift = [shift, ]
    if len(downsampled_sizes) != len(shift):
        shift = shift * len(downsampled_sizes)

    assert datasize.shape == downsampled_sizes[0].shape == patchsize.shape == shift[0].shape,\
            "dimension of parameter does not match"
    operators_list = []

    for i in range(len(downsampled_sizes)):
        downsampled_size = downsampled_sizes[i]
        assert all(datasize >= downsampled_sizes[i]), "Invalid downsampled_size"
        base_ds_matrix = _gen_base_ds_matrix(datasize, downsampled_sizes[i])
        operators_list += _gen_shifted_operators(datasize, downsampled_size, patchsize, shift[i], base_ds_matrix)

    # for downsampled_size in downsampled_sizes:
    #     assert all(datasize >= downsampled_size), "Invalid downsampled_size"
    #     base_ds_matrix = _gen_base_ds_matrix(datasize, downsampled_size)
    #     # operators_list += _gen_shifted_operators()
    #     operators_list += _gen_shifted_operators(datasize, downsampled_size, patchsize, shift, base_ds_matrix)

    return operators_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    datasize = numpy.array([4, 4])
    downsampled_sizes = numpy.array([[i, i] for i in range(4, 2, -1)])
    patchsize = numpy.array([4, 4])
    shift = numpy.array([1, 1])
    # datasize = numpy.array([64,])
    # downsampled_sizes = numpy.array([64, 60, 59])[:, None]
    # patchsize= numpy.array([64,])
    # shift = numpy.array([2,])

    generated_operators = gen_extract_operators(datasize, downsampled_sizes, patchsize, shift)
    for operator in generated_operators:
        plt.imshow(operator)
        plt.show()
        plt.close()

    # print(generated_operators)
