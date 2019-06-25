import numpy
import pdb


class ExtractOperatorsGenerator(object):
    def __init__(self, ancestor_dim, ancestor_size, patchsize, shift):
        # 一応簡易型チェックみたいなのはやった方がよい．
        # extract parameters
        self._ancestor_dim = ancestor_dim
        self._ancestor_size = ancestor_size
        self._patchsize = patchsize
        self._shift = shift

    def _gen_mean_base_ds_vector(self, level):
        """Generate base down sampling vector (down sampling operator for one element)
        """
        ### for mean down sampling
        # generate 1D zero vector to calculate first element of down sampled ancestor
        base_dsvector = numpy.zeros(shape=numpy.prod(self._ancestor_size))

        # mean downsampling for single point
        # generate pair of indices for down sampling using numpy.meshgrid
        grid_idx = numpy.meshgrid(*[range(0, 2**level)] * self._ancestor_dim)
        ds_idx = grid_idx[0].flatten('C')

        # convert pair of indices for each axis to index of 1D array
        # e.g. The indices of 2D downsampling of 4 pixels are (0,0),(1,0),(0,1),(1,1).
        # if ancestor is 32x32px image, these indices are converted to 0, 1, 32, 33
        for i in range(1, self._ancestor_dim):
            ds_idx += self._ancestor_size[i-1] * grid_idx[i].flatten('C')
        base_dsvector[ds_idx] = 1 / len(ds_idx)

        return base_dsvector

    def _gen_base_ds_matrix(self, level, base_ds_vector):
        """Generate base down sampling matrix
        (matrix to generate downsampled ancestor without shift)
        by rotating and stacking array
        """
        ### list for rows of matrix
        rows = []
        # generate pair of amount of rotation for each axis (0 to patchsize-1 for each axis)
        # maximum amount of rotation is min(self._patchsize[i], self._ancestor_size[i] // 2**level)
        # if the (downsampled) ancestor is shorter than the patchsize, it is required to restrict the rotation
        patch_rotation_axis = numpy.meshgrid(
            *[range(min(self._patchsize[i], self._ancestor_size[i] // 2**level)) for i in range(self._ancestor_dim)])

        # convert amount of rotation for each axis to rotation of 1D array
        amount_of_rotate = patch_rotation_axis[0].flatten('C') * 2**level

        # if the ancestor is multidimensional
        for i in range(1, self._ancestor_dim):
            amount_of_rotate +=\
                self._ancestor_size[i-1] * 2**level * patch_rotation_axis[i].flatten('C')
        # append rotated array to list
        for r in amount_of_rotate:
            rows.append(numpy.roll(base_ds_vector, r))
        # generate downsampling matrix without shift by stacking row vectors
        base_extract_operator = numpy.array(rows)

        return base_extract_operator

    def _gen_shifted_operators(self, level, base_extract_operator):
        """Generate all downsampling+shift operators by rotating base_extract_operator
        """
        # list for operators
        operators_list = []
        # calculate number of shifts from ancestor_size, patchsize and shift
        max_shifts = numpy.abs(self._ancestor_size // 2**level - self._patchsize) // self._shift + 1
        # generate list of count of shifts for each axis using meshgrid
        shift_idx = numpy.meshgrid(*[range(x) for x in max_shifts])
        # convert list of count of shifts into amount of rotation of matrix
        amount_of_rotate_shift = shift_idx[0].flatten('C') * self._shift[0]
        for i in range(1, self._ancestor_dim):
            amount_of_rotate_shift +=\
                self._ancestor_size[i-1] * self._shift[i] * shift_idx[i].flatten('C')

        axis_direction = 0
        # if the base_extract_operator's rows are less then columns
        # modify base_extract_operator and rotation axis direction

        # 1Dの場合はこれで良いのだろうけど2Dはダメ(16行をちゃんと4x4の位置に入れないといけないので）
        # if base_extract_operator.shape[0] < base_extract_operator.shape[1]:
        #     axis_direction = 1
        #     additional_rows = base_extract_operator.shape[1] - base_extract_operator.shape[0]
        #     padding_matrix = numpy.zeros(shape=(additional_rows, base_extract_operator.shape[1]))
        #     base_extract_operator = numpy.concatenate((base_extract_operator, padding_matrix), axis=0)

        if base_extract_operator.shape[0] < numpy.prod(self._patchsize):
            padded_operator = numpy.zeros(shape=(numpy.prod(self._ancestor_size), numpy.prod(self._ancestor_size)))
            # mean downsampling for single point
            # generate pair of indices for down sampling using numpy.meshgrid
            grid_idx = numpy.meshgrid(*[range(0, self._ancestor_size[j] // 2**level) for j in range(self._ancestor_dim)])
            ds_idx = grid_idx[0].flatten('C')

            # convert pair of indices for each axis to index of 1D array
            # e.g. The indices of 2D downsampling of 4 pixels are (0,0),(1,0),(0,1),(1,1).
            # if ancestor is 32x32px image, these indices are converted to 0, 1, 32, 33
            print(ds_idx)
            for i in range(1, self._ancestor_dim):
                ds_idx += self._ancestor_size[i-1] * grid_idx[i].flatten('C')
            padded_operator[ds_idx] = base_extract_operator
            base_extract_operator = padded_operator
        else:
            axis_direction = 1

        # append rotated matrix to list
        for r in  amount_of_rotate_shift:
            # rotate matrix along column
            operators_list.append(numpy.roll(base_extract_operator, r, axis=axis_direction))

        return operators_list
    def gen_extract_operators(self, max_level):
        operators_list = []
        import matplotlib.pyplot as plt
        for level in range(0, max_level):
            base_ds_vector = self._gen_mean_base_ds_vector(level)
            base_extract_operator = self._gen_base_ds_matrix(level, base_ds_vector)
            plt.imshow(base_extract_operator)
            plt.show()
            plt.close()
            print(base_extract_operator.shape)
            operators_list += self._gen_shifted_operators(level, base_extract_operator)
        return operators_list

# 逆行列計算可能か確認用コード（でもSGD使うなら要らないかも）
# G_inv = numpy.zeros(shape=(numpy.prod(ancestor_size), numpy.prod(ancestor_size)))
# for operator in operators_list:
#     G_inv += numpy.dot(operator.T, operator)
# G = numpy.linalg.inv(G_inv)
# 
# ダウンサウンプリングは2の累乗個の要素でしかやらないと仮定している．
# トーラス構造を導入するなら，なんかうまいことmod関数を使ってやる．

if __name__ == '__main__':
    # set random seed for debugging
    numpy.random.seed(0)

    # generate ancestors
    # ancestor = numpy.sin(numpy.linspace(0, 2*numpy.pi, 32))
    ancestor = numpy.random.normal(size=(256,))

    ancestor_size = numpy.array(ancestor.shape)
    patchsize = numpy.array([64, ])
    shift = numpy.array([4, ])
    max_level = 3

    generator = ExtractOperatorsGenerator(ancestor.ndim, ancestor_size, patchsize, shift)
    operators_list = generator.gen_extract_operators(max_level)
    print(operators_list[8])
