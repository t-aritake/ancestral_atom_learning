# -*- coding: utf-8 -*-
import numpy
import scipy.cluster.hierarchy


# 二分木みたいにルートから追加できないのがちょっと面倒．
# あとで見直す？
class Node(object):
    ''' class to construct clustering tree'''
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None
        self.annotated_distance = None


# 全ノードへのアクセス（順番は葉，親の順番で）
def traverse(node):
    if node is None:
        return
    for x in traverse(node.left):
        yield x
    for x in traverse(node.right):
        yield x
    yield node


# 葉ノードへのアクセス（annotated distanceによる枝刈りも可能）
def traverse_leaves(node, thresholding=-1):
    # 自分がNoneなら葉
    if node is None:
        return
    if node.annotated_distance is not None:
        if node.annotated_distance <= thresholding:
            yield node
            return
    for x in traverse_leaves(node.left, thresholding):
        yield x
    for x in traverse_leaves(node.right, thresholding):
        yield x
    if node.left is None and node.right is None:
        yield node


def cluster_dimension(D):
    '''
    Parameters
    ----------
    D : numpy.array
        1d array of pairwise distance matrix of all items
        The form should be same as the returned value of scipy.spatial.distance.pdist

    Returns
    ----------
    int
        estimated cluster dimension
    '''

    num_items = 1
    while True:
        if len(D) / num_items * 2 == num_items + 1:
            num_items += 1
            break
        num_items += 1
    D_mat = numpy.zeros(shape=(num_items, num_items))
    D_mat[numpy.triu_indices(num_items, 1)] = D

    # items = [[i] for i in range(num_items)]
    items = []
    for i in range(num_items):
        n = Node()
        n.data = [i]
        items.append(n)
    # items = [Node([i]) for i in range(num_items)]
    item_indices = [i for i in range(num_items)]
    link = scipy.cluster.hierarchy.linkage(D, method='complete')
    # annotated_distance = [0 for i in range(num_items)]
    annotated_distance = []

    for l in link:
        # item_indices.append(len(items))
        item = Node()
        item.left = items[int(l[0])]
        item.right = items[int(l[1])]
        item.data = item.left.data + item.right.data
        item.annotated_distance = numpy.max(D_mat[item.data][:, item.data])
        items.append(item)

    annotated_distance = sorted([item.annotated_distance for item in traverse(items[-1]) if item.annotated_distance is not None])

    rmin = 0
    rmax = numpy.max(D_mat)
    # rminは各クラスタの要素数の中央値が1より大きくなるときの値
    for r in annotated_distance:
        data = sorted([len(item.data) for item in traverse_leaves(items[-1], r)])
        if data[len(data) // 2] > 1:
            rmin = r
            break

    r_list = []
    leafs_list = []

    # TODO: 本当はここはもっと細かく点をとって傾きを求める方法でもよい
    # とりあえずrmin, rmaxの25%, 75%の値でやってる
    rmean = (rmin + rmax) / 2
    rstep = 0.1
    # for r in [(rmin + rmean)/2, (rmean + rmax) / 2]:
    for r in numpy.arange(rmin, rmax, rstep):
        # print(r_prev < link[:, 2])
        data = [len(item.data) for item in traverse_leaves(items[-1], r)]
        r_list.append(r)
        leafs_list.append(len(list(traverse_leaves(items[-1], r))))

    x = numpy.log(r_list)
    y = numpy.log(leafs_list)
    a = -(numpy.dot(x, y) - x.sum() * y.sum() / len(x)) / ((x**2).sum() - x.sum()**2/len(x))
    # print(a)
    return a
    # print(numpy.mean(numpy.array(leafs_list) / numpy.array(r_list)))
    # return -(numpy.log(leafs_list[-1]) - numpy.log(leafs_list[0])) / (numpy.log(r_list[-1])-numpy.log(r_list[0]))


def packing_dimension(D, r1=None, r2=None, epsilon=1e-3):
    num_items = 1
    while True:
        if len(D) / num_items * 2 == num_items + 1:
            num_items += 1
            break
        num_items += 1
    D_mat = numpy.zeros(shape=(num_items, num_items))
    D_mat[numpy.triu_indices(num_items, 1)] = D
    D_mat = D_mat + D_mat.T

    if r1 is None:
        r1 = numpy.percentile(D, 25)
    if r2 is None:
        r2 = numpy.percentile(D, 75)
    r_list = [r1, r2]

    items = numpy.array([i for i in range(num_items)])
    L_list = [[], []] 
    for loop in range(1000):
        numpy.random.shuffle(items)
        for k in range(2):
            pack = [items[0], ]
            for i in range(1, num_items):
                if numpy.all(D_mat[i, pack] >= r_list[k]):
                    pack.append(items[i])
            L_list[k].append(numpy.log(len(pack)))
        denom = numpy.log(r_list[1]) - numpy.log(r_list[0])
        D_pack = - (numpy.mean(L_list[1]) - numpy.mean(L_list[0])) / denom

        criterion = 1.65 * numpy.sqrt(numpy.var(L_list[0]) + numpy.var(L_list[1])) / (numpy.sqrt(loop+1) * denom)
        if loop > 10 and criterion < D_pack * (1 - epsilon) / 2:
            return D_pack


if __name__ == '__main__':
    from sklearn import datasets
    import scipy.spatial.distance
    cdim = []
    pdim = []
    # numpy.random.seed(1)
    for i in range(20):
        r=4
        B = numpy.random.normal(size=(10, r))
        B = B / numpy.sum(B, axis=0)
        C = numpy.random.random(size=(r, 10000))
        C = C / numpy.sum(C, axis=0)
        X = numpy.dot(B, C).T
        print(X.shape)

        D = scipy.spatial.distance.pdist(X)
        pdim.append(packing_dimension(D))
        cdim.append(cluster_dimension(D))
        print(cdim[-1])
        print(pdim[-1])
        print()
    print(numpy.mean(pdim), numpy.var(pdim))
    print(numpy.mean(cdim), numpy.var(cdim))

    # X, color = datasets.samples_generator.make_swiss_roll(n_samples=900)
    # print(X.shape)
    # D = scipy.spatial.distance.pdist(X)
    # print(cluster_dimension(D))
