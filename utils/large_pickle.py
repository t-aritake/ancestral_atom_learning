import pickle


# openしたfileのread, write関数を細工して，2**3
class LargeObject(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        # 項目名'item'を参照されたらf.'item'を返す
        # (itemは任意の文字列）
        # 要するにfile objectのread write以外は元のファイルオブジェクトを見ればよい
        return getattr(self.f, item)

    def write(self, obj):
        # このobjはもうbyteに変換されてくるのでnumpy.arrayに対してもlenでOK
        n = len(obj)
        print("writing total_bytes={0}...".format(n), flush=True)
        idx = 0
        while idx < n:
            # 2の32乗はmacでpickleを用いた保存が不可なので
            # 31bitで表現できる限界の長さで止める
            batch_size = min(n-idx, (1 << 31) - 1)
            print("writing bytes [{0}, {1})... ".format(idx, idx+batch_size), end="", flush=True)
            self.f.write(obj[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

    def read(self, n):
        if n >= (1 << 31):
            obj = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, (1<<31) - 1)
                print("loading bytes [{0}, {1})... ".format(idx, idx+batch_size), end="", flush=True)
                obj[idx:idx+batch_size] = self.f.read(batch_size)
                print("done.", flush=True)
                idx += batch_size
            return obj 
        return self.f.read(n)


def pickle_dump(obj, file_path):
    with open(file_path, 'wb') as f:
        return pickle.dump(obj, LargeObject(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(LargeObject(f))


if __name__ == '__main__':
    import numpy
    data = numpy.random.normal(size=(2**14, 2**15))
    pickle_dump(data, './test.pkl')
