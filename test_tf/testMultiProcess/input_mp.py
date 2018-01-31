import cv2
import random
import numpy as np
import time
import multiprocessing as mp
import globals as g_

W = H = 256

if g_.MODEL.lower() == 'alexnet':
    OUT_W = OUT_H = 227
elif g_.MODEL.lower() == 'vgg16':
    OUT_W = OUT_H = 224


class Image:
    def __init__(self, path, label):
        with open(path) as f:
            self.label = label

        self.data = self._load(path)
        self.done_mean = False
        self.normalized = False

    def _load(self, path):
        im = cv2.imread(path)
        im = cv2.resize(im, (H, W))
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
        assert im.shape == (H, W, 3), 'BGR!'
        im = im.astype('float32')

        return im

    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.data[:, :, i] -= mean_bgr[i]

            self.done_mean = True

    def normalize(self):
        if not self.normalized:
            self.data /= 256.
            self.normalized = True

    def crop_center(self, size=(OUT_H, OUT_W)):
        h, w = self.data.shape[0], self.data.shape[1]
        hn, wn = size
        top = h / 2 - hn / 2
        left = w / 2 - wn / 2
        right = left + wn
        bottom = top + hn
        self.data = self.data[top:bottom, left:right, :]


class Dataset:
    def __init__(self, imagelist_file, subtract_mean, image_size=(OUT_H, OUT_W), name='dataset'):
        self.image_paths, self.labels = self._read_imagelist(imagelist_file)
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.name = name
        self.image_size = image_size

        print('image dataset "' + name + '" inited')
        print('  total size:', len(self.image_paths))

    def __getitem__(self, key):
        return self.image_paths[key], self.labels[key]

    def _read_imagelist(self, listfile):
        path_and_labels = np.loadtxt(listfile, dtype=str).tolist()
        paths, labels = zip(*[(l[0], int(l[1])) for l in path_and_labels])
        return paths, labels

    def load_image(self, path, label):
        i = Image(path, label)
        i.crop_center()
        if self.subtract_mean:
            i.subtract_mean()

        # i.normalize()

        return i.data

    def shuffle(self):
        z = zip(self.image_paths, self.labels)
        random.shuffle(z)
        self.image_paths, self.labels = map(list, zip(*z))
        self.shuffled = True

    def batches(self, batch_size):
        for x, y in self._batches_fast(self.image_paths, self.labels, batch_size):
            yield x, y

    def sample_batches(self, batch_size, k):
        z = zip(self.image_paths, self.labels)
        paths, labels = map(list, zip(*random.sample(z, k)))
        for x, y in self._batches_fast(paths, labels, batch_size):
            yield x, y

    def _batches_fast(self, paths, labels, batch_size):
        QUEUE_END = '__QUEUE_END105834569xx'  # just a random string
        n = len(paths)

        def load(inds, q):
            for ind in inds:
                q.put(self.load_image(paths[ind], labels[ind]))

            # indicate that I'm done
            q.put(QUEUE_END)
            q.close()

        q = mp.Queue(maxsize=1024)

        # background loading Shapes process
        p = mp.Process(target=load, args=(range(len(paths)), q))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()

        h, w = self.image_size
        x = np.zeros((batch_size, h, w, 3))
        y = np.zeros(batch_size)

        for i in xrange(0, n, batch_size):
            starttime = time.time()

            # print 'q size', q.qsize()

            for j in xrange(batch_size):
                im = q.get()

                # queue is done
                if im == QUEUE_END:
                    x = np.delete(x, range(j, batch_size), axis=0)
                    y = np.delete(y, range(j, batch_size), axis=0)
                    break

                x[j, ...] = im
                y[j] = labels[i + j]

                # print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y

    def size(self):
        """ size of paths (if splitted, only count 'train', not 'val')"""
        return len(self.image_paths)

