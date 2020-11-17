import glob
import os
import pickle
from abc import ABC, abstractmethod
import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    import cv2
except:
    pass
import multiprocessing as mp


class OrdinalDataset(Dataset, ABC):
    def __init__(self, config, transform, task, data_splits):
        self.config = config
        self.transform = transform
        self.task = task
        self.data_splits = data_splits

    @abstractmethod
    def get_labels(self):
        pass


class AdienceDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.root_dir = config.root_dir
        self.images_dir = config.data_images
        self.fold = config.fold
        self.XY = self.read_from_txt_file()

    def get_labels(self):
        return np.array([t[1] for t in self.XY])

    def __len__(self):
        return len(self.XY)

    def read_from_txt_file(self):
        txt_file = f'{self.root_dir}/train_val_txt_files_per_fold/test_fold_is_{self.fold}/age_{self.task}.txt'
        data = []
        f = open(txt_file)
        for line in f.readlines():
            image_file, label = line.split()
            label = int(label)
            data.append((image_file, label))
        return data

    def __getitem__(self, idx):
        img_name, label = self.XY[idx]
        image = Image.open(self.images_dir + '/' + img_name)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }


class HCIDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.X, self.Y = data_splits[task]
        if task == 'train':
            X, Y = data_splits['val']
            self.X += X
            self.Y += Y

    @staticmethod
    def split_dataset(root_dir, seed=0):
        classes = {'1930s': 0, '1940s': 1, '1950s': 2, '1960s': 3, '1970s': 4}
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        X_val = []
        Y_val = []
        for c in classes.keys():
            x = glob.glob('{}/{}/*.jpg'.format(root_dir, c))
            y = [classes[c] for _ in range(len(x))]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=seed)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=22, random_state=seed)
            X_train += x_train
            Y_train += y_train
            X_test += x_test
            Y_test += y_test
            X_val += x_val
            Y_val += y_val
        print('size(X_train)={}'.format(len(X_train)))
        print('size(X_test)={}'.format(len(X_test)))
        print('size(X_val)={}'.format(len(X_val)))

        return {
            'train': (X_train, Y_train),
            'test': (X_test, Y_test),
            'val': (X_val, Y_val),
        }

    def get_labels(self):
        return np.array(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        image = image.convert('RGB')
        label = self.Y[idx]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }


class DRDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        if self.config.use_h5f_dataset:
            h5f = h5py.File(self.config.h5f_dataset_path.format(task, config.split), 'r')
            self.X = h5f['x']
            self.Y = h5f['y']
        else:
            self.X, self.Y = self.data_splits[task]

    def __getitem__(self, idx):
        if self.config.use_h5f_dataset:
            image, label = self.X[idx], self.Y[idx]
            image = Image.fromarray(np.uint8(image).reshape((256, 256, 3)), mode='RGB')
        else:
            image = Image.open(self.X[idx]).resize((256, 256)).convert('RGB')
            label = self.Y[idx]
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'label': label
        }

    def __len__(self):
        return len(self.X)

    def get_labels(self):
        return np.array(self.Y)

    @staticmethod
    def convert_image(config, image_path):
        """
        Kaggle winner data preprocessing: https://kaggle-forum-message-attachments.storage.googleapis.com/88655/2795/competitionreport.pdf
        The scale is the same as in https://github.com/christopher-beckham/msc/blob/master/experiments/diabetic_retinopathy/ben_opencv_resize.py
        """
        try:
            def scale_radius(img, scale):
                x = img[int(img.shape[0] / 2), :, :].sum(1)
                r = (x > x.mean() / 10).sum() / 2
                s = scale * 1.0 / r
                return cv2.resize(img, (0, 0), fx=s, fy=s)

            a = cv2.imread(image_path)
            # scale img to a given radius
            a = scale_radius(a, config.radius_scale)
            # subtract local mean color
            a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), config.radius_scale / 30), -4, 128)
            # remove out er 10%
            b = np.zeros(a.shape)
            cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(config.radius_scale * 0.9), (1, 1, 1), -1, 8, 0)
            a = a * b + 128 * (1 - b)
            cv2.imwrite(os.path.join(config.preprocessed_images_dir, os.path.basename(image_path)), a)
        except:
            print('Failed to convert', image_path)

    @staticmethod
    def preprocess_images(config):
        def chunks(l, n):
            for i in range(0, len(l), n): yield l[i:i + n]

        assert not os.path.exists(config.preprocessed_images_dir), 'preprocessed_images_dir already exists! Are you sure?'
        os.makedirs(config.preprocessed_images_dir)
        batch = 16
        count = 0
        for chunk in chunks(glob.glob(os.path.join(config.raw_images_dir, '*.jpeg')), batch):
            count += 1
            jobs = []
            for image_path in chunk: jobs.append(mp.Process(target=DRDataset.convert_image, args=(config, image_path)))
            for j in jobs: j.start()
            for j in jobs: j.join()
            if (count + 1) % 100 == 0: print('Processed {} chunks of {} images...'.format(count, batch))

    @staticmethod
    def create_h5f_dataset(data: dict, hdf_path, split):
        for task in data.keys():
            X, Y = data[task]
            task_hdf_path = hdf_path.format(task, split)
            h5f = h5py.File(task_hdf_path, 'w')
            h5f.create_dataset('x', shape=(len(X), 3, 256, 256), dtype="float32")
            h5f.create_dataset('y', shape=(len(X),), dtype="int32")
            for i in range(0, X.shape[0]):
                img = Image.open(X[i]).resize((256, 256)).convert('RGB')
                h5f['x'][i] = np.array(img).reshape((3, 256, 256))
                h5f['y'][i] = Y[i]
            h5f.close()

    @staticmethod
    def split_dataset(root_dir, pickle_meta, seed=0):
        with open(pickle_meta, mode='rb') as f:
            X_left, X_right, y_left, y_right = pickle.load(f)

        X_left = [("%s/%s.jpeg" % (root_dir, elem)) for elem in X_left]
        X_right = [("%s/%s.jpeg" % (root_dir, elem)) for elem in X_right]

        X_left = np.asarray(X_left)
        X_right = np.asarray(X_right)
        y_left = np.asarray(y_left, dtype="int32")
        y_right = np.asarray(y_right, dtype="int32")

        existing_idxs = [os.path.exists(X_left[i]) and os.path.exists(X_right[i]) for i in range(len(X_left))]
        X_left = X_left[existing_idxs]
        X_right = X_right[existing_idxs]
        y_left = y_left[existing_idxs]
        y_right = y_right[existing_idxs]

        rnd = np.random.RandomState(seed)
        idxs = [x for x in range(len(X_left))]
        rnd.shuffle(idxs)

        test_start = int(0.8 * X_left.shape[0])
        valid_start = int(0.9 * X_left.shape[0])

        X_train_left = X_left[idxs][0: test_start]
        X_train_right = X_right[idxs][0: test_start]

        y_train_left = y_left[idxs][0: test_start]
        y_train_right = y_right[idxs][0: test_start]

        X_valid_left = X_left[idxs][valid_start:]
        X_valid_right = X_right[idxs][valid_start:]

        y_valid_left = y_left[idxs][valid_start:]
        y_valid_right = y_right[idxs][valid_start:]

        X_test_left = X_left[idxs][test_start:valid_start]
        X_test_right = X_right[idxs][test_start:valid_start]

        y_test_left = y_left[idxs][test_start:valid_start]
        y_test_right = y_right[idxs][test_start:valid_start]

        # ok, fix now
        X_train = np.hstack((X_train_left, X_train_right))
        X_valid = np.hstack((X_valid_left, X_valid_right))
        X_test = np.hstack((X_test_left, X_test_right))

        y_train = np.hstack((y_train_left, y_train_right))
        y_valid = np.hstack((y_valid_left, y_valid_right))
        y_test = np.hstack((y_test_left, y_test_right))

        print('size(X_train)={}'.format(len(X_train)))
        print('size(X_test)={}'.format(len(X_test)))
        print('size(X_val)={}'.format(len(X_valid)))

        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'val': (X_valid, y_valid),
        }


class AbaloneDataset(OrdinalDataset):
    def __init__(self, config, transform, task, data_splits):
        super().__init__(config, transform, task, data_splits)
        self.X, self.Y = data_splits[task]

    @staticmethod
    def split_dataset(csv_path, seed=0):
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        X = data[:, :10]
        Y = np.expand_dims(data[:, 10], axis=1)
        Y = Y.astype('int')
        num_classes = int(np.max(Y[:, 0]) + 1)
        Y = Y.astype('float32')
        X = X.astype('float32')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=seed)
        print('size(X_train)={}'.format(len(X_train)))
        print('size(X_test)={}'.format(len(X_test)))
        print('size(X_val)={}'.format(len(X_val)))
        print('num of classes: ', num_classes)
        return {
            'train': (X_train, Y_train),
            'test': (X_test, Y_test),
            'val': (X_val, Y_val),
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'image': self.X[idx],
            'label': self.Y[idx]
        }
