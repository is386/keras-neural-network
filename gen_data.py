import numpy as np
import os
import pickle
from PIL import Image


IMG_PATH = "./yalefaces"
SIZE = (40, 40)

# Organize all the images into folders with the folder named after the category
# Example: Categories "cat" and "dog" = ["cat", "dog"]
CATEGORIES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


# Returns a numpy array of all the Yalefaces images
def create_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(IMG_PATH, str(category))
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img = Image.open(os.path.join(path, img))
            img_arr = np.asarray(img.resize(SIZE))
            data.append([img_arr, class_num])
    np.random.shuffle(data)
    return data


# Splits the data into 2/3 training and 1/3 testing
def split_data(data):
    train = []
    test = []
    total = len(CATEGORIES)
    num_train = int(total / 3) * 2

    for i in range(total):
        k = 0
        for d in data:
            if d[1] == i:
                if k < num_train:
                    train.append(d)
                else:
                    test.append(d)
                k += 1

    np.random.shuffle(train)
    np.random.shuffle(test)
    return np.asarray(train), np.asarray(test)


# Separates the data from their labels into two arrays
def split_labels(data):
    x = []
    y = []

    for features, label in data:
        x.append(features)
        y.append(label)

    return np.asarray(x).reshape(-1, SIZE[0], SIZE[1], 1), np.asarray(y)


def main():
    data = create_data()

    train_data, test_data = split_data(data)
    x_train, y_train = split_labels(train_data)
    x_test, y_test = split_labels(test_data)

    pickle_out = open("train/x.pickle", "wb")
    pickle.dump(x_train, pickle_out)
    pickle_out.close()

    pickle_out = open("train/y.pickle", "wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open("test/x.pickle", "wb")
    pickle.dump(x_test, pickle_out)
    pickle_out.close()

    pickle_out = open("test/y.pickle", "wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
