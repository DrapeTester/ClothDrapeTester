import numpy as np
if __name__ == "__main__":
    with open("../dataset/dataset.csv") as f:
        num_of_data = len(f.readlines()[1:])
        print(num_of_data)
        ratio = 0.8
        num_of_train = int(num_of_data * ratio)

        perm = np.random.permutation(num_of_data)
        train_indices = perm[:num_of_train]
        test_indices = perm[num_of_train:]


        with open("train.txt", 'w') as f:
            for i in train_indices:
                f.write(str(i)+"\n")
            print(f"output to train.txt")
        with open("test.txt", 'w') as f:
            for i in test_indices:
                f.write(str(i)+"\n")
            print(f"output to test.txt")