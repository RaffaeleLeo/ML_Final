from typing import List, Any, Union

import pandas as pd
import numpy as np


def main():
    for P in range(0, 20):
        data = pd.read_csv('TrainingAvi - Sheet1.csv', header=None)

        r = 0.1

        weight_vector = [0, 0, 0, 0]
        for T in range(0, 10):
            data = data.sample(frac=1).reset_index(drop=True)
            for index, row in data.iterrows():
                if (2*row[3] - 1) * (1*weight_vector[0] + row[0] * weight_vector[1] + row[1] * weight_vector[2] + row[2] * weight_vector[3]) <= 0:
                    weight_vector[0] = weight_vector[0] + r*(2*row[3] - 1)*(1)
                    weight_vector[1] = weight_vector[1] + r*(2*row[3] - 1)*(row[0])
                    weight_vector[2] = weight_vector[2] + r * (2 * row[3] - 1) * (row[1])
                    weight_vector[3] = weight_vector[3] + r * (2 * row[3] - 1) * (row[2])

        total = 0
        mistakes = 0
        print(weight_vector)
        test_data = pd.read_csv('TrainingAvi - Sheet1.csv', header=None)
        for index, row in test_data.iterrows():
            prediction = np.sign(1*weight_vector[0] + row[0] * weight_vector[1] + row[1] * weight_vector[2] + row[2] * weight_vector[3])
            total += 1
            if prediction != 2*row[3] - 1:
                mistakes += 1
        print(mistakes/total)


if __name__ == '__main__':
    main()