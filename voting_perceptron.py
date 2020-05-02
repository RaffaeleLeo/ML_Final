from typing import List, Any, Union

import pandas as pd
import numpy as np
import copy


def main():
    data = pd.read_csv('TrainingAvi - Sheet1.csv', header=None)

    r = 0.1
    all_vectors = []
    C = 1
    m = 0
    weight_vector = [0, 0, 0, 0]
    for T in range(0, 10):
        data = data.sample(frac=1).reset_index(drop=True)
        for index, row in data.iterrows():
            if (2*row[3] - 1) * (1*weight_vector[0] + row[0] * weight_vector[1] + row[1] * weight_vector[2] + row[2] * weight_vector[3]) <= 0:
                weight_vector[0] = weight_vector[0] + r*(2*row[3] - 1)*(1)
                weight_vector[1] = weight_vector[1] + r*(2*row[3] - 1)*(row[0])
                weight_vector[2] = weight_vector[2] + r * (2 * row[3] - 1) * (row[1])
                weight_vector[3] = weight_vector[3] + r * (2 * row[3] - 1) * (row[2])
                #weight_vector[4] = weight_vector[4] + r * (2 * row[4] - 1) * (row[3])
                m += 1
                all_vectors.append([copy.deepcopy(weight_vector), C])
                C = 1
            else:
                C += 1
    # s = '['
    # for x in range(0, len(all_vectors)):
    #     s += str(all_vectors[x][1])
    #     s += ', '
    # s += ']'
    # print(s)
    total = 0
    mistakes = 0
    test_data = pd.read_csv('TestingAvi - Sheet1.csv', header=None)

    for index, row in test_data.iterrows():
        prediction = 0
        for x in range(0, len(all_vectors)):
            prediction += all_vectors[x][1] * np.sign(1*all_vectors[x][0][0] + row[0] * all_vectors[x][0][1] + row[1] * all_vectors[x][0][2] + row[2] * all_vectors[x][0][3])

        total += 1
        if np.sign(prediction) != 2 * row[3] - 1:
            mistakes += 1
    print(mistakes/total)


if __name__ == '__main__':
    main()