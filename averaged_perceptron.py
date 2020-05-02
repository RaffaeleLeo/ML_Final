import pandas as pd
import numpy as np


def main():
    for P in range(0, 50):
        data = pd.read_csv('TrainingAvi - Sheet1.csv', header=None)
        data = data.sample(frac=0.5).reset_index(drop=True)
        r = 0.1

        weight_vector = [0, 0, 0, 0]
        average_vector = [0, 0, 0, 0]
        for T in range(0, 10):
            data = data.sample(frac=1).reset_index(drop=True)
            for index, row in data.iterrows():
                #print(row)
                if (2*float(row[3]) - 1) * (1*weight_vector[0] + float(row[0]) * weight_vector[1] + float(row[1]) * weight_vector[2] + float(row[2]) * weight_vector[3]) <= 0:
                    weight_vector[0] = weight_vector[0] + r*(2*float(row[3]) - 1)*(1)
                    weight_vector[1] = weight_vector[1] + r*(2*float(row[3]) - 1)*float(row[0])
                    weight_vector[2] = weight_vector[2] + r * (2 * float(row[3]) - 1) * float(row[1])
                    weight_vector[3] = weight_vector[3] + r * (2 * float(row[3]) - 1) * float(row[2])
                    #weight_vector[4] = weight_vector[4] + r * (2 * row[4] - 1) * (row[3])

                average_vector[0] += weight_vector[0]
                average_vector[1] += weight_vector[1]
                average_vector[2] += weight_vector[2]
                average_vector[3] += weight_vector[3]
                #average_vector[4] += weight_vector[4]

        total = 0
        mistakes = 0
        #print(average_vector)
        test_data = pd.read_csv('TestingAvi - Sheet1.csv', header=None)
        test_data = data.sample(frac=0.5).reset_index(drop=True)
        for index, row in test_data.iterrows():
            prediction = np.sign(1*average_vector[0] + float(row[0]) * average_vector[1] + float(row[1]) * average_vector[2] + float(row[2]) * average_vector[3])
            total += 1
            if prediction != 2*float(row[3]) - 1:
                mistakes += 1
        print(mistakes/total)


if __name__ == '__main__':
    main()