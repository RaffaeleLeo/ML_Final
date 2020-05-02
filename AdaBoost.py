import math
import pandas as pandas
import copy
from AttNode import AttNode
import pandas as pandas
from DecisionStump import StumpBuilder

#Assume Bank Data is already formatted
class AdaBoost:
    def __init__(self, df, attributes, values, labels, length):
        self.stumps = []
        self.alphas = []
        self.stump_builder = StumpBuilder()

        for index in range(length):
            self.stumps.append(self.stump_builder.build_stump(df, attributes, values, labels, 1))

            # Finding Error
            error = 0.0
            for i, row in df.iterrows():
                if self.stump_builder.decide(self.stumps[index], row) != row[len(attributes)]:
                    error = error + row[4]

            alpha = (.5) * math.log((1-error)/error)
            self.alphas.append(alpha)

            if length != length - 1:
                z = 0.0
                for i, row in df.iterrows():
                    weight = row[4]
                    if self.stump_builder.decide(self.stumps[index], row) == row[len(attributes)]:
                        weight = weight * math.exp(-alpha)
                    else:
                        weight = weight * math.exp(alpha)
                    z = z + weight
                    df.loc[i, 4] = weight

                for i, row in df.iterrows():
                    df.loc[i, 4] / z

    def decide(self, data, tests):
        decision = 0.0
        for index in range(tests):
            if self.stump_builder.decide(self.stumps[index], data) == '1':
                decision = decision + self.alphas[index]
            else:
                decision = decision + -1 * self.alphas[index]
        if decision > 0:
            return "1"

        return "0"

    def get_stumps(self):
        return self.stumps
    def get_builder(self):
        return self.stump_builder

