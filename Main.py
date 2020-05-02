import pandas as pandas
from DecisionStump import StumpBuilder, prep_bank_data
from AdaBoost import AdaBoost
from Bagging import Bagging
from RandomForest import RandomForest
from DecisionStump import StumpBuilder
import math

att_ref = ["date", "aspect", "elevation"]
val_ref = [["10", "11", "12", "1", "2", "3", "4", "5", "6"], ["1", "2", "3", "4", "5", "6", "7", "8"], ["5000", "5500", "6000", "6500", "7000", "7500", "8000", "8500", "9000", "9500", "10000",
                                                                   "10500" ,"11000", "11500", "12000"]]
label_ref = ["0", "1"]

file_data = "../../Downloads/EnsemlbleLearning 2/EnsemlbleLearning/data.csv"
df = pandas.read_csv(file_data, header=None)

df = df.applymap(str)

# Project Data
for i in range(50):

    df = df.sample(frac=1)

    train = df.head(len(df) / 2 + 1)
    test = df.tail(len(df)/2)

    prep_bank_data(train)

    train_mistakes = 0.0
    test_mistakes = 0.0

    boost = AdaBoost(train, att_ref, val_ref, label_ref, 50)

    for index, row in train.iterrows():
            if boost.decide(row, 50) != row[3]:
                train_mistakes = train_mistakes + 1.0

    for index, row in test.iterrows():
        if boost.decide(row, 50) != row[3]:
            test_mistakes = test_mistakes + 1.0

    train_er = train_mistakes / len(train)
    test_er = test_mistakes / len(test)

    print(train_er, test_er)






# for i in range(10):
#
#     df = df.sample(frac=1)
#     df = df.applymap(str)
#
#     train = df.head(len(df) / 2 + 1)
#     test = df.tail(len(df)/2)
#
#     prep_bank_data(train)
#
#     train_mistakes = 0.0
#     test_mistakes = 0.0
#
#     tree_builder = StumpBuilder()
#     tree = tree_builder.build_stump(train, att_ref, val_ref, label_ref, 1)
#
#     for index, row in train.iterrows():
#         if tree_builder.decide(tree, row) != row[3]:
#             train_mistakes = train_mistakes + 1.0
#
#     for index, row in test.iterrows():
#         if tree_builder.decide(tree, row) != row[3]:
#             test_mistakes = test_mistakes + 1.0
#
#     train_er = train_mistakes / len(train)
#     test_er = test_mistakes / len(test)
#
#     print(train_er, test_er)








# ada = AdaBoost(df, att_ref, val_ref, label_ref, 200)

# bag = Bagging(df, att_ref, val_ref, label_ref, 20)

# rand = RandomForest(df, att_ref, val_ref, label_ref, 20)



# ---------------------------------Data Analysis------------------------------
# ada_test_er = []
# ada_train_er = []
# bag_test_er = []
# bag_train_er = []
# rand_test_er = []
# rand_train_er = []
# classifiers_used = 20




# stumps = ada.get_stumps()
# str_build = ada.get_builder()
# test_er = []
# train_er = []
# ada_stump_count = 0.0
# for i in range(len(stumps)):
#     for index, row in dt.iterrows():
#         if str_build.decide(stumps[i],row) != row[16]:
#             ada_stump_count = ada_stump_count +1
#     test_er.append(ada_stump_count/len(dt) * 100)
#     ada_stump_count = 0.0
#
# ada_stump_count = 0.0
# for i in range(len(stumps)):
#     for index, row in df.iterrows():
#         if str_build.decide(stumps[i],row) != row[16]:
#             ada_stump_count = ada_stump_count +1
#     train_er.append(ada_stump_count/len(df) * 100)
#     ada_stump_count = 0.0
#
# print("Testing error stumps")
# for i in range(len(test_er)):
#     print(test_er[i])
#
# print("Training error stumps")
# for i in range(len(train_er)):
#     print(train_er[i])


# print("Random Forest Test Error:")
# for i in range(len(rand_test_er)):
#     print(rand_test_er[i])
#
# print("Random Forest Training Error:")
# for i in range(len(rand_train_er)):
#     print(rand_train_er[i])


