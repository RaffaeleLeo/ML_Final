import math
import pandas as pandas
import copy
from AttNode import AttNode
import random


class StumpBuilder:

    def decide(self, node, data):
        if len(node.leaves) == 0:
            return node.attribute
        att_index = self.get_attribute_id(node.attribute)
        leaf_index = self.get_value_id(node.attribute, data[att_index])

        return self.decide(node.leaves[leaf_index], data)

    def build_stump(self, data_frame, attributes, values, labels, levels):
        self.att_ref = attributes
        self.val_ref = values
        return self.build_stump_ent(data_frame, attributes, labels, levels)

    def build_rand(self, data_frame, attributes, values, labels, levels):
        self.att_ref = attributes
        self.val_ref = values
        return self.build_rand_tree(data_frame, attributes, labels, 2)


    def build_stump_ent(self, data_frame, attributes, labels, levels):
        l_column = len(data_frame.columns) - 2
        label_data = []
        for label in labels:
            label_data.append(data_frame.loc[data_frame[l_column] == label][4].sum())

        # total_examples and total_entropy
        total_entropy = self.get_entropy(label_data, 1)

        # Base Cases
        if len(attributes) == 0 or levels == 0:
            common_label = labels[label_data.index(max(label_data))]
            return AttNode(common_label)
        elif total_entropy == 0.0:
            only_label = labels[label_data.index(max(label_data))]
            return AttNode(only_label)

        # Information Gain Calculations
        info_gain = []
        for att in attributes:

            attribute_entropy = 0
            for value in self.get_values(att):
                v_data = []
                val_frame = data_frame.loc[data_frame[self.get_attribute_id(att)] == value]
                for label in labels:
                    v_data.append(val_frame.loc[val_frame[l_column] == label][4].sum())
                    #might be a problem here with the 1
                attribute_entropy = attribute_entropy + (sum(v_data) / float(1)) * self.get_entropy(v_data, sum(v_data))
            info_gain.append(total_entropy - attribute_entropy)

        root_attribute = attributes[info_gain.index(max(info_gain))]
        root_values = self.get_values(root_attribute)
        root_node = AttNode(root_attribute)

        leaf_tables = []
        for value in root_values:
            df_temp = data_frame.loc[data_frame[self.get_attribute_id(root_attribute)] == value]
            leaf_tables.append(df_temp)

        # Copies attributes list to avoid reference confusion
        new_attributes = copy.copy(attributes)
        new_attributes.remove(root_attribute)

        levels = levels - 1
        # Makes a recursive call to all potential leaf nodes
        for value in root_values:
            root_node.add_leaf(self.build_stump_ent(leaf_tables[self.get_value_id(root_attribute, value)], new_attributes, labels, levels))

        return root_node


#--------------------RANDOM TREE BUILDER---------------------------------------------

    def build_rand_tree(self, data_frame, attributes, labels, levels):
        df = pandas.DataFrame()
        for depth in range(6):
            index = random.randint(0, 4999)
            df = df.append(data_frame.loc[[index]])

        l_column = len(df.columns) - 2
        label_data = []
        for label in labels:
            label_data.append(df.loc[df[l_column] == label][4].sum())

        # total_examples and total_entropy
        total_entropy = self.get_entropy(label_data, 1)

        # Base Cases
        if len(attributes) == 0 or levels == 0:
            common_label = labels[label_data.index(max(label_data))]
            return AttNode(common_label)
        elif total_entropy == 0.0:
            only_label = labels[label_data.index(max(label_data))]
            return AttNode(only_label)

        # Information Gain Calculations
        info_gain = []
        for att in attributes:

            attribute_entropy = 0
            for value in self.get_values(att):
                v_data = []
                val_frame = df.loc[df[self.get_attribute_id(att)] == value]
                for label in labels:
                    v_data.append(val_frame.loc[val_frame[l_column] == label][4].sum())
                    #might be a problem here with the 1
                attribute_entropy = attribute_entropy + (sum(v_data) / float(1)) * self.get_entropy(v_data, sum(v_data))
            info_gain.append(total_entropy - attribute_entropy)

        root_attribute = attributes[info_gain.index(max(info_gain))]
        root_values = self.get_values(root_attribute)
        root_node = AttNode(root_attribute)

        # Copies attributes list to avoid reference confusion
        new_attributes = copy.copy(attributes)
        new_attributes.remove(root_attribute)

        levels = levels - 1
        # Makes a recursive call to all potential leaf nodes
        for value in root_values:
            root_node.add_leaf(self.build_rand_tree(data_frame, new_attributes, labels, levels))

        return root_node

    # -----------------------------------------------Helper Methods-------------------------------------------------------------------------------------------------------------------
    def get_entropy(self, data, total):
        value_entropy = 0
        for d in data:
            if d != 0:
                value_entropy = value_entropy + (float(d) / float(total)) * math.log(float(d) / float(total), len(data)).real
        if (value_entropy != 0):
            value_entropy = value_entropy * -1
        return value_entropy

    def get_attribute_id(self, att):
        return self.att_ref.index(att)

    def get_values(self, att):
        index = self.get_attribute_id(att)
        return self.val_ref[index]

    def get_value_id(self, att, value):
        return self.val_ref[self.get_attribute_id(att)].index(value)

def prep_bank_data(df):
    # age_t = df[0].median()
    # bal_t = df[5].median()
    # day_t = df[9].median()
    # dur_t = df[11].median()
    # camp_t = df[12].median()
    # pday_t = df[13].median()
    # prev_t = df[14].median()
    #
    # df.loc[df[0] <= age_t, 0] = 0
    # df.loc[df[0] > age_t, 0] = 1
    #
    # df.loc[df[5] <= bal_t, 5] = 0
    # df.loc[df[5] > bal_t, 5] = 1
    #
    # df.loc[df[9] <= day_t, 9] = 0
    # df.loc[df[9] > day_t, 9] = 1
    #
    # df.loc[df[11] <= dur_t, 11] = 0
    # df.loc[df[11] > dur_t, 11] = 1
    #
    # df.loc[df[12] <= camp_t, 12] = 0
    # df.loc[df[12] > camp_t, 12] = 1
    #
    # df.loc[df[13] <= pday_t, 13] = 0
    # df.loc[df[13] > pday_t, 13] = 1
    #
    # df.loc[df[14] <= prev_t, 14] = 0
    # df.loc[df[14] > prev_t, 14] = 1

    weights = [1 / float(len(df))] * len(df)
    df[4] = weights

