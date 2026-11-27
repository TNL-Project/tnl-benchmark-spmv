#!/usr/bin/python3
import pandas


class MultiindexCreator:

    def __init__(self, depth=0):
        self.depth = depth
        self.index = [[] for _ in range(depth)]
        self.data = [[]]

    def set_multiindex_depth(self, depth):
        self.depth = depth

    def add_entry(self, entry):
        if len(entry) > self.depth:
            raise Exception("Entry length exceeds depth")
        for i in range(self.depth - len(entry)):
            entry.append("")
        for i in range(self.depth):
            self.index[i].append(entry[i])
        self.data[0].append("")

    def add_entries(self, entries):
        for entry in entries:
            self.add_entry(entry)

    def print_multiindex(self):
        for i in range(len(self.index[0])):
            for j in range(self.depth):
                print(f"|{self.index[j][i]}", end="")
            print("|")

    def get_multiindex(self):
        multiColumns = pandas.MultiIndex.from_arrays(self.index)
        return multiColumns, self.data
