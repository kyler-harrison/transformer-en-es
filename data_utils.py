import json
import time
import pandas as pd


def make_csv(vec_path, csv_path):
    """
    From downloaded "csv" writes a formatted csv of word,vec pairs.

    vec_path: path to raw word2vec file
    csv_path: output path to write corrected csv to
    """

    with open(vec_path, "r") as vec_fi:
        with open(csv_path, "w") as csv_fi:
            csv_fi.write("word,vector\n")
            line = vec_fi.readline()  # dont want first line
            line = vec_fi.readline()  # first real line of data
            count = 0

            while line:
                line = line.replace('"', '')
                split_line = line.split(' ')
                split_line[-1] = split_line[-1].replace('\n', '')
                csv_fi.write(f'"{split_line[0]}",[{" ".join(split_line[1:])}]\n')

                line = vec_fi.readline()
                count += 1

                if count % 100000 == 0:
                    print(count)


def main():
    vec_paths = ["cc.es.300.vec", "cc.en.300.vec"]
    csv_paths = ["cc.es.300.csv", "cc.en.300.csv"]

    for (vec_path, csv_path) in zip(vec_paths, csv_paths):
        print(f"starting {vec_path}...")
        make_csv(vec_path, csv_path)


if __name__ == "__main__":
    main()
