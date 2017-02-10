from __future__ import print_function
import os
import argparse
import csv
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_fnames', type=str, nargs='+')
    parser.add_argument('--plot', type=int, default=1)
    args = parser.parse_args()

    errors_row_format = '{:>30}{:>15.3f}{:>15.3f}{:>15.3f}'
    for csv_fname in args.csv_fnames:
        with open(csv_fname, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            lambdas = []
            costs = []
            std_errs = []
            for row in reader:
                lambda_, cost, std_err = map(float, row[:3])
                lambdas.append(lambda_)
                costs.append(cost)
                std_errs.append(std_err)
            condition = os.path.splitext(os.path.basename(csv_fname))[0]
            best_cost, best_lambda, best_std_err = max(zip(costs, lambdas, std_errs))
            print(errors_row_format.format(condition, best_lambda, best_cost, best_std_err))
            if args.plot:
                lambdas, costs, std_errs = zip(*sorted(zip(lambdas, costs, std_errs)))
                plt.plot(lambdas, costs, label=condition)
    if args.plot:
        plt.legend(loc='lower left')
        plt.show()


if __name__ == '__main__':
    main()
