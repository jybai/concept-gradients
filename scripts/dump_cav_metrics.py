import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_path', type=str)
    args = parser.parse_args()

    with np.load(args.npz_path) as f:
        for k, v in f.items():
            print(f"{k:<15}{np.mean(v):.4f}")

if __name__ == '__main__':
    main()
