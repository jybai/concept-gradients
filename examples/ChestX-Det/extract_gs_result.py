import argparse
import os
import pandas as pd
import numpy as np
from glob import glob

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='Path to save extracted grid search results.')
    parser.add_argument('--logs_dir', type=str, help='Directory to save all model checkpoint and logs.')
    parser.add_argument('--metrics_fname', type=str, help='Default filename for all emtrics logs.', default='metrics.csv')
    parser.add_argument('--metric', type=str, help='Metric to retrieve.', default='tst_f1')
    parser.add_argument('--optimal_criteria', type=str, help='Metric smaller or larger is better?', 
                        default='larger', choices=['smaller', 'larger'])
    return parser.parse_args()

def main():
    args = parse_arguments()
    metrics_paths = glob(os.path.join(args.logs_dir, "*", args.metrics_fname))

    exp_names = []
    metrics = []
    for metrics_path in metrics_paths:

        exp_name = os.path.basename(os.path.dirname(metrics_path))

        df = pd.read_csv(metrics_path)
        if args.optimal_criteria == 'larger':
            metric = df[args.metric].max()
        elif args.optimal_criteria == 'smaller':
            metric = df[args.metric].min()

        exp_names.append(exp_name)
        metrics.append(metric)

    metrics_df = pd.DataFrame(data={'exp_name': exp_names, f'best_{args.metric}': metrics})
    metrics_df = metrics_df.sort_values('exp_name')
    metrics_df.to_csv(args.save_path, index=False)

if __name__ == '__main__':
    main()
