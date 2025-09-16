#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import isnan

OUTPUT_DIR = "output"
CSV_PATH = "history.csv"

def load_csv(path):
    df = pd.read_csv(path)
    required = {'epoch','batch','loss','accuracy'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
    return df

def print_summary(df):
    epoch_stats = df.groupby('epoch').agg(
        loss_mean=('loss','mean'),
        loss_std=('loss','std'),
        acc_mean=('accuracy','mean'),
        acc_std=('accuracy','std'),
        batches_count=('batch','count')
    ).reset_index()

    print("\n=== SUMMARY (per epoch) ===")
    print(epoch_stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    all_losses = df['loss'].dropna()
    all_acc = df['accuracy'].dropna()
    print("\nOverall:")
    print(f" Loss: min={all_losses.min():.6f}, median={all_losses.median():.6f}, mean={all_losses.mean():.6f}, last_epoch_mean={epoch_stats['loss_mean'].iloc[-1]:.6f}")
    print(f" Acc : min={all_acc.min():.6f}, median={all_acc.median():.6f}, mean={all_acc.mean():.6f}, last_epoch_mean={epoch_stats['acc_mean'].iloc[-1]:.6f}")
    print("===========================\n")

    return epoch_stats

def plot_batches(df, metric='loss', out_fname='metric_batches.png', ymax=None):
    epochs = sorted(df['epoch'].unique())
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    max_batch_len = 0
    for e in epochs:
        sub = df[df['epoch'] == e].sort_values('batch')
        x = sub['batch'].values
        y = sub[metric].values
        if len(x) == 0:
            continue
        max_batch_len = max(max_batch_len, x.max())
        ax.plot(x, y, linewidth=0.8, alpha=0.35)
    ax.set_xlabel('batch index (within epoch)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric.title()} per batch — each epoch is a thin line')
    ax.grid(True, linestyle='--', alpha=0.4)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    plt.tight_layout()
    plt.savefig(out_fname, dpi=150)
    plt.close()
    print(f"Saved {out_fname}")

def plot_epoch_summary(epoch_stats, metric='loss', out_fname='metric_epoch.png', smooth_window=3):
    x = epoch_stats['epoch'].values
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    y = epoch_stats[mean_col].values
    ystd = epoch_stats[std_col].fillna(0).values
    y_smooth = pd.Series(y).rolling(window=smooth_window, min_periods=1, center=True).mean().values

    plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.plot(x, y, marker='o', linewidth=1.5, label='epoch mean')
    ax.plot(x, y_smooth, marker='o', linewidth=2, linestyle='--', label=f'smoothed (w={smooth_window})')
    ax.fill_between(x, y - ystd, y + ystd, alpha=0.2, label='± std')
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric.title()} per epoch — mean ± std and smoothed')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_fname, dpi=150)
    plt.close()
    print(f"Saved {out_fname}")

def main(csv_path):
    df = load_csv(csv_path)
    epoch_stats = print_summary(df)

    plot_batches(df, metric='loss', out_fname=f'{OUTPUT_DIR}/loss_batches.png')

    plot_epoch_summary(epoch_stats, metric='loss', out_fname=f'{OUTPUT_DIR}/loss_epoch.png', smooth_window=3)

    plot_batches(df, metric='accuracy', out_fname=f'{OUTPUT_DIR}/accuracy_batches.png', ymax=1.0)

    plot_epoch_summary(epoch_stats, metric='acc', out_fname=f'{OUTPUT_DIR}/accuracy_epoch.png', smooth_window=3)

if __name__ == "__main__":
    path = f'{OUTPUT_DIR}/{CSV_PATH}' if len(sys.argv) == 1 else sys.argv[1]
    main(path)
