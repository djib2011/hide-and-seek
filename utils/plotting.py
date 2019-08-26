import tensorflow as tf
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import interpolate
import struct
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

alphas = np.arange(0.1, 1.01, 0.05)


def load_and_process_epoch_logs_from_dir(log_dir, extend_to_length=100):

    def read_hns_epoch_logs(path_to_events_file):
        logs = {'Average loss per epoch': [],
                'Validation accuracy': []}

        for e in tf.compat.v1.train.summary_iterator(path_to_events_file):
            for v in e.summary.value:
                logs[v.tag].append(struct.unpack('f', v.tensor.tensor_content)[0])

        if len(logs['Validation accuracy']) < len(logs['Average loss per epoch']):
            logs['Validation accuracy'].append(logs['Validation accuracy'][-1])

        return pd.DataFrame(logs)

    epoch_event_files = [str(list((d / 'epoch').glob('*'))[0]) for d in Path(log_dir).glob('*')]

    # Read log files
    logs = []
    for ev in epoch_event_files:
        log = read_hns_epoch_logs(ev)
        log['step'] = log.index
        logs.append(log)

    # make extend the logs that have stopped earlier
    if extend_to_length:
        max_len = extend_to_length
    else:
        max_len = max([len(log) for log in logs])

    new_logs = []

    for log in logs:
        steps = range(log.iloc[-1, -1] + 1, max_len)
        length = max_len - log.iloc[-1, -1] - 1
        pad = {c: [None] * length for c in log.columns}
        pad['step'] = steps
        pad = pd.DataFrame(pad)
        fill_vals = log.iloc[-1]
        log = pd.concat([log, pad], sort=False)
        log = log.fillna(fill_vals)
        new_logs.append(log)

    return new_logs


def load_and_process_batch_logs_from_dir(log_dir, resample_one_every=10, extend_to_length=39100):
    def read_hns_batch_logs(path_to_events_file):
        logs = {'classification loss': [],
                'gradients': [],
                'loss monitor': [],
                'loss regulator': [],
                'mask loss': [],
                'percentage hidden': [],
                'pixels hidden': [],
                'pixels kept': [],
                'total loss': []}

        for e in tf.compat.v1.train.summary_iterator(path_to_events_file):
            for v in e.summary.value:
                logs[v.tag].append(struct.unpack('f', v.tensor.tensor_content)[0])

        return pd.DataFrame(logs)

    def resample_log(log, one_every=10):
        interp_values = np.linspace(log.step.iloc[0], log.step.iloc[-1], len(log) // one_every)
        new_columns = {}
        for c in log.columns:
            if c == 'step':
                new_columns[c] = range(len(interp_values))
            else:
                f = interpolate.interp1d(log['step'], log[c])
                new_columns[c] = f(interp_values)

        return pd.DataFrame(new_columns)

    batch_event_files = [str(list((d / 'batch').glob('*'))[0]) for d in Path(log_dir).glob('*')]

    # Read log files
    logs = []
    for ev in batch_event_files:
        log = read_hns_batch_logs(ev)
        log['step'] = log.index
        logs.append(log)

    # make extend the logs that have stopped earlier
    if extend_to_length:
        max_len = extend_to_length
    else:
        max_len = max([len(log) for log in logs])

    new_logs = []

    for log in logs:
        steps = range(log.iloc[-1, -1] + 1, max_len)
        length = max_len - log.iloc[-1, -1] - 1
        pad = {c: [None] * length for c in log.columns}
        pad['step'] = steps
        pad = pd.DataFrame(pad)
        fill_vals = log[-10:].mean()
        log = pd.concat([log, pad], sort=False)
        log = log.fillna(fill_vals)
        if resample_one_every:
            log = resample_log(log, one_every=resample_one_every)
        new_logs.append(log)

    return new_logs


def read_results(result_dir):
    result_files = sorted(Path(result_dir).rglob('results.pkl'))
    complete_alphas = np.arange(0.1, 1.01, 0.05)
    frames = []
    for i, result_file in enumerate(result_files):
        results = pkl.load(open(str(result_file), 'rb'))
        existing_alphas = [x[0] for x in sorted(results.items(), key=lambda kv: kv[0])]
        accuracies = [x[1] for x in sorted(results.items(), key=lambda kv: kv[0])]
        original_df = pd.DataFrame({'alpha': existing_alphas, 'accuracy': accuracies, 'experiment': i})
        pad = pd.DataFrame({'alpha': complete_alphas, 'accuracy': np.nan, 'experiment': i})
        df = pd.merge(left=original_df, right=pad, how='outer', on='alpha')[['accuracy_x', 'alpha', 'experiment_y']]
        df.columns = ['accuracy', 'alpha', 'experiment']
        try:
            df = df.fillna(original_df[original_df['alpha'] == original_df['alpha'].min()].iloc[0])
        except IndexError:
            print('Error in file:', result_file)
        frames.append(df)

    return pd.concat(frames)


def filter_logs_on_val_acc(epoch_logs, batch_logs, baseline, percentage=0.9, return_epoch=False):
    valid_idx = [i for i in range(len(epoch_logs))
                 if epoch_logs[i]['Validation accuracy'].iloc[-1] > baseline * percentage]

    if return_epoch:
        return [epoch_logs[i] for i in range(len(epoch_logs)) if i in valid_idx]
    else:
        return [batch_logs[i] for i in range(len(batch_logs)) if i in valid_idx]


def filter_logs_on_pix_hidden(batch_logs, percentage=0.9):
    valid_idx = [i for i in range(len(batch_logs))
                 if batch_logs[i]['percentage hidden'].iloc[-1] > percentage * 100]

    return [batch_logs[i] for i in range(len(batch_logs)) if i in valid_idx]


def fancy_plot(logs, column_name, best_plot, color='C0', label=None):
    max_trace = [max([log[column_name].iloc[i] for log in logs]) for i in range(len(logs[0]))]
    min_trace = [min([log[column_name].iloc[i] for log in logs]) for i in range(len(logs[0]))]
    plt.plot(logs[best_plot]['step'], logs[best_plot][column_name], color=color, label=label)
    plt.plot(logs[best_plot]['step'], max_trace, color=color, alpha=0.6, zorder=-1)
    plt.plot(logs[best_plot]['step'], min_trace, color=color, alpha=0.6, zorder=-1)
    plt.xlabel('step')
    plt.ylabel(column_name)


def plot_all_logs(logs, column_name, baseline=None):
    for i in range(len(logs)):
        plt.plot(logs[i]['step'], logs[i][column_name], label=str(i))
    if baseline:
        plt.plot(logs[0]['step'], [baseline] * len(logs[0]), label='baseline', c='0.5', ls='--')
    plt.xlabel('step')
    plt.ylabel(column_name)
    plt.legend(bbox_to_anchor=(1.01, 1.0))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_all_results(results, baseline=None):

    for i in range(10):
        f = results[results['experiment'] == i].sort_values(by='alpha')
        plt.plot(f['alpha'], f['accuracy'])
    plt.xlabel('alpha')
    plt.ylabel('accuracy')

    if baseline:
        plt.plot(alphas, [baseline] * len(alphas), label='baseline', c='0.5', ls='--')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_average_results(results, baseline=None, label=None):

    if baseline:
        plt.plot(alphas, [baseline] * len(alphas), label='baseline', c='0.5', ls='--')

    if not label:
        label = 'average performance accross models'

    sns.lineplot(x='alpha', y='accuracy', data=results, label=label)

    plt.legend(bbox_to_anchor=(1.01, 1.0))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
