#!/Users/jdkent/miniconda3/envs/aim1_valid/bin/python

import argparse
import json
import pickle

from bids.layout import BIDSLayout
import pandas as pd

import notebook_functions as nf

def proc_bold_qa(bold_qa_file):
    bold_qa = pd.read_csv(bold_qa_file, sep='\t')
    # drop the rest rows
    bold_qa = bold_qa[~bold_qa['bids_name'].str.contains('.*rest.*')]
    
    split_columns = bold_qa['bids_name'].str.split('_|-', n = 7, expand = True)
    bold_qa['task'] = split_columns[5]
    bold_qa['participant_id'] = split_columns[1]
    return bold_qa


def filter_bad_participants(layout, qa_file):
    bold_qa = proc_bold_qa(qa_file)
    bad_participants = bold_qa[bold_qa['fd_num'] >= 100]['participant_id'].unique()
    all_participants = layout.entities['subject']
    good_participants = list(set(all_participants.unique()) - set(bad_participants))

    return good_participants

    
def main():
    parser = argparse.ArgumentParser(description='Permutation Tests')

    # use dictionary as arguments
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
    parser.add_argument('sample1', type=json.loads,
                        help='dictionary specifying the first sample')
    parser.add_argument('sample2', type=json.loads,
                        help='dictionary specifying the second sample')
    parser.add_argument('sample_dir', help='directory where correlation matrices are')
    parser.add_argument('qa_file', help='quality assurance file to filter bad participants')
    parser.add_argument('n_threads', type=int, help='number of threads to use')
    parser.add_argument('permutations', type=int, help='number of permutations to perform')
    parser.add_argument('outfile', help='pickle file to write results to')

    opts = parser.parse_args()
    layout = BIDSLayout(opts.sample_dir, validate=False, config=['bids', 'derivatives'])
    
    good_participants = filter_bad_participants(layout, opts.qa_file)

    if opts.sample1['task'] != opts.sample2['task']:
        sample1_layout_objs = nf.get_layout_objects(
            layout, ['switch', 'repeat', 'single'],
            suffix='correlation', extension='tsv',
            task=opts.sample1['task'], subject=good_participants)

        sample2_layout_objs = nf.get_layout_objects(
            layout, ['switch', 'repeat', 'single'],
            suffix='correlation', extension='tsv',
            task=opts.sample2['task'], subject=good_participants)
        sample1_label = opts.sample1['task']
        sample2_label = opts.sample2['task']
    else:
        sample1_layout_objs = sample2_layout_objs = nf.get_layout_objects(
            layout, ['switch', 'repeat', 'single'],
            suffix='correlation', extension='tsv',
            task=opts.sample2['task'], subject=good_participants)
        sample1_label = opts.sample1['condition']
        sample2_label = opts.sample2['condition']

    sample1_correlation_matrices = sample1_layout_objs[opts.sample1['condition']]
    sample2_correlation_matrices = sample2_layout_objs[opts.sample2['condition']]

    permutation_results = nf.count_positives_from_permutations(
        sample1_correlation_matrices, sample2_correlation_matrices,
        sample1_label, sample2_label, nthreads=opts.n_threads, permutations=opts.permutations, use_python=True)

    with open(opts.outfile, "wb") as pklw:
        pickle.dump(permutation_results, pklw)


if __name__ == '__main__':
    main()
