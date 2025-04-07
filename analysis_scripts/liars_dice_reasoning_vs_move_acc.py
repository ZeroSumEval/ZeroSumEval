import json
import os
import os.path as osp
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import chess
import chess.engine
from scipy.stats import binomtest


"""
Sample usage:

python chess_reasoning_words_vs_move_acc.py \
    --results_dir="path" \
    --stockfish_path="path" \
    --output_dir="path"
"""

matplotlib.rc('font', family='serif', serif=['Times'], size=9)
plt.rcParams.update({'figure.autolayout': True})

# RGB colors from ZSE logo
COLORS = [
    (41/255, 44/255, 147/255),  # Debate
    (14/255, 140/255, 247/255),  # Chess
]

# COLM template widths in inches
# TODO: update from ACL numbers to COLM
TEXT_WIDTH = 3.03
LINE_WIDTH = 6.30


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with chess match history.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where visualizations will be saved.")
    return parser.parse_args()


def extract_match(match_dir: str) -> dict:
    
    with open(osp.join(match_dir, 'scores.json')) as f:
        scores = json.load(f)

    with open(osp.join(match_dir, 'turns.jsonl')) as f:
        turns = [json.loads(line) for line in f]

    if turns[-1].get('last_move', None) != '[Call]':
        return {}
    
    for model, attributes in scores.items():
        if attributes['role'] == 'player_0':
            player_0 = model
        if attributes['role'] == 'player_1':
            player_1 = model

    last_rationale = turns[-1]['last_trace'].get('rationale', '')
    last_rationale_words = len(last_rationale.split())

    n = len(turns)
    last_model = [player_0, player_1][n%2]

    last_move_time = turns[-1]['last_move_time']

    won = (scores[last_model]['score'] > 0)

    match = {
        'rationale_words': last_rationale_words,
        'move_time': last_move_time,
        'model': last_model,
        'won': won,
    }

    return match


def main(args: argparse.Namespace):

    matches = []
    for match_dir in tqdm(
        os.listdir(osp.join(args.results_dir, 'matches')),
        desc='Analyzing matches',
    ):
        try:
            match = extract_match(osp.join(args.results_dir, 'matches', match_dir))
            if match:
                matches.append(match)
        except (FileNotFoundError, NotADirectoryError) as e:
            print(e)
            continue

    model_matches: Dict[List[dict]] = defaultdict(list)
    for match in matches:
        model = match.pop('model')
        model_matches[model].append(match)

    BUCKET_SIZE = 5
    BUCKET_KEY = 'move_time'
    NUM_BUCKETS = 4

    # BUCKET_SIZE = 50
    # BUCKET_KEY = 'rationale_words'
    # MAX_BUCKET = float('inf')
    for model, matches in model_matches.items():

        bucketed_results = defaultdict(list)
        for match in matches:
            value = match[BUCKET_KEY]
            bucketed_results[value // BUCKET_SIZE].append(match['won'])
        
        x, y, y_low, y_high = [], [], [], []
        for value_class, scores in sorted(bucketed_results.items()):
            if value_class >= NUM_BUCKETS:
                break
            max_value = BUCKET_SIZE * (value_class+1)
            result = binomtest(k=sum(scores), n=len(scores))
            ci = result.proportion_ci()
            low, mean, high = 100*ci.low, 100*result.statistic, 100*ci.high

            # print(max_value)
            # print(f'k={sum(scores)}, n={len(scores)}')

            x.append(max_value)
            y.append(mean)
            y_low.append(mean-low)
            y_high.append(high-mean)

        plt.errorbar(x, y, yerr=[y_low, y_high])
        plt.title(model)
        plt.ylabel('Correct move proportion (%)')
        plt.xlabel(BUCKET_KEY)
        plt.grid()
        plt.show()

def bucket_moves():
    pass

if __name__=='__main__':
    args = get_args()
    main(args)

"""
Sample usage:

uv run python liars_dice_reasoning_vs_move_acc.py \
    --results_dir="../../rankings-3-9-25_liars_dice_reasoning_1000" \
    --output_dir="."
"""