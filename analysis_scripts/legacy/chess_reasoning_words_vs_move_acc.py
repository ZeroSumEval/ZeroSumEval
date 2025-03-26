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

INITIAL_BOARD_STATE = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with chess match history.")
    parser.add_argument("--stockfish_path", type=str, required=True, help="Path to locally installed StockFish from https://stockfishchess.org/download/")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where visualizations will be saved.")
    return parser.parse_args()


def extract_match(matche_dir: str, engine: chess.engine.SimpleEngine) -> dict:
    
    with open(osp.join(matche_dir, 'scores.json')) as f:
        scores = json.load(f)

    with open(osp.join(matche_dir, 'turns.jsonl')) as f:
        turns = [json.loads(line) for line in f]
    
    for model, attributes in scores.items():
        if attributes['role'] == 'white':
            white_model = model
        if attributes['role'] == 'black':
            black_model = model

    board_states = [INITIAL_BOARD_STATE] + [turn['board_state'] for turn in turns]

    rationales = [turn['last_trace'].get('rationale', '') for turn in turns]

    scores = evaluate_match_moves(board_states, engine)

    move_times = [turn.get('last_move_time', None) for turn in turns]

    match = {
        'white_model': white_model,
        'black_model': black_model,
        'board_states': board_states,
        'rationales': rationales,
        'scores': scores,
        'move_times': move_times,
    }

    return match


def evaluate_match_moves(
    board_states: List[str],
    engine: chess.engine.SimpleEngine,
    depth: int = 5,
    mate_score: int = 2_000,
) -> dict:

    scores = []
    for state in board_states:
        board = chess.Board(state)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info['score'].white().score(mate_score=mate_score)
        score /= 100 # Convert centi-pawns advantage evaluation to pawns
        scores.append(score)

    return scores


def main(args: argparse.Namespace):

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    matches = []
    for match_dir in tqdm(
        os.listdir(osp.join(args.results_dir, 'matches')),
        desc='Analyzing matches',
    ):
        try:
            matches.append(
                extract_match(
                    osp.join(args.results_dir, 'matches', match_dir),
                    engine,
                )
            )
        except (FileNotFoundError, NotADirectoryError) as e:
            print(e)
            continue

    engine.quit()

    model_moves: Dict[List[Tuple[int, float, float]]] = defaultdict(list)
    for match in matches:
        n = len(match['scores']) - 1
        # for i in range(10, n): #skip book moves
        for i in range(n):
            sign, player = (+1, 'white') if i%2 == 0 else (-1, 'black')
            model = match[f'{player}_model']
            score_change = sign * (match['scores'][i+1] - match['scores'][i])
            rationale_words = len(match['rationales'][i].split())
            move_time = match['move_times'][i]
            # model_moves[model].append((rationale_words, score_change))
            model_moves[model].append((rationale_words, score_change, move_time))

    for model, moves in model_moves.items():

        # if 'claude' not in model or 'cot' not in model:
        #     continue

        rationale_words, score_changse, move_times = list(zip(*moves))
        plt.figure()
        plt.scatter(move_times, rationale_words)
        plt.title(model)
        plt.xlabel('Move time (s)')
        plt.ylabel('Rationale words')
        plt.grid()
        plt.show()

        plt.figure()
        plt.hist(move_times, range=(0, 15), bins=15, zorder=3, alpha=0.7, edgecolor='black')
        plt.title(model)
        plt.xlabel('Move time (s)')
        plt.ylabel('Frequency')
        plt.grid('--')
        plt.show()

    # bucket_size = 50
    # for model, moves in model_moves.items():

    #     results = defaultdict(list)
    #     for rationale_words, score_change in moves:
    #         correct = (score_change > -0.5)
    #         results[rationale_words // bucket_size].append(correct)

    #     x, y, y_low, y_high = [], [], [], []
    #     for words_class, scores in sorted(results.items()):
    #         max_words = bucket_size * (words_class+1)
    #         result = binomtest(k=sum(scores), n=len(scores))
    #         ci = result.proportion_ci()
    #         low, mean, high = 100*ci.low, 100*result.statistic, 100*ci.high

    #         x.append(max_words)
    #         y.append(mean)
    #         y_low.append(mean-low)
    #         y_high.append(high-mean)

    #     plt.errorbar(x, y, yerr=[y_low, y_high])
    #     plt.title(model)
    #     plt.ylabel('Correct move proportion (%)')
    #     plt.xlabel('Reasoning words')
    #     plt.grid()
    #     plt.show()

        # print(model)
        # correct = [move[1] > -0.5 for move in moves]
        # print(sum(correct) / len(correct))
        # print(round(sum(move[1] for move in moves) / len(moves), 2))
        # # binomtest(k=7, n=50, p=0.1)
        # # x = result.proportion_ci()
        # # x.low, x.high
    

def bucket_moves():
    pass

if __name__=='__main__':
    args = get_args()
    main(args)

"""
Sample usage:

uv run python chess_reasoning_words_vs_move_acc.py \
    --results_dir="../../llama_chess_pool" \
    --stockfish_path="/Users/yazeed/Documents/stockfish/stockfish-macos-m1-apple-silicon" \
    --output_dir="."
"""