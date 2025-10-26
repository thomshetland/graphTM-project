// Copyright (c) 2024 Ole-Christoffer Granmo

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 5
#endif

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
	int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
	int open_positions[BOARD_DIM*BOARD_DIM];
	int number_of_open_positions;
	int moves[BOARD_DIM*BOARD_DIM];
	int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}
			
			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) 
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	int random_empty_position_index = rand() % hg->number_of_open_positions;

	int empty_position = hg->open_positions[random_empty_position_index];

	hg->board[empty_position * 2 + player] = 1;

	hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;

	hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];

	hg->number_of_open_positions--;

	return empty_position;
}

void hg_place_piece_based_on_tm_input(struct hex_game *hg, int player)
{
	printf("TM!\n");
}

int hg_full_board(struct hex_game *hg)
{
	return hg->number_of_open_positions == 0;
}

void hg_print(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < i; j++) {
			printf(" ");
		}

		for (int j = 0; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				printf(" X");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				printf(" O");
			} else {
				printf(" ·");
			}
		}
		printf("\n");
	}
}

int main() {
	struct hex_game hg;

	int winner = -1;

	for (int game = 0; game < 10000000; ++game) {
		hg_init(&hg);

		int player = 0;
		while (!hg_full_board(&hg)) {
			int position = hg_place_piece_randomly(&hg, player);
			
			if (hg_winner(&hg, player, position)) {
				winner = player;
				break;
			}

			player = 1 - player;
		}

		if (hg.number_of_open_positions >= 75) {
			printf("\nPlayer %d wins!\n\n", winner);
			hg_print(&hg);
		}
	}
}

static inline int padded_to_compact(int padded_idx) {
    int stride = BOARD_DIM + 2;
    int i = padded_idx / stride;  // 0..D+1
    int j = padded_idx % stride;  // 0..D+1
    // convert internal cell (1..D,1..D) to 0..D*D-1
    return (i - 1) * BOARD_DIM + (j - 1);
}

#ifdef __cplusplus
extern "C" {
#endif

/* Plays one random game.
   - moves_out: length D*D, compact indices [0..D*D-1]; unused positions untouched
   - *winner_out: 0 or 1
   - returns: number of moves actually played (<= D*D)
*/
int play_random_game(int *moves_out, int *winner_out)
{
    struct hex_game hg;
    hg_init(&hg);

    int player = 0;
    int winner = -1;
    int move_idx = 0;

    while (!hg_full_board(&hg)) {
        int padded_pos = hg_place_piece_randomly(&hg, player);
        moves_out[move_idx++] = padded_to_compact(padded_pos);
        if (hg_winner(&hg, player, padded_pos)) { winner = player; break; }
        player = 1 - player;
    }
    if (winner == -1) winner = 0; // Hex should not draw; default to 0
    *winner_out = winner;
    return move_idx;
}

/* Batch generator.
   - moves_out: shape [n_games, D*D] flattened; fill unused with -1
   - winners_out: [n_games]
   - lengths_out: [n_games]
   - seed: RNG seed
   - returns 0 on success
*/
int generate_games(int n_games, int *moves_out, int *winners_out, int *lengths_out, unsigned int seed)
{
    srand(seed);
    int N = BOARD_DIM * BOARD_DIM;
    for (int g = 0; g < n_games; ++g) {
        int *row = moves_out + g * N;
        int w = -1;
        int len = play_random_game(row, &w);
        winners_out[g] = w;
        lengths_out[g] = len;
        for (int k = len; k < N; ++k) row[k] = -1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif