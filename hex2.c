#include <stdio.h>
#include <stdlib.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 11
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

void hg_print(struct hex_game *hg, FILE *file)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < i; j++) {
			fprintf(file, " ");
		}

		for (int j = 0; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				fprintf(file, " X");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				fprintf(file, " O");
			} else {
				fprintf(file, " .");
			}
		}
		fprintf(file, "\n");
	}
}

int main() {
	struct hex_game hg;

	int winner = -1;
	int count = 0;

	// Open a single file for all game outputs
	FILE *file = fopen("game_output.txt", "w");

	if (!file) {
		perror("Failed to open file");
		return 1;
	}

	for (int game = 0; count < 5; ++game) {
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
			fprintf(file, "\nGame %d\n", game + 1);
			fprintf(file, "Player %d wins!\n\n", winner);
			hg_print(&hg, file);
			count++;
		}
	}

	// Close the file after all games are written
	fclose(file);

	return 0;
}
