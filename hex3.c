#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BOARD_DIM 4  // Board size for a 3x3 Hex game

// Define the neighbors for a Hex grid (6 neighbors per hex cell)
int neighbors[6][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, 1}, {1, -1} // Top, Bottom, Left, Right, Top-Right, Bottom-Left
};

// Structure for the Hex game
struct hex_game {
    char board[BOARD_DIM][BOARD_DIM];  // 'X' for Player 0, 'O' for Player 1, '.' for empty
};

// Initialize the game board with empty positions
void hg_init(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM; i++) {
        for (int j = 0; j < BOARD_DIM; j++) {
            hg->board[i][j] = ' ';  // All positions start empty
        }
    }
}

// Check if the given coordinates are within the board boundaries
int is_within_bounds(int row, int col) {
    return row >= 0 && row < BOARD_DIM && col >= 0 && col < BOARD_DIM;
}

// Depth-first search to check if a path exists for Player 0 ('X') from top to bottom
int dfs_check_winner_X(struct hex_game *hg, int row, int col, int visited[BOARD_DIM][BOARD_DIM]) {
    if (row == BOARD_DIM - 1) {
        return 1;  // Reached bottom row, Player 0 wins
    }

    visited[row][col] = 1;  // Mark as visited

    for (int i = 0; i < 6; i++) {
        int new_row = row + neighbors[i][0];
        int new_col = col + neighbors[i][1];
        if (is_within_bounds(new_row, new_col) && hg->board[new_row][new_col] == 'X' && !visited[new_row][new_col]) {
            if (dfs_check_winner_X(hg, new_row, new_col, visited)) {
                return 1;
            }
        }
    }
    return 0;
}

// Check if Player 0 ('X') has won (connected top to bottom)
int check_winner_X(struct hex_game *hg) {
    int visited[BOARD_DIM][BOARD_DIM] = {0};

    // Start the search from the top row
    for (int col = 0; col < BOARD_DIM; col++) {
        if (hg->board[0][col] == 'X' && !visited[0][col]) {
            if (dfs_check_winner_X(hg, 0, col, visited)) {
                return 1;
            }
        }
    }
    return 0;
}

// Depth-first search to check if a path exists for Player 1 ('O') from left to right
int dfs_check_winner_O(struct hex_game *hg, int row, int col, int visited[BOARD_DIM][BOARD_DIM]) {
    if (col == BOARD_DIM - 1) {
        return 1;  // Reached rightmost column, Player 1 wins
    }

    visited[row][col] = 1;  // Mark as visited

    for (int i = 0; i < 6; i++) {
        int new_row = row + neighbors[i][0];
        int new_col = col + neighbors[i][1];
        if (is_within_bounds(new_row, new_col) && hg->board[new_row][new_col] == 'O' && !visited[new_row][new_col]) {
            if (dfs_check_winner_O(hg, new_row, new_col, visited)) {
                return 1;
            }
        }
    }
    return 0;
}

// Check if Player 1 ('O') has won (connected left to right)
int check_winner_O(struct hex_game *hg) {
    int visited[BOARD_DIM][BOARD_DIM] = {0};

    // Start the search from the leftmost column
    for (int row = 0; row < BOARD_DIM; row++) {
        if (hg->board[row][0] == 'O' && !visited[row][0]) {
            if (dfs_check_winner_O(hg, row, 0, visited)) {
                return 1;
            }
        }
    }
    return 0;
}

// Place a piece randomly on the board for the given player ('X' or 'O')
void place_piece_randomly(struct hex_game *hg, char player) {
    int row, col;
    do {
        row = rand() % BOARD_DIM;
        col = rand() % BOARD_DIM;
    } while (hg->board[row][col] != ' ');  // Find an empty spot

    hg->board[row][col] = player;
}

// Write the final board state and winner to a CSV file
void write_final_board_to_csv(struct hex_game *hg, int winner) {
    FILE *f = fopen("hex_game_results.csv", "a");

    if (f == NULL) {
        printf("Error opening file!\n");
        return;
    }

    // Write the final board state as a string
    for (int i = 0; i < BOARD_DIM; i++) {
        for (int j = 0; j < BOARD_DIM; j++) {
            fprintf(f, "%c", hg->board[i][j]);
        }
    }

    // Write the winner (0 or 1)
    fprintf(f, ",%d\n", winner);

    fclose(f);
}

int main() {
    srand(time(NULL));  // Initialize random seed

    struct hex_game hg;
    int game_count = 5000;  // Number of games to simulate

    for (int game = 0; game < game_count; ++game) {
        hg_init(&hg);  // Initialize a new game

        int winner = -1;
        int turn = 0;  // 0 for Player X, 1 for Player O

        // Play until we have a winner or the board is full
        for (int moves = 0; moves < BOARD_DIM * BOARD_DIM; moves++) {
            place_piece_randomly(&hg, (turn == 0) ? 'X' : 'O');

            // Check if any player has won
            if (check_winner_X(&hg)) {
                winner = 0;  // Player X wins
                break;
            }
            if (check_winner_O(&hg)) {
                winner = 1;  // Player O wins
                break;
            }

            // Alternate turns
            turn = 1 - turn;
        }

        // Write the final board state and winner to the CSV file
        write_final_board_to_csv(&hg, winner);
    }

    printf("Simulation completed. Results written to hex_game_results.csv\n");
    return 0;
}