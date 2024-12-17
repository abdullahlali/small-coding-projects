#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

//do NOT change this part

#define INVALID_SYMBOL -1

struct forest {
  char ghost_map[200];
  char player_map[200];
  int width;
  int height;
};

const int forest_reveal_alive = 1;
const int forest_reveal_dead  = 2;
const int forest_reveal_win   = 3;
static const char ghost_presence = 'G';
static const char ghost_absence = '0';
static const int MAX_IDX = 200;
static const char MARKED = '*';
static const char UNREVEALED = '_';
static const char NO_ADJACENT_GHOST = ' ';
static const char TRANSFER_CHAR = '0';
static const int find_last_digit = 10;


// forest_enter(wdth, hght) returns an initialized forest of width wdth and
//   height hght. The data for the initialization is read from input using the
//   following format: wdth * hght characters of either 'G' or '0', where 'G'
//   represents the presence and '0' the absence of a ghost.
// requires: wdth and height must be positive
//           wdth * hght must not exceed 200
// effects:  reads from input
struct forest forest_enter(int wdth, int hght) {
  assert(wdth > 0);
  assert(hght > 0);
  assert(wdth * hght <= MAX_IDX);
  struct forest frst = {0};
  frst.width = wdth;
  frst.height = hght;
  for (int i = 0; i < wdth * hght; i++) {
    frst.player_map[i] = UNREVEALED;
  }
  char input = 0;
  int j = 0;
  while (j < wdth * hght && (scanf(" %c", &input) == 1)) {
    frst.ghost_map[j] = input;
    j += 1;
  }
  return frst;
}


// transfer-2D_into_1D_pos(frst, x, y) transfers a position in 2D division into
//   the position in 1D array
// requires: frst is valid
static int transfer_2D_into_1D_pos(const struct forest *frst, int x, int y) {
  assert(frst);
  return frst->width * y + x;
}


// check_valid_pos(frst, x, y) returns true if x is in range bwteen 0 and width
//   and y is in range between 0 and height, false otherwise.
// requires: frst is valid
static bool check_valid_pos(const struct forest *frst, int x, int y) {
  assert(frst);
  return (x >= 0 && y >= 0 && x < frst->width && y < frst->height);
}


// forest_mark(frst, x, y) toggles the mark at x- and y-coordinates (x, y) in
//   the forest frst, indicating that it might / might not be haunted by a
//   ghost.
// effects: mutates frst
void forest_mark(struct forest *frst, int x, int y) {
  assert(frst);
  int pos = transfer_2D_into_1D_pos(frst, x, y);
  if (check_valid_pos(frst, x, y)) {
    if (frst->player_map[pos] == UNREVEALED) {
      frst->player_map[pos] = MARKED;
    } else if (frst->player_map[pos] == MARKED) {
      frst->player_map[pos] = UNREVEALED;
    } 
  } 
}


// count_surrounding_ghost(frst, x, y) returns the number of ghosts which
//   contained in the surrounding positions of (x, y).
// requires: frst is vaid
static int count_surrounding_ghost(const struct forest *frst, int x, int y) {
  assert(frst);
  int count = 0;
  int pos = transfer_2D_into_1D_pos(frst, x - 1, y - 1);
  if (check_valid_pos(frst, x - 1, y - 1) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  } 
  pos = transfer_2D_into_1D_pos(frst, x - 1, y + 1);
  if (check_valid_pos(frst, x - 1, y + 1) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  } 
  pos = transfer_2D_into_1D_pos(frst, x + 1, y - 1);
  if (check_valid_pos(frst, x + 1, y - 1) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  }
  pos = transfer_2D_into_1D_pos(frst, x + 1, y + 1);
  if (check_valid_pos(frst, x + 1, y + 1) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  }
  pos = transfer_2D_into_1D_pos(frst, x, y - 1);
  if (check_valid_pos(frst, x, y - 1) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  }
  pos = transfer_2D_into_1D_pos(frst, x, y + 1);
  if (check_valid_pos(frst, x, y + 1) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  } 
  pos = transfer_2D_into_1D_pos(frst, x - 1, y);
  if (check_valid_pos(frst, x - 1, y) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  } 
  pos = transfer_2D_into_1D_pos(frst, x + 1, y);
  if (check_valid_pos(frst, x + 1, y) && 
      frst->ghost_map[pos] == ghost_presence) {
    count += 1;
  } 
  return count;
}


// mutate_player_map_pos(frst, x, y) mutates the positions in player_map into
//   the corresponding status
// requires: frst is valid
// effects: modifies *frst
static void mutate_player_map_pos(struct forest *frst, int x, int y) {
  int count = count_surrounding_ghost(frst, x, y);
  int init_pos = transfer_2D_into_1D_pos(frst, x, y);
  if (count == 0) {
    frst->player_map[init_pos] = NO_ADJACENT_GHOST;
  } else {
    frst->player_map[init_pos] = count + TRANSFER_CHAR;
    return;
  }
  
  int pos = transfer_2D_into_1D_pos(frst, x - 1, y - 1);
  if (check_valid_pos(frst, x - 1, y - 1) &&
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x - 1, y - 1);
  }
  pos = transfer_2D_into_1D_pos(frst, x - 1, y + 1);
  if (check_valid_pos(frst, x - 1, y + 1) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x - 1, y + 1);
  }
  pos = transfer_2D_into_1D_pos(frst, x + 1, y - 1);
  if (check_valid_pos(frst, x + 1, y - 1) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x + 1, y - 1);
  }
  pos = transfer_2D_into_1D_pos(frst, x + 1, y + 1);
  if (check_valid_pos(frst, x + 1, y + 1) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x + 1, y + 1);
  }
  pos = transfer_2D_into_1D_pos(frst, x, y - 1);
  if (check_valid_pos(frst, x, y - 1) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x, y - 1);
  }
  pos = transfer_2D_into_1D_pos(frst, x, y + 1);
  if (check_valid_pos(frst, x, y + 1) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x, y + 1);
  }
  pos = transfer_2D_into_1D_pos(frst, x - 1, y);
  if (check_valid_pos(frst, x - 1, y) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x - 1, y);
  }
  pos = transfer_2D_into_1D_pos(frst, x + 1, y);
  if (check_valid_pos(frst, x + 1, y) && 
      frst->player_map[pos] == UNREVEALED) {
    mutate_player_map_pos(frst, x + 1, y);
  }
}

// this is the part you need to complete!!!

// check_win(frst) returns true if the number of ghosts in player_map is
//   the same as the number of ghosts in ghost_map, false otherwise
// requires: frst is valid
static bool check_win(const struct forest *frst) {
  assert(frst);
  int ghost_map_cnt, uncovered_positions;
  for (int i = 0; i< frst->width * frst->height; i++){
    if (frst->player_map[i] == UNREVEALED && frst->ghost_map[i] != ghost_presence) return false;
  }
  return true;
}

      
// forest_reveal(frst, x, y) reveals the forest frst at x- and y-coordinates
//   (x,y). The function returns forest_reveal_alive if there was no ghost at
//   (x,y), forest_reveal_dead if there was, or forest_reveal_win if all
//   ghosts in the forest have been marked correctly.
// If the x or y coordinates are not within the bounds of frst, return INVALID_SYMBOL
// effects: mutates frst
int forest_reveal(struct forest *frst, int x, int y) {
  assert(frst);
  // To complete
  if (check_valid_pos(frst, x, y)){
    if (frst->ghost_map[transfer_2D_into_1D_pos(frst, x, y)] == ghost_presence){
      frst->player_map[transfer_2D_into_1D_pos(frst, x, y)] = ghost_presence;
      return forest_reveal_dead;
    }
    if (frst->ghost_map[transfer_2D_into_1D_pos(frst, x, y)] == ghost_absence){
      mutate_player_map_pos(frst, x, y);
      if (check_win(frst)){
        return forest_reveal_win;
      }
      return forest_reveal_alive;
    }
  }
  return INVALID_SYMBOL;
}



// THIS PART IS GIVEN FOR TESTING AND PLAYING THE GAME do not change

// forest_print(frst) prints the content of the player map to the console.
//   The following symbols are used in the player map:
//   * '_' indicates an unrevealed tile.
//   * '*' indicates a marked tile.
//   * ' ' indicates a revealed tile with no adjacent ghosts.
//   * '1' - '8' indicate a revealed tile with 1 - 8 adjacent ghosts.
//   * 'G' indicates a revealed tile with a ghost.
// effects: writes to input
void forest_print(const struct forest *frst) {
  assert(frst);
  printf("  +");
  for (int i = 0; i < frst->width; i++) {
    printf(" %d", i % find_last_digit);
  }
  printf(" +\n");
  for (int j = 0; j < frst->height; j++) {
    printf("%d | ", j % find_last_digit);
    for (int k = 0; k < frst->width; k++) {
      int idx = transfer_2D_into_1D_pos(frst, k, j);
      printf("%c ", frst->player_map[idx]);
    }
    printf("|\n");
  }
  printf("\n");
}


// Function to look up a symbol and return its code
int lookup_symbol(const char *symbol) {
    // Define an array of symbols and their corresponding codes
    const char *symbols[] = {"MARK", "STEP"};
    const int codes[] = {0, 1};
    const int num_symbols = sizeof(symbols) / sizeof(symbols[0]);

    // Loop through the symbols and find the matching code
    for (int i = 0; i < num_symbols; i++) {
        if (strcmp(symbol, symbols[i]) == 0) {
            return codes[i];
        }
    }

    // If the symbol is not found, return INVALID_SYMBOL
    return INVALID_SYMBOL;
}

// Function to read the next symbol code from standard input
int read_symbol() {
    char symbol[50]; // Assuming symbols won't exceed 49 characters

    if (scanf("%49s", symbol) == 1) {
        int cmd = lookup_symbol(symbol);
        if (cmd != INVALID_SYMBOL) {
            return cmd; // Return the code if symbol is valid
        }
    }

    return INVALID_SYMBOL; // Return INVALID_SYMBOL if symbol is not valid
}


// This part do not change and do not submit
int main(void) {
  const int cmd_mark = lookup_symbol("MARK"); // Given function
  const int cmd_step = lookup_symbol("STEP"); 
  int width = -1;
  int height = -1;
  printf("Enter the width and the height: ");
  assert(scanf("%d", &width) == 1);
  assert(scanf("%d", &height) == 1);
  printf("Enter the map row by row 0/G:\n");
  struct forest frst = forest_enter(width, height); // Given function
  forest_print(&frst);  // Given function
  int cmd = INVALID_SYMBOL;
  while ((cmd = read_symbol()) != INVALID_SYMBOL) {   // Given function
    int x_coord = -1;
    int y_coord = - 1;
    assert(scanf("%d", &x_coord) == 1);
    assert(scanf("%d", &y_coord) == 1);
    if (cmd == cmd_mark) {
      forest_mark(&frst, x_coord, y_coord);  // for you to implement
    } else if (cmd == cmd_step) {
      int result = forest_reveal(&frst, x_coord, y_coord);  // for you to implement
      if (result == forest_reveal_dead) {
        forest_print(&frst);
        printf("BOOOOOOOO! You've encountered a ghost. Game over!\n");
        break;
      } else if (result == forest_reveal_win) {
        forest_print(&frst);
        printf("You've found every ghosts. "
               "The town folks are safe. Well done!\n");
        break;
      }
    } else {
      assert(false);
    }
    forest_print(&frst);
  }
}
