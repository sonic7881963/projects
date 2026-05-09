# Reversi

A C++ implementation of Reversi/Othello using a grid model, cell state, text display, and observer-style display updates. The game accepts text commands to initialize the board and play moves.

## What This Demonstrates

- Rule-based board game implementation.
- Grid and cell state management.
- Turn handling for black and white players.
- Text-based display updates.
- Defensive command handling for invalid board sizes and illegal moves.

## Key Files

| File | Purpose |
| --- | --- |
| `main.cc` | Command loop for starting games and playing moves. |
| `grid.*` | Board initialization, move validation, scoring, and game-state logic. |
| `cell.*` | Individual cell state and neighbor relationships. |
| `textdisplay.*` | Text rendering for the board. |
| `observer.h`, `subject.h` | Observer pattern support for display updates. |
| `state.h`, `info.h` | Shared state and cell information types. |

## Build

```bash
make
```

## Usage

Start a new game by providing an even board size of at least 4:

```text
./a4q3
new 8
```

Play a move by row and column:

```text
play 2 3
```

The game alternates turns after successful moves and reports the winner when the board is full.
