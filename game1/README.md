# CC3K Dungeon Game

A C++ object-oriented dungeon crawler based on the CC3K project structure. The game models a grid-based dungeon with player movement, enemies, items, attacks, potions, gold, level transitions, and restart/quit commands.

## What This Demonstrates

- Object-oriented game architecture in C++.
- Entity modeling for players, enemies, items, stairs, cells, and floors.
- Turn-based gameplay flow with movement, item use, combat, enemy actions, and game-over handling.
- Separation of responsibilities across factories, game state, board state, characters, and interaction logic.

## Key Files

| Path | Purpose |
| --- | --- |
| `cc3k/main.cc` | Main game loop and command handling. |
| `cc3k/game.*` | Game lifecycle, round updates, restart logic, and level state. |
| `cc3k/player.*` | Player state and actions. |
| `cc3k/enemy.*` | Enemy behavior and interaction logic. |
| `cc3k/floor.*`, `cc3k/cell.*` | Dungeon board representation. |
| `cc3k/board.txt` | Board layout data used by the executable. |

## Controls

The main loop accepts text commands such as movement directions (`no`, `so`, `ea`, `we`, `ne`, `nw`, `se`, `sw`), item use (`u <direction>`), attack (`a <direction>`), restart (`r`), quit (`q`), and a test command used during development.

## Build

From the `cc3k` directory:

```bash
make
```

Then run the generated executable from the same directory so `board.txt` can be loaded correctly.

## Notes

This project is useful as a software design example because the core difficulty is coordinating many interacting game entities while keeping responsibilities understandable.
