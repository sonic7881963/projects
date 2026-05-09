# Tower

A Cocos Creator 3.4.0 tower-defense style game project. The project uses TypeScript scripts, prefabs, animations, sprites, and scene assets to coordinate hero spawning, enemy movement, collision-based combat, rewards, and win-state behavior.

## What This Demonstrates

- Cocos Creator 3.x project structure.
- TypeScript gameplay scripting with component lifecycle methods.
- Runtime prefab instantiation for heroes and enemies.
- Collision detection and combat resolution with Cocos 2D physics.
- UI state updates for coins, experience, unit availability, and win conditions.

## Tech Stack

Cocos Creator 3.4.0, TypeScript, Cocos 2D physics, prefabs, animation clips, scene assets.

## Key Files

| Path | Purpose |
| --- | --- |
| `package.json` | Cocos Creator project metadata and editor version. |
| `assets/typescript/mainScene.ts` | Main scene controller for spawning units, updating UI, and handling win/close interactions. |
| `assets/typescript/soldier.ts` | Unit movement, collision, combat, death, reward, and animation behavior. |
| `assets/typescript/gloabl.ts` | Shared game state such as coin and speed values. |
| `assets/prefab/` | Hero and soldier prefabs. |
| `assets/ani/` | Movement, fight, and death animations. |
| `assets/scene/` | Cocos scene files. |

## How To Open

Open this folder in Cocos Creator 3.4.0 or a compatible Cocos Creator 3.x editor. Keep `.meta` files with their matching assets so the editor can resolve UUID references.

## Notes

The project is best reviewed as a game-engine workspace. The most relevant source code is in `assets/typescript/`.
