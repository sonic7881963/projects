# Software Engineering Projects

This repository is a curated portfolio of independent software and game-development projects. Each project is organized in its own directory to keep the codebase easy to inspect, maintain, and extend.

The collection demonstrates practical SDE competencies across application architecture, mobile development, backend service design, image processing, algorithmic game logic, TypeScript scripting, asset-driven development, and maintainable repository organization.

## Project Index

| Project | Description | Engineering Focus |
| --- | --- | --- |
| `game1` | Game project workspace. | Gameplay implementation, project organization, iterative development. |
| `imageProcessor` | Image processing project. | File handling, data transformation, processing pipeline design. |
| `microservice` | Microservice-oriented project. | Backend service structure, modular design, API-oriented architecture. |
| `Reversi` | Reversi / Othello game project. | Rule-based game logic, board-state management, algorithmic thinking. |
| `TowerDefense` | Tower defense game project. | Real-time gameplay systems, enemy waves, defense mechanics, state management. |
| `tower` | Cocos Creator 3.4.0 project with game assets, scripts, engine configuration, and generated resources. | TypeScript-based game scripting, Cocos Creator workflow, scene/resource management. |
| `QuanMinZhuGong` | Cocos Creator mini-game project with characters, levels, combat scenes, image assets, and TypeScript scripts. | Level scripting, component-based game development, asset integration. |
| `SJ2215 Text Life Mini Game` | Text-life mini-game asset project with preview screens, luck selection, initial attributes, and life-event UI resources. | UI asset organization, game flow design, content-driven interaction structure. |
| `Stock Trading Simulator` | Android mobile stock-trading simulator where users can practice portfolio management in sandbox mode or compete in multiplayer trading challenges. | Android development, Gradle project structure, mobile UI flows, financial simulation logic, gamified user experience. |

## Repository Structure

Each top-level folder is treated as an independent project. This layout keeps unrelated experiments separated while still making the repository useful as a single engineering portfolio.

```text
projects/
  game1/
  imageProcessor/
  microservice/
  Reversi/
  TowerDefense/
  tower/
  QuanMinZhuGong/
  SJ2215 Text Life Mini Game/
  Stock Trading Simulator/
```

## SDE-Relevant Highlights

- Modular organization: projects are separated by responsibility and can be reviewed independently.
- Multiple domains: the repository includes backend-oriented, mobile, image-processing, algorithmic game, and game-engine projects.
- Practical implementation: several projects include real assets, scripts, configuration files, and development-environment structures from working applications.
- Maintainability: the repository-level README provides a clear project map for reviewers, collaborators, and future development.
- Product thinking: the projects cover user-facing workflows such as mobile onboarding, trading simulation, game progression, and interactive UI states.

## Notes

Some game projects include editor-generated files and resource directories. For the best development experience, open Cocos Creator projects with the matching editor version listed in their project configuration. The stock trading simulator is an Android/Gradle project and should be opened with Android Studio or another compatible Android development environment.
