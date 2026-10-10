
"""Watch a trained agent or play Snake yourself."""

import argparse
import os
import pygame

from config import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, FPS, MODELS_DIR, SEED
from snake_env import SnakeEnv
from agents.tabular import TabularAgent
from agents.deep import DQNAgent


# Assumed direction order: Up=0, Right=1, Down=2, Left=3
KEY_DIRECTIONS = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3,
    pygame.K_w: 0,
    pygame.K_d: 1,
    pygame.K_s: 2,
    pygame.K_a: 3,
}


def load_agent(name):
    if name in ("q_learning", "sarsa"):
        agent = TabularAgent(name, seed=SEED)
        path = os.path.join(MODELS_DIR, name + ".npz")
    else:
        agent = DQNAgent(name, seed=SEED)
        path = os.path.join(MODELS_DIR, name + ".pt")

    if not os.path.exists(path):
        raise FileNotFoundError(
            path + " missing; run python train.py first."
        )

    agent.load(path)
    agent.epsilon = 0.0
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["human", "q_learning", "sarsa", "dqn", "double_dqn"],
        default="human",
    )
    args = parser.parse_args()

    pygame.init()

    width = GRID_WIDTH * CELL_SIZE
    height = GRID_HEIGHT * CELL_SIZE
    panel_height = 75

    screen = pygame.display.set_mode((width, height + panel_height))
    pygame.display.set_caption("Snake RL: " + args.agent)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 42)

    env = SnakeEnv(seed=SEED)
    state = env.reset()

    agent = None if args.agent == "human" else load_agent(args.agent)

    paused = False
    started = agent is not None
    human_action = 0

    move_delay = 150  # Human movement interval in milliseconds
    last_move = pygame.time.get_ticks()

    running = True

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    state = env.reset()
                    paused = False
                    started = agent is not None
                    human_action = 0
                    last_move = now

                elif event.key == pygame.K_SPACE:
                    if started and not env.done:
                        paused = not paused
                        last_move = now

                elif agent is None and not env.done:
                    desired = KEY_DIRECTIONS.get(event.key)

                    if desired is not None:
                        current = env.direction

                        # First keypress selects the starting direction.
                        if not started:
                            if desired == current:
                                human_action = 0
                                started = True
                            elif desired == (current + 1) % 4:
                                human_action = 1
                                started = True
                            elif desired == (current - 1) % 4:
                                human_action = 2
                                started = True

                            last_move = now

                        else:
                            # Prevent a 180-degree reversal.
                            if desired == (current + 2) % 4:
                                continue

                            if desired == current:
                                human_action = 0
                            elif desired == (current + 1) % 4:
                                human_action = 1
                            else:
                                human_action = 2

        # Update the game.
        if started and not paused and not env.done:
            if agent is None:
                delay = move_delay
            else:
                delay = max(40, int(1000 / max(FPS, 1)))

            if now - last_move >= delay:
                if agent is None:
                    action = human_action
                    human_action = 0
                elif args.agent in ("q_learning", "sarsa"):
                    action = agent.act(env.state_key(state), False)
                else:
                    action = agent.act(state, False)

                state, _, _, _ = env.step(action)
                last_move = now

        # Draw the background.
        screen.fill((18, 22, 30))

        # Draw the grid.
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                pygame.draw.rect(
                    screen,
                    (34, 40, 50),
                    (
                        x * CELL_SIZE,
                        y * CELL_SIZE,
                        CELL_SIZE - 1,
                        CELL_SIZE - 1,
                    ),
                )

        # Draw the snake.
        for i, (x, y) in enumerate(env.snake):
            color = (
                (76, 220, 120)
                if i == 0
                else (38, 165, 90)
            )

            pygame.draw.rect(
                screen,
                color,
                (
                    x * CELL_SIZE + 2,
                    y * CELL_SIZE + 2,
                    CELL_SIZE - 4,
                    CELL_SIZE - 4,
                ),
                border_radius=5,
            )

        # Draw the food.
        if env.food is not None:
            x, y = env.food
            pygame.draw.circle(
                screen,
                (245, 85, 85),
                (
                    x * CELL_SIZE + CELL_SIZE // 2,
                    y * CELL_SIZE + CELL_SIZE // 2,
                ),
                max(3, CELL_SIZE // 3),
            )

        # Draw the information panel.
        panel_y = height
        pygame.draw.rect(
            screen,
            (25, 30, 40),
            (0, panel_y, width, panel_height),
        )

        score_text = font.render(
            f"Score: {env.score} | Mode: {args.agent}",
            True,
            (245, 245, 245),
        )
        screen.blit(score_text, (10, panel_y + 7))

        if env.done:
            status = "GAME OVER - Press R to restart"
        elif paused:
            status = "PAUSED - Press SPACE to resume"
        elif not started:
            status = "Press an arrow key or WASD to start"
        elif agent is None:
            status = "Arrows/WASD: Move | SPACE: Pause | R: Restart"
        else:
            status = "SPACE: Pause | R: Restart | ESC: Quit"

        status_color = (
            (245, 100, 100)
            if env.done
            else (220, 225, 235)
        )

        status_text = font.render(status, True, status_color)
        screen.blit(status_text, (10, panel_y + 39))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
