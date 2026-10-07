import pygame
import random
import math
import sys
import time

WIDTH, HEIGHT = 800, 600
TILE_SIZE = 32
COLS, ROWS = WIDTH // TILE_SIZE, HEIGHT // TILE_SIZE
NUM_MINES = 50
VISION_RADIUS = 2

FPS = 60

C_BG = (10, 12, 10)
C_UNREVEALED_BORDER = (25, 30, 25)
C_REVEALED_BORDER = (50, 45, 35)

C_FLAG = (180, 40, 40)
C_BLOOD = (130, 10, 10)
C_GHOST_OBSERVE = (150, 150, 170, 100)
C_GHOST_CHASE = (200, 200, 220, 180)
C_GHOST_ANGRY = (220, 50, 50, 200)

TERRAIN_COLORS = {
    'grass': (40, 55, 35),
    'dirt': (55, 48, 38),
    'mud': (35, 30, 25),
    'mud_rev': (75, 68, 55),
    'grass_rev': (80, 95, 75),
    'dirt_rev': (95, 88, 75)
}

NUM_COLORS = {
    1: (100, 150, 240),
    2: (100, 200, 100),
    3: (240, 90, 90),
    4: (180, 70, 180),
    5: (180, 70, 70),
    6: (70, 180, 180),
    7: (220, 220, 220),
    8: (120, 120, 120)
}

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CARCOSA: Field of Echoes")
clock = pygame.time.Clock()

font_numbers = pygame.font.SysFont('courier', 20, bold=True)
font_ui = pygame.font.SysFont('courier', 16)
font_dialogue = pygame.font.SysFont('courier', 17, bold=True)
font_title = pygame.font.SysFont('courier', 42, bold=True)

INTRO_DIALOGUE = [
    ("COMMANDER", "Peace envoy, your task is clear. Cross the zero-line to the neighboring state."),
    ("COMMANDER", "A ceasefire was signed, but the retreat was messy. The battlefield remains rigged."),
    ("ENVOY", "Understood. The peace treaty rests in my hands. I will navigate the field."),
    ("COMMANDER", "Be cautious... something strange haunts those trenches. Do not lose your mind out there.")
]

def make_player_sprite():
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    pygame.draw.ellipse(surf, (0, 0, 0, 80), (6, 22, 20, 8))
    pygame.draw.rect(surf, (45, 60, 45), (10, 16, 5, 12))
    pygame.draw.rect(surf, (45, 60, 45), (17, 16, 5, 12))
    pygame.draw.rect(surf, (20, 25, 20), (9, 26, 6, 4))
    pygame.draw.rect(surf, (20, 25, 20), (17, 26, 6, 4))
    pygame.draw.rect(surf, (70, 95, 65), (8, 10, 16, 10))
    pygame.draw.rect(surf, (50, 70, 45), (10, 12, 12, 7))
    pygame.draw.ellipse(surf, (220, 180, 140), (10, 5, 12, 9))
    pygame.draw.rect(surf, (50, 75, 45), (8, 3, 16, 6))
    pygame.draw.rect(surf, (35, 55, 30), (7, 6, 18, 3))
    pygame.draw.rect(surf, (30, 30, 30), (12, 8, 2, 2))
    pygame.draw.rect(surf, (30, 30, 30), (18, 8, 2, 2))
    return surf

PLAYER_SPRITE = make_player_sprite()

class Tile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_mine = False
        self.is_revealed = False
        self.is_flagged = False
        self.adj_mines = 0
        self.has_blood = False
        self.terrain = "grass"

class Game:
    def __init__(self):
        self.death_count = 0
        self.state = "MENU"
        self.message = ""
        self.message_timer = 0
        self.dialogue_idx = 0
        self.reset_board()
        self.jumpscare_timer = 0

    def generate_terrain(self):
        for y in range(ROWS):
            for x in range(COLS):
                v = math.sin(x * 0.3) + math.cos(y * 0.3) + random.uniform(-0.3, 0.3)
                if v > 0.6:
                    self.grid[y][x].terrain = "mud"
                elif v < -0.2:
                    self.grid[y][x].terrain = "dirt"
                else:
                    self.grid[y][x].terrain = "grass"

    def reset_board(self):
        self.grid = [[Tile(x, y) for x in range(COLS)] for y in range(ROWS)]
        self.generate_terrain()
        self.player_x = COLS // 2
        self.player_y = ROWS // 2
        self.first_move = True
        self.game_over_timer = 0
        self.move_cooldown = 0
        
        self.ghost_active = False
        self.ghost_x = -10
        self.ghost_y = -10
        self.ghost_timer = time.time()
        self.ghost_step_delay = max(0.2, 1.0 - (self.death_count * 0.15))
        self.ghost_last_move = time.time()
        self.ghost_state = "HIDDEN"
        
        self.grid[self.player_y][self.player_x].is_revealed = True

    def get_phase_message(self):
        if self.death_count == 0: return "Navigate the remnants. Right-Click to flag."
        elif self.death_count == 1: return "You feel you are being watched..."
        elif self.death_count == 2: return "Carcosa has taken form."
        else: return "IT HUNGERS."
    def get_nearby_mines(self, target_x, target_y, radius=3):
        mines = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = target_x + dx, target_y + dy
                if 0 <= nx < COLS and 0 <= ny < ROWS:
                    tile = self.grid[ny][nx]
                    # We only care about unrevealed mines (flagged or unflagged)
                    if tile.is_mine and not tile.is_revealed:
                        mines.append((nx, ny))
        return mines
    def start_intro(self):
        self.dialogue_idx = 0
        self.state = "DIALOGUE"

    def place_mines(self, safe_x, safe_y):
        safe_zone = [(safe_x + dx, safe_y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        
        mines_placed = 0
        while mines_placed < NUM_MINES:
            rx = random.randint(0, COLS - 1)
            ry = random.randint(0, ROWS - 1)
            if (rx, ry) not in safe_zone and not self.grid[ry][rx].is_mine:
                self.grid[ry][rx].is_mine = True
                mines_placed += 1
                
        for y in range(ROWS):
            for x in range(COLS):
                if not self.grid[y][x].is_mine:
                    count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < COLS and 0 <= ny < ROWS and self.grid[ny][nx].is_mine:
                                count += 1
                    self.grid[y][x].adj_mines = count

    def reveal_zeros(self, sx, sy):
        queue = [(sx, sy)]
        while queue:
            cx, cy = queue.pop(0)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < COLS and 0 <= ny < ROWS:
                        tile = self.grid[ny][nx]
                        if not tile.is_revealed and not tile.is_flagged:
                            tile.is_revealed = True
                            if tile.adj_mines == 0 and not tile.is_mine:
                                queue.append((nx, ny))

    def move_player(self, dx, dy):
        if self.state != "PLAYING": return
        
        nx, ny = self.player_x + dx, self.player_y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS:
            self.player_x, self.player_y = nx, ny
            tile = self.grid[ny][nx]
            
            if self.first_move:
                self.place_mines(nx, ny)
                self.first_move = False
                
            if not tile.is_flagged:
                if not tile.is_revealed:
                    tile.is_revealed = True
                    if tile.is_mine:
                        self.trigger_death(nx, ny)
                    elif tile.adj_mines == 0:
                        self.reveal_zeros(nx, ny)
            
            self.check_victory()

    def toggle_flag(self, mx, my):
        if self.state != "PLAYING": return
        gx, gy = mx // TILE_SIZE, my // TILE_SIZE
        if 0 <= gx < COLS and 0 <= gy < ROWS:
            tile = self.grid[gy][gx]
            if not tile.is_revealed:
                tile.is_flagged = not tile.is_flagged


    def check_victory(self):
        if self.state != "PLAYING": return
        for y in range(ROWS):
            for x in range(COLS):
                tile = self.grid[y][x]
                if not tile.is_mine and not tile.is_revealed:
                    return
        self.state = "VICTORY"
        self.set_message("YOU SURVIVED... TREATY DELIVERED.", 5)
        self.game_over_timer = time.time()

    def set_message(self, text, duration):
        self.message = text
        self.message_timer = time.time() + duration

    def update_carcosa(self):
        if self.state != "PLAYING" or self.first_move:
            return

        now = time.time()

        if self.death_count == 0:
            if now - self.ghost_timer > 5:
                self.ghost_timer = now
                self.ghost_x, self.ghost_y = random.choice([0, COLS - 1]), random.randint(0, ROWS - 1)
                self.ghost_state = "OBSERVING"
            elif now - self.ghost_timer > 1.5:
                self.ghost_state = "HIDDEN"
        else:
            if self.ghost_state == "HIDDEN":
                if now - self.ghost_timer > 3:
                    self.ghost_timer = now
                    gx, gy = random.randint(0, COLS - 1), random.randint(0, ROWS - 1)
                    if abs(gx - self.player_x) > 5:
                        self.ghost_x, self.ghost_y = gx, gy
                        self.ghost_state = "CHASING" if self.death_count == 1 else "HERDING"

            if self.ghost_state in ("CHASING", "HERDING") and now - self.ghost_last_move > self.ghost_step_delay:
                self.ghost_last_move = now
                target_x, target_y = self.player_x, self.player_y

                if self.ghost_state == "HERDING":
                    nearby_mines = self.get_nearby_mines(self.player_x, self.player_y, radius=5)
                    if nearby_mines:
                        avg_mx = sum(m[0] for m in nearby_mines) / len(nearby_mines)
                        avg_my = sum(m[1] for m in nearby_mines) / len(nearby_mines)
                        push_dir_x = self.player_x - avg_mx
                        push_dir_y = self.player_y - avg_my
                        target_x = max(0, min(COLS - 1, int(self.player_x + (push_dir_x * 2))))
                        target_y = max(0, min(ROWS - 1, int(self.player_y + (push_dir_y * 2))))

                if self.ghost_x < target_x:
                    self.ghost_x += 1
                elif self.ghost_x > target_x:
                    self.ghost_x -= 1
                elif self.ghost_y < target_y:
                    self.ghost_y += 1
                elif self.ghost_y > target_y:
                    self.ghost_y -= 1

                if self.ghost_x == self.player_x and self.ghost_y == self.player_y:
                    self.trigger_death(self.player_x, self.player_y, caught_by_ghost=True)

    def trigger_death(self, x, y, caught_by_ghost=False):
        if self.state != "PLAYING":
            return

        self.death_count += 1
        self.ghost_state = "HIDDEN"
        self.ghost_timer = time.time()
        self.ghost_x = x
        self.ghost_y = y
        self.ghost_step_delay = max(0.2, 1.0 - (self.death_count * 0.15))
        self.grid[y][x].has_blood = True
        self.set_message("THE TRENCHES REMEMBER...", 2.5 if not caught_by_ghost else 3.0)
        self.jumpscare_timer = time.time()
        self.state = "JUMPSCARE"

    def draw_ghost(self, screen):
        if self.ghost_state == "HIDDEN":
            return

        ghost_rect = pygame.Rect(
            self.ghost_x * TILE_SIZE,
            self.ghost_y * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )

        if self.ghost_state == "OBSERVING":
            ghost_color = C_GHOST_OBSERVE
        elif self.ghost_state == "HERDING":
            ghost_color = C_GHOST_ANGRY
        else:
            ghost_color = C_GHOST_CHASE

        ghost_surface = pygame.Surface(ghost_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(ghost_surface, ghost_color, (6, 3, 20, 26))
        pygame.draw.circle(ghost_surface, (15, 15, 15, 220), (12, 12), 2)
        pygame.draw.circle(ghost_surface, (15, 15, 15, 220), (20, 12), 2)
        screen.blit(ghost_surface, ghost_rect)

    def draw_world(self, screen):
        screen.fill(C_BG)

        for y in range(ROWS):
            for x in range(COLS):
                tile = self.grid[y][x]
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

                if tile.is_revealed:
                    base_col = C_BLOOD if tile.has_blood else TERRAIN_COLORS[tile.terrain + "_rev"]
                    pygame.draw.rect(screen, base_col, rect)
                    pygame.draw.rect(screen, C_REVEALED_BORDER, rect, 1)

                    if tile.is_mine:
                        pygame.draw.circle(screen, (20, 20, 20), rect.center, TILE_SIZE // 3)
                        pygame.draw.circle(screen, (90, 90, 90), (rect.centerx - 4, rect.centery - 4), 3)
                    elif tile.adj_mines > 0:
                        distance = max(abs(x - self.player_x), abs(y - self.player_y))
                        if distance <= VISION_RADIUS:
                            number = font_numbers.render(str(tile.adj_mines), True, NUM_COLORS[tile.adj_mines])
                            screen.blit(number, number.get_rect(center=rect.center))
                else:
                    pygame.draw.rect(screen, TERRAIN_COLORS[tile.terrain], rect)
                    pygame.draw.rect(screen, C_UNREVEALED_BORDER, rect, 1)
                    if tile.is_flagged:
                        flag_rect = rect.inflate(-12, -12)
                        pygame.draw.polygon(
                            screen,
                            C_FLAG,
                            [flag_rect.midtop, flag_rect.bottomright, flag_rect.bottomleft]
                        )

        self.draw_ghost(screen)

        if self.player_x >= 0 and self.player_y >= 0:
            screen.blit(PLAYER_SPRITE, (self.player_x * TILE_SIZE, self.player_y * TILE_SIZE))

        status = font_ui.render(self.get_phase_message(), True, (190, 180, 150))
        screen.blit(status, (10, HEIGHT - 24))

        if self.message and time.time() < self.message_timer:
            message = font_dialogue.render(self.message, True, (220, 80, 80))
            screen.blit(message, message.get_rect(center=(WIDTH // 2, 16)))

        if self.state == "GAMEOVER":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            title = font_title.render("MISSION LOST", True, (220, 60, 60))
            prompt = font_ui.render("Press R to try again or ESC for menu", True, (220, 220, 220))
            screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 35)))
            screen.blit(prompt, prompt.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 25)))
        elif self.state == "VICTORY":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 130))
            screen.blit(overlay, (0, 0))
            title = font_title.render("TREATY DELIVERED", True, (160, 220, 160))
            prompt = font_ui.render("Press R for another mission or ESC for menu", True, (220, 220, 220))
            screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 35)))
            screen.blit(prompt, prompt.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 25)))
        elif self.state == "JUMPSCARE":
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((10, 0, 0, 170))
            screen.blit(overlay, (0, 0))
            flicker = 200 + int(50 * math.sin(time.time() * 30))
            title = font_title.render("YOU SAW IT", True, (flicker, 40, 40))
            screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    def draw_menu(self, screen):
        screen.fill((8, 10, 8))
        title = font_title.render("CARCOSA", True, (220, 200, 160))
        subtitle = font_ui.render("FIELD OF ECHOES", True, (170, 170, 170))
        screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 80)))
        screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 35)))

        info = [
            "Press SPACE to begin the mission",
            "Press H for field guide",
            "WASD / Arrows move, Right click to flag",
        ]
        for idx, line in enumerate(info):
            surf = font_ui.render(line, True, (220, 220, 220))
            screen.blit(surf, surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30 + idx * 28)))

    def draw_guide(self, screen):
        screen.fill((10, 10, 12))
        title = font_title.render("FIELD GUIDE", True, (220, 210, 160))
        screen.blit(title, title.get_rect(center=(WIDTH // 2, 80)))

        lines = [
            "- Reveal every safe tile without stepping on a mine to deliver the treaty.",
            "- Use WASD or arrow keys to move one tile at a time.",
            "- Right-click to place or remove a flag.",
            "- Numbers show nearby mines only within your vision radius.",
            "- A ghost watches the trenches. If it catches you, press R to retry.",
            "- Press SPACE or ESC to return to the menu.",
        ]
        for idx, line in enumerate(lines):
            surf = font_ui.render(line, True, (210, 210, 210))
            screen.blit(surf, (80, 150 + idx * 34))

    def draw_dialogue(self, screen):
        screen.fill((5, 8, 5))
        pygame.draw.rect(screen, (20, 25, 20), (40, HEIGHT - 180, WIDTH - 80, 140))
        pygame.draw.rect(screen, (80, 100, 80), (40, HEIGHT - 180, WIDTH - 80, 140), 2)

        speaker, text = INTRO_DIALOGUE[self.dialogue_idx]
        spk_surf = font_dialogue.render(f"[{speaker}]", True, (220, 80, 80) if speaker == "COMMANDER" else (100, 200, 100))
        screen.blit(spk_surf, (60, HEIGHT - 165))

        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + (1 if current else 0) > 52:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)

        for idx, line in enumerate(lines[:4]):
            surf = font_ui.render(line, True, (220, 220, 220))
            screen.blit(surf, (60, HEIGHT - 130 + idx * 22))

        prompt = font_ui.render("Press SPACE to continue", True, (180, 180, 180))
        screen.blit(prompt, (WIDTH - 240, HEIGHT - 38))

    def draw(self, screen):
        if self.state == "MENU":
            self.draw_menu(screen)
        elif self.state == "GUIDE":
            self.draw_guide(screen)
        elif self.state == "DIALOGUE":
            self.draw_dialogue(screen)
        elif self.state in ("PLAYING", "GAMEOVER", "VICTORY", "JUMPSCARE"):
            self.draw_world(screen)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE and self.state not in ("MENU", "GUIDE"):
                self.state = "MENU"
            elif self.state == "MENU":
                if event.key == pygame.K_SPACE:
                    self.start_intro()
                elif event.key == pygame.K_h:
                    self.state = "GUIDE"
            elif self.state == "GUIDE":
                if event.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                    self.state = "MENU"
            elif self.state == "DIALOGUE":
                if event.key == pygame.K_SPACE:
                    self.dialogue_idx += 1
                    if self.dialogue_idx >= len(INTRO_DIALOGUE):
                        self.state = "PLAYING"
            elif self.state == "PLAYING":
                directions = {
                    pygame.K_w: (0, -1), pygame.K_UP: (0, -1),
                    pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
                    pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0),
                    pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0),
                }
                if event.key in directions:
                    self.move_player(*directions[event.key])
            elif self.state in ("GAMEOVER", "VICTORY"):
                if event.key == pygame.K_r:
                    self.reset_board()
                    self.state = "PLAYING"
                elif event.key == pygame.K_ESCAPE:
                    self.state = "MENU"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self.toggle_flag(*event.pos)

    def update(self):
        if self.state == "JUMPSCARE":
            if time.time() - self.jumpscare_timer > 1.2:
                self.state = "GAMEOVER"
        self.update_carcosa()


def main():
    game = Game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(event)
        game.update()
        game.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
