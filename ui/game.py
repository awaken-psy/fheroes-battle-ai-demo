"""Game class — window management, main loop, state routing."""

import sys
import pygame

import config
from . import fonts
from engine.hex_grid import HexGrid
from .hex_renderer import HexRenderer
from engine.battle_state import BattleState
from engine.hero import Hero
from engine.castle import Castle
from .screens.setup import SetupScreen
from .screens.battle import BattleScreen

VW, VH = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
ASPECT = VW / VH

from config import SETUP, BATTLE, GAME_OVER


class Game:
    def __init__(self):
        self.fullscreen = False
        # Fit initial window to 80 % of the desktop so it never overflows
        info = pygame.display.Info()
        scale = min(info.current_w * 0.8 / VW, info.current_h * 0.8 / VH)
        self.win_w, self.win_h = int(VW * scale), int(VH * scale)
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("HoMM2 Battle AI Demo")
        self.clock = pygame.time.Clock()

        self._rs = self.win_w / VW
        self.canvas = pygame.Surface((self.win_w, self.win_h))
        fonts.init(self._rs)
        self._init_grid()

        self.units = []
        self._siege = False
        self.player_team = 0  # 0=Blue, 1=Red, None=Auto (AI vs AI)
        self.state = SETUP

        # screens
        self.screen_setup = SetupScreen(self)
        self.screen_battle = BattleScreen(self, player_team=self.player_team)
        self._playagain_rect = None

    # ── scale helpers ───────────────────────────────────────

    def _s(self, v):
        return v * self._rs

    def _init_grid(self):
        self.grid = HexGrid()
        self.hex_renderer = HexRenderer(self.grid, self._rs)

    def _rebuild_canvas(self):
        self._rs = self.win_w / VW
        self.canvas = pygame.Surface((self.win_w, self.win_h))
        fonts.init(self._rs)
        self._init_grid()

    # ── window sizing ────────────────────────────────────────

    def _apply_window_size(self):
        self.win_h = max(360, self.win_h)
        self.win_w = max(560, int(self.win_h * ASPECT))
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.win_w, self.win_h = info.current_w, info.current_h
        else:
            self.screen = pygame.display.set_mode(
                (self.win_w, self.win_h), pygame.RESIZABLE)
        self._rebuild_canvas()

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self._apply_window_size()

    # ── main loop ────────────────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(config.FPS) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.win_w, self.win_h = ev.w, ev.h
                    self._apply_window_size()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F12:
                    import tempfile, os
                    shot_path = os.path.join(tempfile.gettempdir(), 'demo-screenshot.png')
                    pygame.image.save(self.canvas, shot_path)
                    print(f'Screenshot saved to {shot_path}')
                self._handle(ev)
            self._update(dt)
            self._draw()
            self.screen.blit(self.canvas, (0, 0))
            pygame.display.flip()

    # ── event routing ────────────────────────────────────────

    def _handle(self, ev):
        if self.state == SETUP:
            self.screen_setup.handle(ev)
        elif self.state == BATTLE:
            self.screen_battle.handle(ev)
        elif self.state == GAME_OVER:
            if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                    and self._playagain_rect
                    and self._playagain_rect.collidepoint(ev.pos)):
                self.reset()

        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.win_h += 60; self._apply_window_size()
            elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.win_h -= 60; self._apply_window_size()
            elif ev.key == pygame.K_F11:
                self._toggle_fullscreen()

    def _update(self, dt):
        if self.state == BATTLE:
            self.screen_battle.update(dt)

    def _draw(self):
        self.canvas.fill(config.BG)
        if self.state == SETUP:
            self.screen_setup.draw()
        elif self.state == BATTLE:
            self.screen_battle.draw()
        elif self.state == GAME_OVER:
            self._playagain_rect = self.screen_battle.draw_gameover()

    # ── game logic ───────────────────────────────────────────

    def start_battle(self):
        # Both sides get a default spellcasting hero so the demo shows spells.
        heroes = {0: Hero(), 1: Hero()}
        castle = Castle() if self._siege else None
        self.screen_battle.player_team = self.player_team
        self.screen_battle.ai_strategy = dict(self.screen_setup.ai_strategy)
        # Full reset to clear any stale state from previous battle
        self.screen_battle.reset()
        self.screen_battle.battle = BattleState(self.grid, self.units, heroes=heroes,
                                                 castle=castle)
        self.screen_battle._round_order = None
        self.screen_battle._order_idx = 0
        self.screen_battle._round_num = 0
        self.screen_battle._ph = 0
        self.screen_battle._await_input = False
        self.screen_battle._cast_mode = False
        self.screen_battle._pending_unit = None
        self.screen_battle._selected_unit = None
        self.screen_battle._legal_mask = None
        self.screen_battle._await_spell_target = False
        self.screen_battle._selected_spell_slot = None
        self.screen_battle._spell_list = []
        self.screen_battle._spell_sel = 0
        self.screen_battle._spell_scroll = 0
        self.screen_battle._actions_remaining = 1
        self.screen_battle._ai_cache = {}
        # Position grid for battle screen before first update runs
        grid_w = self.grid.cols * self.hex_renderer.hex_w
        self.hex_renderer.reposition((self.win_w - grid_w) / 2, self._s(config.GRID_OFFSET_Y))
        self.screen_battle.logger.start(self.units)
        self.state = BATTLE

    def reset(self):
        self.state = SETUP
        self.screen_battle.reset()
        self.units = []
        self._siege = False
        self._playagain_rect = None
