"""Game class — window management, main loop, state routing."""

import sys
import pygame

import config
from . import fonts
from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from .screens.setup import SetupScreen
from .screens.battle import BattleScreen

VW, VH = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
ASPECT = VW / VH

SETUP, BATTLE, GAME_OVER = 0, 1, 2


class Game:
    def __init__(self):
        self.fullscreen = False
        self.win_w, self.win_h = int(VW * 3.5), int(VH * 3.5)
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("HoMM2 Battle AI Demo")
        self.clock = pygame.time.Clock()

        self._rs = self.win_w / VW
        self.canvas = pygame.Surface((self.win_w, self.win_h))
        fonts.init(self._rs)
        self._init_grid()

        self.units = []
        self.state = SETUP

        # screens
        self.screen_setup = SetupScreen(self)
        self.screen_battle = BattleScreen(self)
        self._playagain_rect = None

    # ── scale helpers ───────────────────────────────────────

    def _s(self, v):
        return v * self._rs

    def _init_grid(self):
        self.grid = HexGrid(self._rs)

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
                    pygame.image.save(self.canvas, '/tmp/demo-screenshot.png')
                    print('Screenshot saved to /tmp/demo-screenshot.png')
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
        self.screen_battle.battle = BattleState(self.grid, self.units)
        self.screen_battle.b_log = []
        self.screen_battle._popups = []
        self.screen_battle._round_order = None
        self.screen_battle._order_idx = 0
        self.screen_battle._round_num = 0
        self.screen_battle._ph = 0
        self.state = BATTLE

    def reset(self):
        self.state = SETUP
        self.screen_battle.reset()
        self.units = []
        self._playagain_rect = None
