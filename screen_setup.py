"""Setup screen: unit placement, preset loading, team selection."""

import pygame

import config
import fonts
from renderer import draw_btn, draw_unit
from battle import Unit


class SetupScreen:
    """Handles the pre-battle setup phase."""

    def __init__(self, game):
        self.game = game
        self.sel_type: str | None = None
        self.sel_team = 0
        self.hover: tuple | None = None

    # ── layout rects (virtual canvas coords) ──────────────────

    def _palette_rect(self, i):
        s = self.game._s
        return pygame.Rect(s(12), s(74 + i * 56), s(200), s(50))

    def _start_btn_rect(self):
        s = self.game._s
        cx = s(config.WINDOW_WIDTH) // 2
        return pygame.Rect(cx - s(100), s(config.WINDOW_HEIGHT - 55), s(200), s(44))

    def _preset_rect(self, i):
        s = self.game._s
        total = len(config.PRESETS)
        vh = config.WINDOW_HEIGHT
        return pygame.Rect(s(12), s(vh - 100 - 10 - (total - i) * 32), s(200), s(28))

    # ── event handling ────────────────────────────────────────

    def handle(self, ev):
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.game.grid.pixel_to_hex(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            s = self.game._s
            if ev.button == 1:
                for i, name in enumerate(config.UNIT_TYPES):
                    if self._palette_rect(i).collidepoint(mx, my):
                        self.sel_type = name; return
                team_rect = pygame.Rect(int(s(14)), int(s(12)),
                                        int(s(100)), int(s(32)))
                if team_rect.collidepoint(mx, my):
                    self.sel_team = 1 - self.sel_team; return
                for i, pname in enumerate(config.PRESETS):
                    if self._preset_rect(i).collidepoint(mx, my):
                        self._load_preset(pname); return
                if self._start_btn_rect().collidepoint(mx, my):
                    if self._can_start():
                        self.game.start_battle()
                    return
                hex_pos = self.game.grid.pixel_to_hex(mx, my)
                if hex_pos and self.sel_type:
                    col, row = hex_pos
                    if self.game.grid.half_of(col) == self.sel_team:
                        self.game.units = [u for u in self.game.units if u.pos != hex_pos]
                        self.game.units.append(
                            Unit.from_type(self.sel_type, self.sel_team, col, row))
            elif ev.button == 3:
                hex_pos = self.game.grid.pixel_to_hex(mx, my)
                if hex_pos:
                    self.game.units = [u for u in self.game.units if u.pos != hex_pos]

    # ── drawing ───────────────────────────────────────────────

    def draw(self):
        g = self.game
        s = g._s
        canvas = g.canvas

        # Reposition grid to centre in available space after left panel
        panel_right = s(228)
        grid_w = g.grid.cols * g.grid.hex_w
        avail = g.win_w - panel_right
        g.grid.reposition(panel_right + (avail - grid_w) / 2, s(config.GRID_OFFSET_Y))

        # ── left panel background ──
        panel = pygame.Rect(int(s(4)), int(s(4)), int(s(216)),
                            int(s(config.WINDOW_HEIGHT) - s(8)))
        pygame.draw.rect(canvas, config.PANEL_BG, panel, border_radius=int(s(6)))
        pygame.draw.rect(canvas, (55, 65, 90), panel, 1, border_radius=int(s(6)))

        # team selector
        draw_btn(canvas, s(14), s(12), s(100), s(32),
                 f"Team: {fonts.team_name(self.sel_team)}",
                 fonts.team_color(self.sel_team), config.WHITE)
        hint = fonts.BODY.render("Click palette -> hex. Right-click remove.",
                                 True, (170, 180, 200))
        canvas.blit(hint, (s(230), s(16)))

        # UNITS section
        canvas.blit(fonts.TITLE.render("UNITS", True, config.WHITE), (s(18), s(52)))

        for i, name in enumerate(config.UNIT_TYPES):
            r = self._palette_rect(i)
            sel = self.sel_type == name
            bg = (50, 55, 78) if sel else (38, 45, 65)
            pygame.draw.rect(canvas, bg, r, border_radius=int(s(4)))
            if sel:
                pygame.draw.rect(canvas, config.YELLOW, r, 2, border_radius=int(s(4)))
            else:
                pygame.draw.rect(canvas, (60, 68, 92), r, 1, border_radius=int(s(4)))
            ut = config.UNIT_TYPES[name]
            tc = fonts.team_light(self.sel_team)
            canvas.blit(fonts.LABEL.render(ut["symbol"], True, tc),
                        (r.x + s(8), r.y + s(5)))
            canvas.blit(fonts.BODY.render(name, True, config.WHITE),
                        (r.x + s(28), r.y + s(4)))
            canvas.blit(fonts.DATA.render(
                f"A{ut['attack']} D{ut['defense']} H{ut['hp']} S{ut['speed']} x{ut['count']}",
                True, (170, 180, 200)), (r.x + s(28), r.y + s(24)))

        # separator
        sep_y = self._preset_rect(0).y - s(8)
        pygame.draw.line(canvas, (55, 65, 90), (s(14), sep_y), (s(210), sep_y), 1)
        canvas.blit(fonts.LABEL.render("PRESETS", True, config.WHITE),
                    (s(18), sep_y - s(20)))

        # hex grid
        highlights = {}
        if self.hover and self.sel_type:
            if g.grid.half_of(self.hover[0]) == self.sel_team:
                highlights[self.hover] = fonts.team_color(self.sel_team)
        g.grid.draw_grid(canvas, highlights)
        self._draw_units()

        # start button
        sr = self._start_btn_rect()
        can = self._can_start()
        draw_btn(canvas, sr.x, sr.y, sr.w, sr.h, "Start Battle",
                 config.GREEN if can else config.GRAY, config.BLACK)

        for i, pname in enumerate(config.PRESETS):
            r = self._preset_rect(i)
            pygame.draw.rect(canvas, config.DARK, r, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (60, 68, 92), r, 1, border_radius=int(s(3)))
            canvas.blit(fonts.BODY.render(f"Preset: {pname}", True, config.WHITE),
                        (r.x + s(8), r.y + s(5)))

        # team unit counts above grid
        for team in (0, 1):
            n = sum(1 for u in g.units if u.team == team)
            txt = fonts.BODY.render(f"{fonts.team_name(team)}: {n} units",
                                    True, fonts.team_light(team))
            if team == 0:
                canvas.blit(txt, (int(g.grid.ox) + int(s(10)), int(s(56))))
            else:
                canvas.blit(txt, (int(g.grid.ox) + int(grid_w)
                                  - txt.get_width() - int(s(10)), int(s(56))))

    def _draw_units(self):
        g = self.game
        for u in g.units:
            if not u.is_alive:
                continue
            cx, cy = g.grid.center(*u.pos)
            draw_unit(g.canvas, g._s, g.grid, u, cx, cy)

    # ── game logic helpers (delegated from Game) ──────────────

    def _can_start(self):
        return (any(u.team == 0 for u in self.game.units)
                and any(u.team == 1 for u in self.game.units))

    def _load_preset(self, name):
        preset = config.PRESETS[name]
        self.game.units = []
        for team, placements in preset.items():
            for type_name, col, row in placements:
                self.game.units.append(Unit.from_type(type_name, team, col, row))
