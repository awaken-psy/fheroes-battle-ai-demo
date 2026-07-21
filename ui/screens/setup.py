"""Setup screen: unit placement, preset loading, team selection."""

import pygame

import config
from .. import fonts
from ..renderer import draw_btn, draw_unit
from engine.unit import Unit


class SetupScreen:
    """Handles the pre-battle setup phase."""

    # Layout constants (in design pixels, before scaling)
    PANEL_X = 4
    PANEL_W = 216
    HEADER_Y = 48        # "UNITS" label
    UNIT_AREA_Y = 64     # unit list starts here
    UNIT_ROW_H = 50
    UNIT_ROW_GAP = 4
    UNIT_VISIBLE = 6     # visible unit rows
    PRESET_HEADER_Y = 430  # "PRESETS" label
    PRESET_AREA_Y = 446   # preset list starts here
    PRESET_ROW_H = 28
    PRESET_ROW_GAP = 3
    PRESET_VISIBLE = 5    # visible preset rows

    def __init__(self, game):
        self.game = game
        self.sel_type: str | None = None
        self.sel_team = 0
        self.hover: tuple | None = None
        self.unit_scroll = 0
        self.preset_scroll = 0
        self._unit_sb_drag = False
        self._preset_sb_drag = False

    # ── layout rects (design pixels → scaled) ─────────────────

    def _unit_rect(self, slot):
        """Screen rect for visible unit row `slot`."""
        s = self.game._s
        y = self.UNIT_AREA_Y + slot * (self.UNIT_ROW_H + self.UNIT_ROW_GAP)
        return pygame.Rect(s(12), s(y), s(196), s(self.UNIT_ROW_H))

    def _unit_max_scroll(self):
        return max(0, len(config.UNIT_TYPES) - self.UNIT_VISIBLE)

    def _unit_scrollbar_track(self):
        s = self.game._s
        y = self.UNIT_AREA_Y
        h = self.UNIT_VISIBLE * (self.UNIT_ROW_H + self.UNIT_ROW_GAP) - self.UNIT_ROW_GAP
        return pygame.Rect(int(s(212)), int(s(y)), int(s(7)), int(s(h)))

    def _unit_scrollbar_thumb(self):
        track = self._unit_scrollbar_track()
        n = len(config.UNIT_TYPES)
        th = max(int(track.h * self.UNIT_VISIBLE / n), int(self.game._s(20)))
        ms = self._unit_max_scroll()
        off = int((track.h - th) * (self.unit_scroll / ms)) if ms else 0
        return pygame.Rect(track.x, track.y + off, track.w, th)

    def _preset_rect(self, slot):
        """Screen rect for visible preset row `slot`."""
        s = self.game._s
        y = self.PRESET_AREA_Y + slot * (self.PRESET_ROW_H + self.PRESET_ROW_GAP)
        return pygame.Rect(s(12), s(y), s(196), s(self.PRESET_ROW_H))

    def _preset_max_scroll(self):
        return max(0, len(config.PRESETS) - self.PRESET_VISIBLE)

    def _preset_scrollbar_track(self):
        s = self.game._s
        y = self.PRESET_AREA_Y
        h = self.PRESET_VISIBLE * (self.PRESET_ROW_H + self.PRESET_ROW_GAP) - self.PRESET_ROW_GAP
        return pygame.Rect(int(s(212)), int(s(y)), int(s(7)), int(s(h)))

    def _preset_scrollbar_thumb(self):
        track = self._preset_scrollbar_track()
        n = len(config.PRESETS)
        th = max(int(track.h * self.PRESET_VISIBLE / n), int(self.game._s(20)))
        ms = self._preset_max_scroll()
        off = int((track.h - th) * (self.preset_scroll / ms)) if ms else 0
        return pygame.Rect(track.x, track.y + off, track.w, th)

    def _start_btn_rect(self):
        s = self.game._s
        cx = s(config.WINDOW_WIDTH) // 2
        return pygame.Rect(cx - s(100), s(config.WINDOW_HEIGHT - 55), s(200), s(44))

    # ── scrollbar helpers ─────────────────────────────────────

    def _scroll_unit_to_pixel(self, my):
        track = self._unit_scrollbar_track()
        ms = self._unit_max_scroll()
        if ms <= 0:
            return
        frac = (my - track.y) / max(track.h, 1)
        self.unit_scroll = max(0, min(ms, round(frac * ms)))

    def _scroll_preset_to_pixel(self, my):
        track = self._preset_scrollbar_track()
        ms = self._preset_max_scroll()
        if ms <= 0:
            return
        frac = (my - track.y) / max(track.h, 1)
        self.preset_scroll = max(0, min(ms, round(frac * ms)))

    # ── event handling ────────────────────────────────────────

    def handle(self, ev):
        if ev.type == pygame.MOUSEWHEEL:
            mx, my = ev.pos if hasattr(ev, 'pos') else (0, 0)
            s = self.game._s
            # Determine which area the mouse is over
            if my < s(self.PRESET_HEADER_Y):
                self.unit_scroll = max(0, min(self._unit_max_scroll(),
                                              self.unit_scroll - ev.y))
            else:
                self.preset_scroll = max(0, min(self._preset_max_scroll(),
                                                self.preset_scroll - ev.y))
            return
        if ev.type == pygame.MOUSEBUTTONUP:
            self._unit_sb_drag = False
            self._preset_sb_drag = False
            return
        if ev.type == pygame.MOUSEMOTION:
            if self._unit_sb_drag:
                self._scroll_unit_to_pixel(ev.pos[1]); return
            if self._preset_sb_drag:
                self._scroll_preset_to_pixel(ev.pos[1]); return
            self.hover = self.game.hex_renderer.pixel_to_hex(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            s = self.game._s
            if ev.button == 1:
                # Unit scrollbar
                if self._unit_max_scroll() > 0:
                    if self._unit_scrollbar_thumb().collidepoint(mx, my):
                        self._unit_sb_drag = True; return
                    if self._unit_scrollbar_track().collidepoint(mx, my):
                        self._scroll_unit_to_pixel(my); return
                # Preset scrollbar
                if self._preset_max_scroll() > 0:
                    if self._preset_scrollbar_thumb().collidepoint(mx, my):
                        self._preset_sb_drag = True; return
                    if self._preset_scrollbar_track().collidepoint(mx, my):
                        self._scroll_preset_to_pixel(my); return
                # Unit rows
                names = list(config.UNIT_TYPES)
                for slot in range(min(self.UNIT_VISIBLE, len(names))):
                    if self._unit_rect(slot).collidepoint(mx, my):
                        self.sel_type = names[self.unit_scroll + slot]; return
                # Team / player buttons
                team_rect = pygame.Rect(int(s(14)), int(s(12)),
                                        int(s(100)), int(s(32)))
                if team_rect.collidepoint(mx, my):
                    self.sel_team = 1 - self.sel_team; return
                player_rect = pygame.Rect(int(s(120)), int(s(12)),
                                          int(s(100)), int(s(32)))
                if player_rect.collidepoint(mx, my):
                    self.game.player_team = 1 - self.game.player_team; return
                # Preset rows
                preset_names = list(config.PRESETS)
                for slot in range(min(self.PRESET_VISIBLE, len(preset_names))):
                    if self._preset_rect(slot).collidepoint(mx, my):
                        self._load_preset(preset_names[self.preset_scroll + slot]); return
                # Start button
                if self._start_btn_rect().collidepoint(mx, my):
                    if self._can_start():
                        self.game.start_battle()
                    return
                # Hex grid placement
                hex_pos = self.game.hex_renderer.pixel_to_hex(mx, my)
                if hex_pos and self.sel_type:
                    col, row = hex_pos
                    if self.game.grid.half_of(col) == self.sel_team:
                        self.game.units = [u for u in self.game.units if u.pos != hex_pos]
                        self.game.units.append(
                            Unit.from_type(self.sel_type, self.sel_team, col, row))
            elif ev.button == 3:
                hex_pos = self.game.hex_renderer.pixel_to_hex(mx, my)
                if hex_pos:
                    self.game.units = [u for u in self.game.units if u.pos != hex_pos]

    # ── drawing ───────────────────────────────────────────────

    def draw(self):
        g = self.game
        s = g._s
        canvas = g.canvas

        # Reposition grid to centre in available space after left panel
        panel_right = s(228)
        grid_w = g.grid.cols * g.hex_renderer.hex_w
        avail = g.win_w - panel_right
        g.hex_renderer.reposition(panel_right + (avail - grid_w) / 2, s(config.GRID_OFFSET_Y))

        # left panel background
        panel = pygame.Rect(int(s(self.PANEL_X)), int(s(4)), int(s(self.PANEL_W)),
                            int(s(config.WINDOW_HEIGHT) - s(8)))
        pygame.draw.rect(canvas, config.PANEL_BG, panel, border_radius=int(s(6)))
        pygame.draw.rect(canvas, (55, 65, 90), panel, 1, border_radius=int(s(6)))

        # team selector
        draw_btn(canvas, s(14), s(12), s(100), s(32),
                 f"Team: {fonts.team_name(self.sel_team)}",
                 fonts.team_color(self.sel_team), config.WHITE)
        # player side selector
        draw_btn(canvas, s(120), s(12), s(100), s(32),
                 f"Play: {fonts.team_name(self.game.player_team)}",
                 config.YELLOW, config.BLACK)
        hint = fonts.BODY.render("Click palette -> hex. Right-click remove.",
                                 True, (170, 180, 200))
        canvas.blit(hint, (s(230), s(16)))

        # ── UNITS section ──
        canvas.blit(fonts.TITLE.render("UNITS", True, config.WHITE),
                    (s(18), s(self.HEADER_Y)))

        names = list(config.UNIT_TYPES)
        for slot in range(min(self.UNIT_VISIBLE, len(names))):
            name = names[self.unit_scroll + slot]
            r = self._unit_rect(slot)
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
                        (r.x + s(8), r.y + s(4)))
            canvas.blit(fonts.BODY.render(name, True, config.WHITE),
                        (r.x + s(28), r.y + s(3)))
            canvas.blit(fonts.DATA.render(
                f"A{ut['attack']} D{ut['defense']} H{ut['hp']} S{ut['speed']} x{ut['count']}",
                True, (170, 180, 200)), (r.x + s(28), r.y + s(22)))

        # Unit scrollbar
        if self._unit_max_scroll() > 0:
            track = self._unit_scrollbar_track()
            thumb = self._unit_scrollbar_thumb()
            pygame.draw.rect(canvas, (28, 34, 50), track, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (95, 108, 140), thumb, border_radius=int(s(3)))

        # ── Separator ──
        sep_y = s(self.PRESET_HEADER_Y - 8)
        pygame.draw.line(canvas, (55, 65, 90),
                         (s(14), sep_y), (s(210), sep_y), 1)

        # ── PRESETS section ──
        canvas.blit(fonts.LABEL.render("PRESETS", True, config.WHITE),
                    (s(18), s(self.PRESET_HEADER_Y)))

        preset_names = list(config.PRESETS)
        for slot in range(min(self.PRESET_VISIBLE, len(preset_names))):
            pname = preset_names[self.preset_scroll + slot]
            r = self._preset_rect(slot)
            pygame.draw.rect(canvas, (30, 38, 55), r, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (60, 68, 92), r, 1, border_radius=int(s(3)))
            canvas.blit(fonts.BODY.render(f"Preset: {pname}", True, config.WHITE),
                        (r.x + s(8), r.y + s(4)))

        # Preset scrollbar
        if self._preset_max_scroll() > 0:
            track = self._preset_scrollbar_track()
            thumb = self._preset_scrollbar_thumb()
            pygame.draw.rect(canvas, (28, 34, 50), track, border_radius=int(s(3)))
            pygame.draw.rect(canvas, (95, 108, 140), thumb, border_radius=int(s(3)))

        # hex grid
        highlights = {}
        if self.hover and self.sel_type:
            if g.grid.half_of(self.hover[0]) == self.sel_team:
                highlights[self.hover] = fonts.team_color(self.sel_team)
        g.hex_renderer.draw_grid(canvas, highlights)
        self._draw_units()

        # start button
        sr = self._start_btn_rect()
        can = self._can_start()
        draw_btn(canvas, sr.x, sr.y, sr.w, sr.h, "Start Battle",
                 config.GREEN if can else config.GRAY, config.BLACK)

        # team unit counts above grid
        for team in (0, 1):
            n = sum(1 for u in g.units if u.team == team)
            txt = fonts.BODY.render(f"{fonts.team_name(team)}: {n} units",
                                    True, fonts.team_light(team))
            if team == 0:
                canvas.blit(txt, (int(g.hex_renderer.ox) + int(s(10)), int(s(56))))
            else:
                canvas.blit(txt, (int(g.hex_renderer.ox) + int(grid_w)
                                  - txt.get_width() - int(s(10)), int(s(56))))

    def _draw_units(self):
        g = self.game
        for u in g.units:
            if not u.is_alive:
                continue
            cx, cy = g.hex_renderer.center(*u.pos)
            draw_unit(g.canvas, g._s, g.hex_renderer, u, cx, cy)

    # ── game logic helpers ────────────────────────────────────

    def _can_start(self):
        return (any(u.team == 0 for u in self.game.units)
                and any(u.team == 1 for u in self.game.units))

    def _load_preset(self, name):
        preset = config.PRESETS[name]
        self.game.units = []
        # Siege flag: if preset has "siege": True, store for start_battle.
        self.game._siege = preset.get("siege", False)
        for team, placements in preset.items():
            if isinstance(team, str):
                continue  # skip "siege" key
            for type_name, col, row in placements:
                self.game.units.append(Unit.from_type(type_name, team, col, row))
