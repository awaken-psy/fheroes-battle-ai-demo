"""Battle screen: animation engine, combat rendering, player + AI control.

Supports two modes:
  - Player control: when the current unit belongs to player_team, the
    screen waits for mouse/keyboard input instead of calling AI.
    Click a hex to move/attack, right-click to wait, S to cast a spell.
  - AI control: ClassicAI drives all non-player units (retreat, cast, act).
"""

import math
import pygame
import numpy as np

import config
from .. import fonts
from ..renderer import Popup, draw_btn, draw_unit
from engine.battle_state import BattleState
from engine.actions import MoveAction, AttackAction, SkipAction, CastAction, RetreatAction
from engine.battle_logger import BattleLogger
from ai import create_ai
from ai.action_space import (
    legal_mask, enumerate_legal, index_to_action, action_to_index,
    WAIT_IDX, DEFEND_IDX, RETREAT_IDX,
    MOVE_START, MOVE_END, ATTACK_START, ATTACK_END,
    CAST_START, CAST_END,
    GRID_CELLS, cell_to_index, index_to_cell,
    _SPELL_ORDER, _SPELL_INDEX,
)

# Animation phases
PH_IDLE, PH_MOVE, PH_STRIKE, PH_RETAL, PH_AFTER = range(5)


class BattleScreen:
    """Handles the battle phase: player input, AI turns, animation, rendering."""

    def __init__(self, game, player_team=0):
        self.game = game
        self.player_team = player_team
        self.ai = create_ai("classic")
        self.ai_strategy = {0: "classic", 1: "classic"}
        self._ai_cache = {}  # team → AI instance
        self.logger = BattleLogger()

        self.battle: BattleState | None = None
        self.b_action = None
        self.b_desc = ""
        self.b_log: list[str] = []
        self.b_path: set | None = None
        self.b_target = None
        self.speed = 2
        self.paused = False
        self.debug = True

        self._ph = PH_IDLE
        self._ph_t = 0.0
        self._anim_unit = None
        self._anim_px = (0.0, 0.0)
        self._move_px = []
        self._move_idx = 0
        self._move_frac = 0.0
        self._popups: list[Popup] = []
        self._projectile = None
        self._flash = None
        self._exec_result = None

        self._round_order = None
        self._order_idx = 0
        self._round_num = 0

        # Player input state
        self._await_input = False       # waiting for player to act
        self._pending_unit = None       # unit the player controls
        self._legal_mask = None         # cached legal mask for current unit
        self._hover_cell = None         # hex under mouse
        self._cast_mode = False         # spell selection mode
        self._spell_list = []           # available spell names
        self._spell_sel = 0             # selected spell index in list
        self._selected_spell_slot = None  # spell slot for casting
        self._await_spell_target = False  # waiting for spell target hex
        self._actions_remaining = 1     # morale: 2 if good morale

    # ── event handling ────────────────────────────────────────

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif ev.key == pygame.K_1:
                self.speed = 0
            elif ev.key == pygame.K_2:
                self.speed = 1
            elif ev.key == pygame.K_3:
                self.speed = 2
            elif ev.key == pygame.K_r:
                if self.battle:
                    self.logger.end(None, self.battle.round_num)
                self.game.reset()
                self._await_input = False
                self._cast_mode = False
            elif ev.key == pygame.K_f:
                self._fast_forward()
            elif ev.key == pygame.K_d and not self._await_input:
                self.debug = not self.debug

        # Player input — only when waiting and not paused
        if self._await_input and not self.paused:
            self._handle_player_input(ev)

    def _handle_player_input(self, ev):
        """Process mouse/keyboard events for player-controlled units."""

        # ── Spell selection mode ──
        if self._cast_mode:
            self._handle_spell_input(ev)
            return

        # ── Normal mode ──
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_s:
                self._open_spell_list()
            elif ev.key == pygame.K_d:
                self._player_act(WAIT_IDX)  # Defend/Wait
            elif ev.key == pygame.K_ESCAPE:
                pass  # no-op in normal mode
        elif ev.type == pygame.MOUSEMOTION:
            self._hover_cell = self.game.hex_renderer.pixel_to_hex(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:  # left click: move or attack
                hex_pos = self.game.hex_renderer.pixel_to_hex(*ev.pos)
                if hex_pos:
                    self._handle_hex_click(hex_pos)
            elif ev.button == 3:  # right click: wait
                self._player_act(WAIT_IDX)

    def _handle_hex_click(self, hex_pos):
        """Left-click on a hex: determine if it's a move or attack action."""
        if self._legal_mask is None or self._pending_unit is None:
            return

        cell_idx = cell_to_index(*hex_pos)

        # Check if clicking an enemy → attack
        enemies = self.battle.enemies_of(self._pending_unit)
        clicked_enemy = self.battle.unit_at(hex_pos)

        if clicked_enemy and clicked_enemy in enemies:
            # Find the attack action index for this enemy
            from ai.action_space import _enemy_list, _attack_positions, MAX_ATTACK_POSITIONS
            enemy_idx = enemies.index(clicked_enemy)
            positions = _attack_positions(self.battle.grid, self._pending_unit, clicked_enemy)

            # Try ranged first (position 0)
            if self._pending_unit.is_archer:
                idx = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + 0
                if self._legal_mask[idx] == 1.0:
                    self._player_act(idx)
                    return

            # Try melee positions
            for pos_idx in range(1, min(len(positions), MAX_ATTACK_POSITIONS)):
                idx = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + pos_idx
                if self._legal_mask[idx] == 1.0:
                    self._player_act(idx)
                    return
            return

        # Check if clicking a reachable hex → move
        move_idx = MOVE_START + cell_idx
        if move_idx <= MOVE_END and self._legal_mask[move_idx] == 1.0:
            self._player_act(move_idx)

    def _player_act(self, action_idx: int):
        """Execute a player action from its flat index."""
        if self._pending_unit is None or self.battle is None:
            return

        action = index_to_action(action_idx, self.battle, self._pending_unit)
        self.b_action = action

        if isinstance(action, SkipAction):
            self.b_desc = "Player: Wait"
        elif isinstance(action, MoveAction):
            self.b_desc = f"Player: Move -> {action.path[-1]}"
        elif isinstance(action, AttackAction):
            self.b_desc = f"Player: Attack {action.target.name}"
        elif isinstance(action, CastAction):
            self.b_desc = f"Player: Cast {action.spell.name}"

        self._await_input = False
        self._legal_mask = None
        self._cast_mode = False
        self._pending_unit = None
        self._actions_remaining -= 1
        self._start_anim(action)

    # ── Spell UI ──

    def _open_spell_list(self):
        """Open the spell selection panel."""
        if self._pending_unit is None or self.battle is None:
            return
        hero = self.battle.heroes.get(self._pending_unit.team)
        if hero is None or hero._cast_this_round:
            return

        from engine.spells import SPELLS
        self._spell_list = []
        for spell_name in _SPELL_ORDER:
            spell = SPELLS[spell_name]
            if spell_name in [s.name for s in hero.spellbook] and hero.can_cast(spell):
                self._spell_list.append(spell_name)

        if not self._spell_list:
            return

        self._cast_mode = True
        self._spell_sel = 0

    def _handle_spell_input(self, ev):
        """Handle input while spell panel is open."""
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self._cast_mode = False
            elif ev.key == pygame.K_UP:
                self._spell_sel = max(0, self._spell_sel - 1)
            elif ev.key == pygame.K_DOWN:
                self._spell_sel = min(len(self._spell_list) - 1, self._spell_sel + 1)
            elif ev.key == pygame.K_RETURN:
                self._select_spell()
        elif ev.type == pygame.MOUSEMOTION:
            self._hover_cell = self.game.hex_renderer.pixel_to_hex(*ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            # Click on hex → cast selected spell on that hex
            self._select_spell(target_hex=self._hover_cell)

    def _select_spell(self, target_hex=None):
        """Confirm spell selection and enter target selection or cast directly."""
        if not self._spell_list:
            return

        spell_name = self._spell_list[self._spell_sel]
        spell_slot = _SPELL_INDEX[spell_name]

        if target_hex is None:
            # Need to pick a target — switch to target mode
            self._selected_spell_slot = spell_slot
            # Compute legal targets for this spell
            if self._legal_mask is None:
                return
            # Set mask to only show cast targets for this spell
            self._cast_spell_mask = np.zeros_like(self._legal_mask)
            base = CAST_START + spell_slot * GRID_CELLS
            self._cast_spell_mask[base:base + GRID_CELLS] = self._legal_mask[base:base + GRID_CELLS]
            # If there are legal targets, keep cast_mode and wait for click
            if self._cast_spell_mask[base:base + GRID_CELLS].sum() > 0:
                self._await_spell_target = True
                return
            else:
                self._cast_mode = False
                return

        # Have target hex — cast
        cell_idx = cell_to_index(*target_hex)
        base = CAST_START + spell_slot * GRID_CELLS
        cast_idx = base + cell_idx
        if self._legal_mask[cast_idx] == 1.0:
            # Execute spell
            action = index_to_action(cast_idx, self.battle, self._pending_unit)
            if isinstance(action, CastAction):
                self._do_cast((action, f"Player: Cast {action.spell.name}"))
                self._cast_mode = False
                self._await_spell_target = False
                self._selected_spell_slot = None
                # After casting, refresh legal mask (state may have changed)
                if self._pending_unit and self._pending_unit.is_alive:
                    self._legal_mask = legal_mask(self.battle, self._pending_unit)

    # ── update ────────────────────────────────────────────────

    def update(self, dt):
        self._popups = [p for p in self._popups if p.update(dt)]
        if self._flash:
            pos, timer = self._flash
            timer -= dt
            self._flash = (pos, timer) if timer > 0 else None
        if self.paused:
            return
        if self._await_input:
            return  # wait for player

        spd = [1.5, 1.0, 0.5][self.speed]

        if self._ph == PH_IDLE:
            self._next_unit()
        elif self._ph == PH_MOVE:
            self._anim_move(dt, spd)
        elif self._ph == PH_STRIKE:
            self._anim_strike(dt, spd)
        elif self._ph == PH_RETAL:
            self._ph_t += dt
            if self._ph_t >= 0.35 * spd:
                self._ph = PH_AFTER; self._ph_t = 0
        elif self._ph == PH_AFTER:
            self._ph_t += dt
            if self._ph_t >= 0.25 * spd:
                self._anim_unit = None; self._exec_result = None
                self.b_path = None; self.b_target = None
                if self.battle.is_over():
                    timeout = self.battle.round_num >= BattleState.MAX_ROUNDS
                    self.logger.end(self.battle.winner(), self.battle.round_num,
                                    timeout=timeout)
                    self.game.state = config.GAME_OVER; return

                # Good morale: extra action for same unit
                if self._actions_remaining > 0:
                    if isinstance(self.b_action, MoveAction):
                        unit = self.b_action.unit
                    elif isinstance(self.b_action, AttackAction):
                        unit = self.b_action.attacker
                    elif isinstance(self.b_action, SkipAction):
                        unit = self.b_action.unit
                    else:
                        unit = None
                    if unit and unit.is_alive and not self.battle.is_over():
                        if unit.team == self.player_team and self.player_team is not None:
                            self._await_input = True
                            self._pending_unit = unit
                            self._legal_mask = legal_mask(self.battle, unit)
                            self._ph = PH_IDLE
                            return
                        else:
                            team_ai = self._get_team_ai(unit.team)
                            action, desc = team_ai.decide(self.battle, unit)
                            self.b_action = action; self.b_desc = desc
                            self._start_anim(action)
                            return

                self._ph = PH_IDLE

    def _get_team_ai(self, team):
        """Get or create the AI instance for a team's selected strategy."""
        if team not in self._ai_cache:
            strategy = self.ai_strategy.get(team, "classic")
            if strategy == "deep":
                try:
                    self._ai_cache[team] = create_ai("deep", device="cpu")
                except Exception:
                    self._ai_cache[team] = create_ai("classic")
            else:
                self._ai_cache[team] = create_ai("classic")
        return self._ai_cache[team]

    def _next_unit(self):
        if not self.battle.alive():
            return
        order = self.battle.turn_order()
        if (not hasattr(self, '_round_order')
                or self._round_order != order
                or self._round_num != self.battle.round_num):
            self._round_order = order
            self._order_idx = 0
            self.battle.start_round()
            self._round_num = self.battle.round_num
            self.logger.round_start(self._round_num)
        while self._order_idx < len(self._round_order):
            unit = self._round_order[self._order_idx]
            self._order_idx += 1
            if unit.is_alive:
                # Morale roll
                morale = self.battle.roll_morale(unit.team, unit)
                if morale < 0:
                    continue  # bad morale → skip

                self._actions_remaining = 2 if morale > 0 else 1

                if unit.team == self.player_team and self.player_team is not None:
                    # Player-controlled unit — wait for input
                    self._await_input = True
                    self._pending_unit = unit
                    self._legal_mask = legal_mask(self.battle, unit)
                    return
                else:
                    # AI-controlled unit — use team's selected strategy
                    team_ai = self._get_team_ai(unit.team)
                    retreat = team_ai.check_retreat(self.battle, unit)
                    if retreat is not None:
                        farewell, retreat_action = retreat
                        if farewell is not None:
                            self._do_cast(farewell)
                        rr = self.battle.execute(retreat_action)
                        self.logger.action("[RETREAT]", rr['desc'])
                        skip = SkipAction(unit)
                        self.b_action = skip; self.b_desc = f"[RETREAT] {rr['desc']}"
                        self._start_anim(skip)
                        return
                    cast = team_ai.maybe_cast_spell(self.battle, unit)
                    if cast is not None:
                        self._do_cast(cast)
                    action, desc = team_ai.decide(self.battle, unit)
                    self.b_action = action; self.b_desc = desc
                    self._start_anim(action)
                    return
        # all units in this round processed
        self._round_order = None

    def _do_cast(self, cast):
        """Execute a hero spellcast instantly (no movement animation), with a popup."""
        action, desc = cast
        result = self.battle.execute(action)
        self.b_log.append(result['desc'])
        if len(self.b_log) > 5:
            self.b_log.pop(0)
        self.logger.action(desc, result['desc'])
        g = self.game; s = g._s
        tx, ty = g.hex_renderer.center(*action.target.pos)
        spell = action.spell.name
        text = f"{spell} -{result['dmg']}" if result.get('dmg') else spell
        self._popups.append(
            Popup(tx, ty - s(18), text, config.CYAN, life=2.0, speed=g._rs, big=True))
        self._flash = (action.target.pos, 0.4)

    def _fast_forward(self):
        """Skip all animations, resolve battle to completion, log and return."""
        if not self.battle or self.battle.is_over():
            return
        self._await_input = False
        from headless import _take_unit_turn
        while not self.battle.is_over():
            order = self.battle.turn_order()
            if not order:
                break
            self.battle.start_round()
            self.logger.round_start(self.battle.round_num)
            for unit in order:
                if not unit.is_alive:
                    continue
                if self.battle.is_over():
                    break
                team_ai = self._get_team_ai(unit.team)
                _take_unit_turn(self.battle, team_ai, unit, log=self.logger.action)
            if self.battle._retreated is not None:
                break
        timeout = self.battle.round_num >= BattleState.MAX_ROUNDS
        self.logger.end(self.battle.winner(), self.battle.round_num, timeout=timeout)
        self.game.reset()

    # ── animation engine ──────────────────────────────────────

    def _start_anim(self, action):
        self._exec_result = None
        self._projectile = None

        if isinstance(action, SkipAction):
            self._anim_unit = action.unit
            self._anim_px = self.game.hex_renderer.center(*action.unit.pos)
            self.logger.action(self.b_desc, "skip")
            self._ph = PH_AFTER; self._ph_t = 0
            return

        if isinstance(action, MoveAction):
            self._anim_unit = action.unit
            self._move_px = [self.game.hex_renderer.center(*p) for p in action.path]
            self._move_idx = 0; self._move_frac = 0.0
            self._anim_px = self._move_px[0]
            self.b_path = set(action.path)
            self._ph = PH_MOVE; self._ph_t = 0
            return

        if isinstance(action, AttackAction):
            self._anim_unit = action.attacker
            self.b_target = action.target
            if action.ranged:
                self._anim_px = self.game.hex_renderer.center(*action.attacker.pos)
                self._ph = PH_STRIKE; self._ph_t = 0
            else:
                if action.from_pos and action.from_pos != action.attacker.pos:
                    occ = self.battle.occupied(exclude=action.attacker)
                    path = self.game.grid.find_path(
                        action.attacker.pos, action.from_pos, occ,
                        action.attacker.is_flying, action.attacker.speed)
                    self._move_px = (
                        [self.game.hex_renderer.center(*p) for p in path]
                        if path
                        else [self.game.hex_renderer.center(*action.attacker.pos),
                              self.game.hex_renderer.center(*action.from_pos)])
                    self.b_path = {action.from_pos}
                else:
                    self._move_px = [self.game.hex_renderer.center(*action.attacker.pos)]
                self._move_idx = 0; self._move_frac = 0.0
                self._anim_px = self._move_px[0]
                self._ph = PH_MOVE; self._ph_t = 0

    def _anim_move(self, dt, spd):
        px_per_sec = 280.0 / spd

        if self._move_idx >= len(self._move_px) - 1:
            self._anim_px = self._move_px[-1]
            if isinstance(self.b_action, AttackAction):
                self._ph = PH_STRIKE; self._ph_t = 0
            else:
                result = self.battle.execute(self.b_action)
                self.b_log.append(result['desc'])
                if len(self.b_log) > 5: self.b_log.pop(0)
                self.logger.action(self.b_desc, result['desc'])
                self._ph = PH_AFTER; self._ph_t = 0
            return

        a = self._move_px[self._move_idx]
        b = self._move_px[self._move_idx + 1]
        seg_len = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1.0)
        self._move_frac += px_per_sec * dt / seg_len

        while self._move_frac >= 1.0 and self._move_idx < len(self._move_px) - 1:
            self._move_frac -= 1.0
            self._move_idx += 1

        if self._move_idx >= len(self._move_px) - 1:
            self._anim_px = self._move_px[-1]
        else:
            a = self._move_px[self._move_idx]
            b = self._move_px[self._move_idx + 1]
            t = min(self._move_frac, 1.0)
            self._anim_px = (a[0] + (b[0] - a[0]) * t,
                             a[1] + (b[1] - a[1]) * t)

    def _anim_strike(self, dt, spd):
        action = self.b_action
        self._ph_t += dt
        s = self.game._s

        if action.ranged:
            lunge_dur = 0.25 * spd
            src = self.game.hex_renderer.center(*action.attacker.pos)
            dst = self.game.hex_renderer.center(*action.target.pos)
            prog = min(1.0, self._ph_t / lunge_dur)
            self._projectile = (src, dst, prog)

            if self._ph_t >= lunge_dur:
                self._projectile = None
                self._exec_result = self.battle.execute(self.b_action)
                self.b_log.append(self._exec_result['desc'])
                if len(self.b_log) > 5: self.b_log.pop(0)
                self.logger.action(self.b_desc, self._exec_result['desc'])
                self._popups.append(
                    Popup(dst[0], dst[1] - s(12),
                          f"-{self._exec_result['dmg']}", config.RED, speed=self.game._rs))
                if not self._exec_result['target_alive']:
                    self._popups.append(
                        Popup(dst[0], dst[1] - s(32), "DEAD", config.YELLOW, speed=self.game._rs))
                self._flash = (action.target.pos, 0.15)
                self._goto_retal_or_after()
        else:
            lunge_dur = 0.15 * spd
            ret_dur = 0.12 * spd
            atk_px = (self._move_px[-1] if self._move_px
                      else self.game.hex_renderer.center(*action.attacker.pos))
            tgt_px = self.game.hex_renderer.center(*action.target.pos)
            dx = tgt_px[0] - atk_px[0]
            dy = tgt_px[1] - atk_px[1]

            if self._ph_t < lunge_dur:
                t = self._ph_t / lunge_dur
                t = t * t * (3 - 2 * t)
                self._anim_px = (atk_px[0] + dx * 0.35 * t,
                                 atk_px[1] + dy * 0.35 * t)
            elif self._ph_t < lunge_dur + ret_dur:
                if self._exec_result is None:
                    self._exec_result = self.battle.execute(self.b_action)
                    self.b_log.append(self._exec_result['desc'])
                    if len(self.b_log) > 5: self.b_log.pop(0)
                    self.logger.action(self.b_desc, self._exec_result['desc'])
                    self._popups.append(
                        Popup(tgt_px[0], tgt_px[1] - s(12),
                              f"-{self._exec_result['dmg']}", config.RED, speed=self.game._rs))
                    if not self._exec_result['target_alive']:
                        self._popups.append(
                            Popup(tgt_px[0], tgt_px[1] - s(32), "DEAD",
                                  config.YELLOW, speed=self.game._rs))
                    self._flash = (action.target.pos, 0.15)
                t = min(1.0, (self._ph_t - lunge_dur) / ret_dur)
                self._anim_px = (atk_px[0] + dx * 0.35 * (1 - t),
                                 atk_px[1] + dy * 0.35 * (1 - t))
            else:
                self._anim_px = atk_px
                self._goto_retal_or_after()

    def _goto_retal_or_after(self):
        r = self._exec_result
        if r and r['ret_dmg'] > 0:
            atk = self.b_action.attacker
            ax, ay = self.game.hex_renderer.center(*atk.pos)
            s = self.game._s
            self._popups.append(
                Popup(ax, ay - s(12), f"-{r['ret_dmg']}", config.ORANGE, speed=self.game._rs))
            if not r['attacker_alive']:
                self._popups.append(
                    Popup(ax, ay - s(32), "DEAD", config.YELLOW, speed=self.game._rs))
            self._flash = (atk.pos, 0.15)
            self._ph = PH_RETAL; self._ph_t = 0
        else:
            self._ph = PH_AFTER; self._ph_t = 0

    # ── drawing ───────────────────────────────────────────────

    def draw(self):
        g = self.game
        s = g._s
        canvas = g.canvas

        # centre grid
        grid_w = g.grid.cols * g.hex_renderer.hex_w
        g.hex_renderer.reposition((g.win_w - grid_w) / 2, s(config.GRID_OFFSET_Y))

        # top bar
        top_bar = pygame.Rect(0, 0, s(config.WINDOW_WIDTH), s(42))
        pygame.draw.rect(canvas, config.PANEL_BG, top_bar)
        pygame.draw.line(canvas, (55, 65, 90),
                         (0, s(42)), (s(config.WINDOW_WIDTH), s(42)), 1)

        if self.battle:
            cx = int(g.win_w // 2)
            bar_cy = int(s(21))
            round_surf = fonts.BIG.render(
                f"Round {self.battle.round_num}", True, config.WHITE)
            round_rect = round_surf.get_rect(center=(cx, bar_cy))
            canvas.blit(round_surf, round_rect)
            div_x_l = round_rect.left - int(s(12))
            div_x_r = round_rect.right + int(s(12))
            pygame.draw.line(canvas, (55, 65, 90),
                             (div_x_l, int(s(8))), (div_x_l, int(s(34))), 1)
            pygame.draw.line(canvas, (55, 65, 90),
                             (div_x_r, int(s(8))), (div_x_r, int(s(34))), 1)
            div_pad = int(s(20))
            for team in (0, 1):
                units = self.battle.alive(team)
                total = sum(u.strength for u in units)
                n = len(units)
                label = f"{fonts.team_name(team)}: {n} units  STR {total:.0f}"
                txt = fonts.TITLE.render(label, True, fonts.team_light(team))
                if team == 0:
                    rect = txt.get_rect(right=div_x_l - div_pad, centery=bar_cy)
                else:
                    rect = txt.get_rect(left=div_x_r + div_pad, centery=bar_cy)
                canvas.blit(txt, rect)

        # hex highlights
        highlights = {}
        if self._await_input and self._legal_mask is not None:
            self._draw_legal_highlights(highlights)
        if self.b_path:
            for p in self.b_path:
                highlights.setdefault(p, config.PATH_COLOR)
        if self.b_action and isinstance(self.b_action, MoveAction):
            for p in self.b_action.path:
                highlights[p] = tuple(
                    min(highlights.get(p, config.BG)[i] + 40, 255) for i in range(3))
        if self._flash and self._flash[1] > 0:
            intensity = int(min(255, 160 * self._flash[1] / 0.15))
            highlights[self._flash[0]] = (intensity, 40, 40)
        g.hex_renderer.draw_grid(canvas, highlights)

        # Siege structures
        if self.battle and self.battle.castle:
            g.hex_renderer.draw_siege(canvas, self.battle.castle)

        if self.b_target and self.b_action and isinstance(self.b_action, AttackAction):
            g.hex_renderer.draw_dashed_line(canvas, self.b_action.attacker.pos,
                                            self.b_target.pos, config.TARGET_COLOR, 2)
            g.hex_renderer.draw_overlay(canvas, self.b_target.pos, config.TARGET_COLOR, 3)

        if self._projectile:
            src, dst, prog = self._projectile
            ex = src[0] + (dst[0] - src[0]) * prog
            ey = src[1] + (dst[1] - src[1]) * prog
            pygame.draw.line(canvas, config.YELLOW,
                             (int(src[0]), int(src[1])), (int(ex), int(ey)), 2)
            pygame.draw.circle(canvas, config.YELLOW, (int(ex), int(ey)), 4)

        current_unit = None
        if self._await_input and self._pending_unit:
            current_unit = self._pending_unit
        elif self.b_action:
            if isinstance(self.b_action, MoveAction):
                current_unit = self.b_action.unit
            elif isinstance(self.b_action, AttackAction):
                current_unit = self.b_action.attacker
            elif isinstance(self.b_action, SkipAction):
                current_unit = self.b_action.unit
        self._draw_units(current_unit)

        for p in self._popups:
            p.draw(canvas)

        # Spell panel overlay
        if self._cast_mode:
            self._draw_spell_panel(canvas, s)

        if self.debug and self.b_desc:
            self._draw_debug(canvas, s)
        self._draw_hints(canvas, s)

    def _draw_legal_highlights(self, highlights):
        """Highlight legal move/attack/spell hexes for player."""
        mask = self._legal_mask
        unit = self._pending_unit
        if mask is None or unit is None:
            return

        # Move highlights (blue)
        for i in range(MOVE_START, MOVE_END + 1):
            if mask[i] == 1.0:
                col, row = index_to_cell(i - MOVE_START)
                highlights.setdefault((col, row), (40, 80, 140))

        # Attack highlights (red overlay on enemy positions)
        enemies = self.battle.enemies_of(unit)
        from ai.action_space import MAX_ATTACK_POSITIONS
        for enemy_idx, enemy in enumerate(enemies):
            if enemy_idx >= 7:
                break
            for pos_idx in range(MAX_ATTACK_POSITIONS):
                idx = ATTACK_START + enemy_idx * MAX_ATTACK_POSITIONS + pos_idx
                if mask[idx] == 1.0:
                    # Mark enemy position with red
                    highlights[enemy.pos] = (140, 40, 40)
                    break

        # Cast highlights (cyan) — only in cast mode
        if self._cast_mode and self._selected_spell_slot is not None:
            base = CAST_START + self._selected_spell_slot * GRID_CELLS
            for i in range(GRID_CELLS):
                if mask[base + i] == 1.0:
                    col, row = index_to_cell(i)
                    highlights.setdefault((col, row), (40, 140, 140))

        # Hover highlight
        if self._hover_cell:
            highlights[self._hover_cell] = highlights.get(
                self._hover_cell, (200, 200, 200))

    def _draw_spell_panel(self, canvas, s):
        """Draw the spell selection panel on the right side."""
        panel_w = s(220)
        panel_h = s(40 + len(self._spell_list) * 28)
        panel_x = self.game.win_w - panel_w - s(10)
        panel_y = s(50)

        panel = pygame.Rect(int(panel_x), int(panel_y), int(panel_w), int(panel_h))
        pygame.draw.rect(canvas, (20, 28, 45), panel, border_radius=int(s(6)))
        pygame.draw.rect(canvas, (80, 100, 140), panel, 2, border_radius=int(s(6)))

        canvas.blit(fonts.TITLE.render("SPELLS", True, config.WHITE),
                    (panel_x + s(10), panel_y + s(6)))

        from engine.spells import SPELLS
        for i, spell_name in enumerate(self._spell_list):
            spell = SPELLS[spell_name]
            y = panel_y + s(34 + i * 28)
            sel = (i == self._spell_sel)
            bg = (50, 55, 78) if sel else (30, 38, 55)
            row = pygame.Rect(int(panel_x + s(6)), int(y),
                              int(panel_w - s(12)), int(s(24)))
            pygame.draw.rect(canvas, bg, row, border_radius=int(s(3)))
            if sel:
                pygame.draw.rect(canvas, config.YELLOW, row, 2, border_radius=int(s(3)))
            text = f"{spell_name} ({spell.cost}sp)"
            canvas.blit(fonts.BODY.render(text, True, config.WHITE),
                        (row.x + s(8), row.y + s(3)))

    def _draw_units(self, current=None):
        g = self.game
        for u in g.units:
            if not u.is_alive:
                continue
            if u is self._anim_unit and self._anim_px:
                cx, cy = self._anim_px
            else:
                cx, cy = g.hex_renderer.center(*u.pos)
            draw_unit(g.canvas, g._s, g.hex_renderer, u, cx, cy, current=(u is current))

    def _draw_debug(self, canvas, s):
        vw, vh = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
        bar_h = s(60)
        bar_y = s(vh) - s(88)
        bar = pygame.Rect(0, bar_y, s(vw), bar_h)
        pygame.draw.rect(canvas, config.PANEL_BG, bar)
        pygame.draw.line(canvas, (55, 65, 90), (0, bar_y), (s(vw), bar_y), 1)
        steps = self.b_desc.split(" -> ")
        action = steps[-1].strip() if steps else ""
        ai_label = fonts.BODY.render("AI ", True, config.GRAY)
        ai_text = fonts.BODY.render(action, True, config.CYAN)
        row1_y = bar_y + s(6)
        canvas.blit(ai_label, (s(14), row1_y))
        canvas.blit(ai_text, (s(14) + ai_label.get_width() + s(4), row1_y))
        if self.b_log:
            log_label = fonts.DATA.render("LOG ", True, config.GRAY)
            log_text = fonts.DATA.render(self.b_log[-1], True, (170, 180, 200))
            row2_y = bar_y + s(30)
            canvas.blit(log_label, (s(14), row2_y))
            canvas.blit(log_text, (s(14) + log_label.get_width() + s(4), row2_y))

    def _draw_hints(self, canvas, s):
        vw, vh = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
        spd_names = ["Slow", "Normal", "Fast"]
        if self._await_input:
            hints = ("[L-Click] Move/Attack  [R-Click] Wait  "
                     "[S] Spells  [D] Defend  [R] Reset  "
                     f"[1/2/3] {spd_names[self.speed]}  [F] Auto-Finish")
        else:
            hints = (f"[Space] {'>> Play' if self.paused else '|| Pause'}   "
                     f"[1/2/3] Speed: {spd_names[self.speed]}   "
                     f"[F] Auto-Finish   [R] Reset   "
                     f"[D] Debug: {'ON' if self.debug else 'OFF'}   "
                     f"[+/-] Size  [F11] Fullscreen")
        hint_y = s(vh) - s(22)
        pygame.draw.rect(canvas, config.PANEL_BG, (0, hint_y - s(4), s(vw), s(26)))
        pygame.draw.line(canvas, (55, 65, 90),
                         (0, hint_y - s(4)), (s(vw), hint_y - s(4)), 1)
        canvas.blit(fonts.DATA.render(hints, True, config.GRAY), (s(14), hint_y))

    # ── game over overlay ─────────────────────────────────────

    def draw_gameover(self):
        self.draw()
        g = self.game
        s = g._s
        overlay = pygame.Surface((s(config.WINDOW_WIDTH), s(config.WINDOW_HEIGHT)),
                                 pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        g.canvas.blit(overlay, (0, 0))
        if self.battle:
            w = self.battle.winner()
            txt = fonts.BIG.render(f"{fonts.team_name(w)} Wins!", True,
                                   fonts.team_color(w))
            g.canvas.blit(txt, txt.get_rect(
                center=(s(config.WINDOW_WIDTH) // 2,
                        s(config.WINDOW_HEIGHT) // 2 - s(20))))
        cx = s(config.WINDOW_WIDTH) // 2
        r = pygame.Rect(cx - s(100),
                        s(config.WINDOW_HEIGHT) // 2 + s(30),
                        s(200), s(48))
        draw_btn(g.canvas, r.x, r.y, r.w, r.h, "Play Again", config.GREEN, config.BLACK)
        return r

    # ── reset ─────────────────────────────────────────────────

    def reset(self):
        self.battle = None
        self.b_action = None; self.b_desc = ""
        self.b_path = None; self.b_target = None; self.b_log = []
        self._round_order = None
        self._ph = PH_IDLE; self._anim_unit = None
        self._popups = []; self._projectile = None; self._flash = None
        self._await_input = False
        self._cast_mode = False
        self._pending_unit = None
        self._legal_mask = None
        self._hover_cell = None
        self._await_spell_target = False
        self._selected_spell_slot = None
        self._spell_list = []
        self._spell_sel = 0
        self._actions_remaining = 1
        self._ai_cache = {}
        self.logger.reset()
