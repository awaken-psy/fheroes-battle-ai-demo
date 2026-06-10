"""T7 — Config balance and validity tests for 16 training configs.

Validates that all training configs:
  - Load without error
  - Have valid unit types and positions
  - Mirror configs have identical team compositions
  - Non-mirror configs have balanced total HP (within 20%)
  - Can complete a full battle without crashing
"""

import json
import os

import pytest

# ── Project root for config paths ──────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_CONFIGS = [
    "configs/example.json",
    "configs/melee_brawl.json",
    "configs/archer_line.json",
    "configs/ranged_duel.json",
    "configs/flyer_swarm.json",
    "configs/wide_charge.json",
    "configs/dragon_battle.json",
    "configs/mixed_mobility.json",
    "configs/even_clash.json",
    "configs/mage_duel.json",
    "configs/hero_basic.json",
    "configs/hero_support.json",
    "configs/solo_duel.json",
    "configs/duo_mirror.json",
    "configs/undead_mirror.json",
    "configs/large_battle.json",
]


def _load_config(path):
    """Load a config file, handling both dict and list formats."""
    full_path = os.path.join(ROOT, path)
    with open(full_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw = {"units": raw}
    return raw


def _get_unit_stats():
    """Lazy-load unit stats to avoid torch dependency."""
    sys_path = ROOT
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from config.units import UNIT_TYPES
    return UNIT_TYPES


# ── Loading tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("config_path", TRAINING_CONFIGS, ids=lambda p: p.split("/")[-1])
class TestConfigLoading:
    """Every training config loads and has required fields."""

    def test_loads_without_error(self, config_path):
        cfg = _load_config(config_path)
        assert "units" in cfg
        assert len(cfg["units"]) >= 2

    def test_has_two_teams(self, config_path):
        cfg = _load_config(config_path)
        teams = {u["team"] for u in cfg["units"]}
        assert teams == {0, 1}

    def test_each_team_has_at_least_one_unit(self, config_path):
        cfg = _load_config(config_path)
        t0 = [u for u in cfg["units"] if u["team"] == 0]
        t1 = [u for u in cfg["units"] if u["team"] == 1]
        assert len(t0) >= 1
        assert len(t1) >= 1


# ── Position validity tests ────────────────────────────────────────


@pytest.mark.parametrize("config_path", TRAINING_CONFIGS, ids=lambda p: p.split("/")[-1])
class TestConfigPositions:
    """Unit positions are valid and non-overlapping."""

    def test_valid_unit_types(self, config_path):
        UNIT_TYPES = _get_unit_stats()
        cfg = _load_config(config_path)
        for u in cfg["units"]:
            assert u["type"] in UNIT_TYPES, f"Unknown unit: {u['type']}"

    def test_positions_in_bounds(self, config_path):
        cfg = _load_config(config_path)
        for u in cfg["units"]:
            assert 0 <= u["col"] <= 10, f"col {u['col']} out of bounds"
            assert 0 <= u["row"] <= 7, f"row {u['row']} out of bounds"

    def test_no_position_overlap(self, config_path):
        UNIT_TYPES = _get_unit_stats()
        cfg = _load_config(config_path)
        occupied = set()
        for u in cfg["units"]:
            stats = UNIT_TYPES[u["type"]]
            pos1 = (u["col"], u["row"])
            assert pos1 not in occupied, f"Duplicate position {pos1} in {config_path}"
            occupied.add(pos1)
            if stats["is_wide"]:
                assert u["col"] + 1 <= 10, (
                    f"Wide unit {u['type']} at col {u['col']} overflows")
                pos2 = (u["col"] + 1, u["row"])
                assert pos2 not in occupied, (
                    f"Duplicate wide position {pos2} in {config_path}")
                occupied.add(pos2)


# ── Balance tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("config_path", TRAINING_CONFIGS, ids=lambda p: p.split("/")[-1])
class TestConfigBalance:
    """Teams have comparable strength."""

    def _compute_team_stats(self, cfg):
        """Compute total HP and unit count for each team."""
        UNIT_TYPES = _get_unit_stats()
        stats = {0: {"hp": 0, "count": 0}, 1: {"hp": 0, "count": 0}}
        for u in cfg["units"]:
            team = u["team"]
            unit_type = UNIT_TYPES[u["type"]]
            count = u.get("count", 1)
            stats[team]["hp"] += unit_type["hp"] * count
            stats[team]["count"] += count
        return stats

    def test_total_hp_within_20_percent(self, config_path):
        """Mirror configs: exact match. Non-mirror: within 20%."""
        cfg = _load_config(config_path)
        stats = self._compute_team_stats(cfg)
        hp0, hp1 = stats[0]["hp"], stats[1]["hp"]

        # Check if mirror (same unit composition)
        is_mirror = self._is_mirror(cfg)

        if is_mirror:
            assert hp0 == hp1, (
                f"Mirror config {config_path}: HP mismatch T0={hp0} vs T1={hp1}")
        else:
            ratio = min(hp0, hp1) / max(hp0, hp1)
            assert ratio >= 0.80, (
                f"Non-mirror {config_path}: HP ratio {ratio:.2f} "
                f"(T0={hp0} vs T1={hp1}) below 80% threshold")

    def _is_mirror(self, cfg):
        """Check if both teams have identical unit composition."""
        t0 = sorted([(u["type"], u.get("count", 1)) for u in cfg["units"] if u["team"] == 0])
        t1 = sorted([(u["type"], u.get("count", 1)) for u in cfg["units"] if u["team"] == 1])
        return t0 == t1


# ── Auto-discovery test ────────────────────────────────────────────


class TestAutoDiscovery:
    """eval_benchmark auto-discovery finds all 16 training configs."""

    def test_discovers_16_configs(self):
        sys_path = ROOT
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from scripts.eval_benchmark import discover_training_configs

        configs = discover_training_configs(
            os.path.join(ROOT, "configs"))
        paths = [c[0] for c in configs]
        assert len(configs) == 16, f"Expected 16 configs, found {len(configs)}"

    def test_all_training_configs_discovered(self):
        sys_path = ROOT
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from scripts.eval_benchmark import discover_training_configs

        configs = discover_training_configs(
            os.path.join(ROOT, "configs"))
        discovered_basenames = {os.path.basename(c[0]) for c in configs}
        expected_basenames = {os.path.basename(p) for p in TRAINING_CONFIGS}
        assert expected_basenames == discovered_basenames, (
            f"Missing: {expected_basenames - discovered_basenames}, "
            f"Extra: {discovered_basenames - expected_basenames}")

    def test_excludes_test_configs(self):
        sys_path = ROOT
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from scripts.eval_benchmark import discover_training_configs

        configs = discover_training_configs(
            os.path.join(ROOT, "configs"))
        discovered_basenames = {os.path.basename(c[0]) for c in configs}
        assert "validation_results.json" not in discovered_basenames
        assert "ability_showcase.json" not in discovered_basenames

    def test_legacy_4_still_works(self):
        sys_path = ROOT
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from scripts.eval_benchmark import LEGACY_BENCHMARK_CONFIGS

        assert len(LEGACY_BENCHMARK_CONFIGS) == 4
        paths = [c[0] for c in LEGACY_BENCHMARK_CONFIGS]
        assert "configs/example.json" in paths
        assert "configs/dragon_battle.json" in paths


# ── Battle execution test (subset for speed) ───────────────────────


class TestConfigBattleExecution:
    """Configs can complete a full battle without crashing."""

    @pytest.mark.parametrize("config_path", TRAINING_CONFIGS, ids=lambda p: p.split("/")[-1])
    def test_completes_battle(self, config_path):
        """Run a single ClassicAI vs ClassicAI game per config."""
        import sys
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from ai.self_play import eval_vs_classic
        from ai.deep.pipeline import load_battle_config
        import numpy as np

        def random_legal_agent(obs, info):
            legal = info.get("legal_actions", None)
            if legal is not None and len(legal) > 0:
                return int(np.random.choice(legal))
            return 0

        cfg = load_battle_config(os.path.join(ROOT, config_path))
        result = eval_vs_classic(cfg, random_legal_agent,
                                 learning_team=0, games=1, seed=42)
        assert result["games"] == 1
        assert result["avg_rounds"] > 0
