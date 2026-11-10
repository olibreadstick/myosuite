# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward terms for high-level boxing skill selection (kinematic skill playback).

Pure NumPy functions over a window of frames. A hit is a moving fist entering
the opponent's head or body zone, so a punch scores once, not on every frame
the glove rests on the target.
"""

from __future__ import annotations

import numpy as np


def fist_hit_damage(
    fists: np.ndarray,
    fist_speed: np.ndarray,
    head: np.ndarray,
    body: np.ndarray,
    was_in_zone: np.ndarray,
    head_radius: float,
    body_radius: float,
    w_head: float,
    w_body: float,
    min_speed: float,
    full_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Damage of fists entering the opponent's scoring zones.

    A hit is a stretch of frames with a fist inside a zone. Its damage is the
    zone weight times the closest proximity ``1 - d / radius`` reached in the
    stretch, scaled by the fist speed on entry from 0 at ``min_speed`` to 1 at
    ``full_speed``, and is booked on the entry frame. A stretch that started
    before the window (``was_in_zone``) scores nothing.

    Args:
        fists: ``(T, 2, 3)`` attacker right/left fist positions.
        fist_speed: ``(T, 2)`` fist speeds (m/s).
        head: ``(T, 3)`` defender head-zone positions.
        body: ``(T, 3)`` defender body-zone positions.
        was_in_zone: ``(2,)`` whether each fist was in a zone before the window.
        head_radius: Head zone radius (m).
        body_radius: Body zone radius (m).
        w_head: Head hit weight.
        w_body: Body hit weight.
        min_speed: Entry speed below which a hit does no damage (m/s).
        full_speed: Entry speed of full damage (m/s).

    Returns:
        ``(damage, head_hit, in_zone_last)``: per-frame damage ``(T,)``, per-frame
        flag of a damaging head hit ``(T,)`` and each fist's zone state at the
        last frame ``(2,)``.
    """
    d_head = np.linalg.norm(fists - head[:, None], axis=-1)
    d_body = np.linalg.norm(fists - body[:, None], axis=-1)
    head_dmg = w_head * np.clip(1.0 - d_head / head_radius, 0.0, None)
    body_dmg = w_body * np.clip(1.0 - d_body / body_radius, 0.0, None)
    in_zone = (head_dmg > 0) | (body_dmg > 0)
    speed = np.clip((fist_speed - min_speed) / (full_speed - min_speed), 0.0, 1.0)
    damage = np.zeros(len(fists))
    head_hit = np.zeros(len(fists), dtype=bool)
    for f in range(fists.shape[1]):
        edges = np.diff(np.r_[int(was_in_zone[f]), in_zone[:, f].astype(int), 0])
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for a, b in zip(starts, stops[-len(starts) :] if len(starts) else []):
            h, bd = head_dmg[a:b, f].max(), body_dmg[a:b, f].max()
            dmg = max(h, bd) * speed[a, f]
            damage[a] += dmg
            head_hit[a] |= bool(dmg > 0 and h >= bd)
    return damage, head_hit, in_zone[-1]


def in_range(distance: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1 on frames whose boxer separation is within ``[lo, hi]`` m, else 0."""
    return ((distance >= lo) & (distance <= hi)).astype(np.float64)


def interrupt_mask(
    damage: np.ndarray,
    punching: np.ndarray,
    interrupted: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale down the rest of a punch once its thrower is hit.

    A boxer hit while punching lands only ``factor`` of its damage from the
    frame after the hit to the end of that punch. Hits on the same frame both
    land (a trade). Interruptions are resolved in time order, so a punch cut
    short cannot interrupt the opponent with a hit it no longer lands.

    Args:
        damage: ``(2, T)`` damage each boxer deals per frame.
        punching: ``(2,)`` whether each boxer's current skill is a punch.
        interrupted: ``(2,)`` whether that punch was already interrupted.
        factor: Damage scale of an interrupted punch.

    Returns:
        ``(mask, interrupted)``: per-frame damage scale ``(2, T)`` and each
        boxer's interrupted state at the end of the window ``(2,)``.
    """
    n = damage.shape[1]
    base = np.where(interrupted, 0, n)
    start = base.copy()
    frames = np.arange(n)
    for _ in range(4):  # two boxers: settles within a few passes
        mask = np.where(frames >= start[:, None], factor, 1.0)
        new = base.copy()
        for victim in (0, 1):
            hits = np.flatnonzero(damage[1 - victim] * mask[1 - victim] > 0)
            if punching[victim] and hits.size:
                new[victim] = min(new[victim], hits[0] + 1)
        if np.array_equal(new, start):
            break
        start = new
    mask = np.where(frames >= start[:, None], factor, 1.0)
    return mask, start < n
