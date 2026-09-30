from SACS_API import *
import math
import sacs_extension as s
import numpy as np
import pandas as pd
import glob
from rich import print
import sys, string, os  # systems parameters / string formatting / operating system functionality
import subprocess # creates processes
import sqlite3
import fileinput # text iteration functions
from scipy.spatial import distance
import re
import xlwings as xw
import time
from tabulate import tabulate
import copy


start_time = time.time()
avoid = pd.read_csv("avoid_groups.csv", dtype=str)
UPPER_SPLASH = 8.2
LOWER_SPLASH = -4.3

FY_TABLE = {
    "S355": [
        (0.0, 1.6, 355),
        (1.6, 4.0, 345),
        (4.0, 6.3, 335),
        (6.3, 8.0, 325),
        (8.0, 10.0, 315),
        (10.0, 12.0, 295),
    ],
    "S420": [
        (0.0, 1.6, 420),
        (1.6, 2.5, 400),
        (2.5, 4.0, 390),
        (4.0, 6.3, 380),
        (6.3, 8.0, 370),
        (8.0, 10.0, 360),
        (10.0, 12.0, 340),
    ],
    "S460": [
        (0.0, 1.6, 460),
        (1.6, 2.5, 440),
        (2.5, 4.0, 420),
        (4.0, 6.3, 415),
        (6.3, 8.0, 405),
        (8.0, 10.0, 400),
        (10.0, 12.0, 400),
    ]
}

def expected_FY(grade, thickness):
    """Expected FY in kN/cm2 for a grade at a thickness given in cm."""
    ranges = FY_TABLE[grade]
    thickness = float(thickness)
    for tmin, tmax, fy in ranges:
        if tmin < thickness <= tmax:
            return fy / 10
    return None  # thickness out of range

materials_df = pd.read_csv("group_materials.csv")
MATERIAL_LOOKUP = dict(zip(materials_df["group_id"], materials_df["material"]))

def grade_from_group(group_id):
    g = group_id.upper()
    # Check longest prefixes first (e.g. "WW" before "W")
    for prefix in sorted(MATERIAL_LOOKUP.keys(), key=len, reverse=True):
        if g.startswith(prefix):
            return MATERIAL_LOOKUP[prefix]
    # Default if no prefix matches
    return "S460"

def expected_FY(grade, thickness):
    ranges = FY_TABLE[grade]
    thickness = float(thickness)
    for tmin, tmax, fy in ranges:
        if tmin < thickness <= tmax:
            return fy/10
    return None  # thickness out of range


def compute_possible_reduction(OD, THK, strength_allowance, fatigue_allowance, target, group_id,
                               is_joint, m=3):
    """
    Estimate the max reduction (in OD or THK, per `target`) that keeps the
    member within its strength and fatigue allowances, holding the other
    dimension fixed.

    Governing case = whichever gives the larger A_new (smaller reduction),
    since exceeding it would violate the other criterion.

    OD is floored at THK * 18 to preserve the OD/THK >= 18 ratio.
    THK is floored at 1.5 cm. If the member is part of a joint (is_joint=True),
    THK is also limited so the section stays at least Class 2:
        OD / THK <= 70 * eps^2,  eps = sqrt(23.5 / expected_FY(grade, THK))  [FY in kN/cm2]
    (expected FY depends on thickness, so the check is made at the reduced THK).

    OD reduction is rounded down to the nearest 1 cm (10 mm).
    THK reduction is rounded down to the nearest 0.5 cm (5 mm).

    target: "OD" or "THK" - which dimension's reduction to compute and return.
    group_id: used to get the steel grade for the Class 2 limit.
    is_joint: True if the member is part of a joint -> Class 2 limit applies.
    """
    A_old = math.pi * THK * (OD - THK)  # exact annulus area
    strength_factor = (1 - strength_allowance) ** (1 / 4)
    fatigue_factor = (1 - fatigue_allowance) ** (1 / 10)
    A_new_strength = A_old * strength_factor
    A_new_fatigue = A_old * fatigue_factor

    A_new = min(max(A_new_strength, A_new_fatigue), A_old)

    if target == "OD":
        # A = pi * THK * (OD - THK)  =>  OD_new = A_new / (pi * THK) + THK
        OD_new = A_new / (math.pi * THK) + THK

        min_OD = THK * 18.0  # preserves OD/THK >= 18 at this THK
        OD_new = max(OD_new, min_OD)

        raw_reduction = max(0.0, OD - OD_new)
        return math.floor(raw_reduction / 1.0) * 1.0

    elif target == "THK":
        MIN_THK = 1.5
        # A = pi * (OD*THK - THK^2)  =>  pi*THK_new^2 - pi*OD*THK_new + A_new = 0
        a, b, c = math.pi, -math.pi * OD, A_new
        disc = b**2 - 4 * a * c
        if disc < 0:
            THK_new = THK  # no real solution -> no reduction
        else:
            THK_new = (-b - math.sqrt(disc)) / (2 * a)  # smaller root = physical one

        THK_new = max(THK_new, MIN_THK)

        raw_reduction = max(0.0, THK - THK_new)
        reduction = math.floor(raw_reduction / 0.5) * 0.5

        # Class 2 limit (joint members only): step the reduction back until the section complies
        if is_joint:
            grade = grade_from_group(group_id)

            def is_class_2(od, thk):
                exp_fy = expected_FY(grade, thk)
                eps = np.sqrt(23.5 / exp_fy)
                return od / thk <= 70 * eps**2

            while reduction > 0 and not is_class_2(OD, THK - reduction):
                reduction -= 0.5

        return reduction

    else:
        raise ValueError(f"target must be 'OD' or 'THK', got {target!r}")
    

def get_final_colinear_sections(colinear_members_data, brace_member_data, chord_member_data,
                                is_leg,
                                chain_joint_ids=None,
                                sl_threshold=0.5,
                                fls_threshold=0.3,
                                cone_log=None,
                                ring_log=None,
                                cone_length=1.0,          # m
                                min_cone_reduction=0.20,  # fraction of end member OD
                                clear_factor=0.75,        # x brace/chord OD at the far joint
                                clear_allowance=0.20,     # m
                                min_cone_spacing=2.0,     # m
                                beta_min=0.20,            # OD_end / OD_chord floor at every chain end
                                ring_sl_limit=0.70,       # leg ends: strength/lift UC gateway, stub AND leg
                                ring_fls_trigger=1.0,     # leg ends WITHOUT rings: estimated FLS cap
                                fls_cap=10.0,             # leg ends WITH rings: estimated FLS cap
                                fls_exponent=10,          # FLS_new = FLS_old * (A_old / A_new) ** fls_exponent
                                ring_beta_min=0.26,       # IRS: OD_end / OD_leg floor (Smedley)
                                ring_beta_max=0.80,       # IRS: OD_end / OD_leg upper limit (Smedley)
                                min_ring_reduction=0.20,  # IRS only if end OD drops by this fraction
                                cone_min_ratio=1.05):     # a kept / inserted cone must have OD_L >= ratio * OD_S
    """Returns a dictionary of the colinear members that received section updates.

    Sizing:
    - OD & THK in cm, FY in kN/cm2, lengths in m. ODs in whole cm (10 mm), THK in 0.5 cm.
    - Tubular members grouped into OD segments split by cones; OD reduction per segment,
      THK reduction per member; legs outer-flushed, braces inner-flushed.
    - Area rule (compute_possible_reduction): strength and fatigue each capped at an
      estimated UC of 1.0 (exponents 4 and 10).
    - OD / THK >= 18 on tubulars; Class 2 only for joint members.
    - At chain end joints: brace OD <= chord OD and OD / chord OD >= beta_min (floor).
    - Cones follow the final tubular sections (large end at joint1 inner-flushed,
      small end at joint2 outer-flushed), THK1 / THK2 from the neighbours, no flipping.

    Leg ends of brace chains (stub reduction, with or without IRS):
    - Stub = brace members from the leg joint up to the cone belonging to that end
      (large end facing the leg); the end member alone if there is no such cone.
    - Gateway: every stub member AND the leg at the joint have UC_SL < ring_sl_limit.
    - The middle is sized without the stub (and its cone); the stub then reduces towards
      the middle's ID. Two sizes are worked out, with the same rules as the standard sizing:
        * without rings: strength via the area rule (highest UC_SL of stub and leg, cap 1.0),
          fatigue cap ring_fls_trigger (stub and leg), beta >= beta_min
        * with rings:    same strength, fatigue cap fls_cap, beta >= ring_beta_min
      Both also floored by the cone step (OD_L >= cone_min_ratio * OD_S) when a cone stays,
      and the end OD rounded UP to a whole cm (all floors are minimums).
    - Rings are used when the ring size goes further than the ring-less one and passes:
      final OD_end / OD_leg <= ring_beta_max, end OD reduction >= min_ring_reduction.
      Otherwise the ring-less size is used ("reduced_no_IRS"); if neither reduces, the end
      falls back to the normal behaviour.
    - Stub reaches the middle ID -> cone removed; otherwise cone kept / inserted.
    - All leg ends considered go to ring_log.

    Cone insertion (other brace chain ends):
    - Candidate: a chain end member fails the thresholds. Cone of cone_length cut from the
      next member at the joint after the end member; no braces there; clearance and spacing
      checks; remainder OD reduction >= min_cone_reduction * end OD.
    - Returned as "insert_cone" rows keyed by the next member. Candidates go to cone_log.
    """
    end_ids = set(chord_member_data.keys())
    MIN_THK = 1.5

    def val(x):
        return 0.0 if x is None or pd.isna(x) else x

    def ok(members):
        return all(val(uc) < sl_threshold and val(f) < fls_threshold for uc, f in members)

    def joints_of(member_id):
        return (member_id[:4], member_id[-4:])

    def fy_at(d, thk):
        return expected_FY(grade_from_group(d["group_id"]), thk)

    def area(od, thk):
        return math.pi * thk * (od - thk)

    def max_chord_od(j):
        return max((c["OD"] for c in chord_member_data.get(j, []) if c.get("OD")), default=None)

    def tube_sec(od, thk, fy):
        return {"new_OD": od, "new_OD_L": None, "new_OD_S": None,
                "new_THK": thk, "new_THK1": None, "new_THK2": None, "new_FY": fy}

    def cone_sec(od_l, od_s, thk, thk1, thk2, fy):
        return {"new_OD": None, "new_OD_L": od_l, "new_OD_S": od_s,
                "new_THK": thk, "new_THK1": thk1, "new_THK2": thk2, "new_FY": fy}

    # =====================================================================
    # Sizing of one chain (data may contain virtual cone / remainder members)
    # =====================================================================
    def solve(data):
        tubes = {m: d for m, d in data.items() if not d["is_cone"]}
        cones = {m: d for m, d in data.items() if d["is_cone"]}

        def neighbour_id(member_id, j):
            return next((n for n in tubes if n != member_id and j in joints_of(n)), None)

        # --- 0. OD segments: tubular members connected through shared joints ---
        seg_of = {}
        segments = []
        for start in tubes:
            if start in seg_of:
                continue
            idx = len(segments)
            seg_of[start] = idx
            seg, stack = [], [start]
            while stack:
                cur = stack.pop()
                seg.append(cur)
                for j in joints_of(cur):
                    for n in tubes:
                        if n not in seg_of and j in joints_of(n):
                            seg_of[n] = idx
                            stack.append(n)
            segments.append(seg)

        # --- 1. OD reduction per segment ---
        can_reduce_OD = []
        for seg in segments:
            if is_leg or any(tubes[m]["skip_od"] for m in seg):
                can_reduce_OD.append(0.0)
                continue

            seg_joints = {j for m in seg for j in joints_of(m)}
            chain = [(tubes[m]["UC_max"], tubes[m]["UC_FLS_max"]) for m in seg]
            end_chords = [(c["chord_UCmax"], c["chord_FLS_UCmax"])
                          for j in seg_joints & end_ids for c in chord_member_data.get(j, [])]

            if ok(chain) and ok(end_chords):
                group = chain + end_chords
                strength_allowance = 1.0 - max(val(uc) for uc, _ in group)
                fatigue_allowance = 1.0 - max(val(f) for _, f in group)
                red = min(
                    compute_possible_reduction(tubes[m]["OD"], tubes[m]["THK"], strength_allowance,
                                               fatigue_allowance, "OD", tubes[m]["group_id"],
                                               tubes[m]["is_joint"])
                    for m in seg
                )
                # beta floor at chain ends: OD_end / OD_chord >= beta_min
                for m in seg:
                    for j in set(joints_of(m)) & end_ids:
                        dmax = max_chord_od(j)
                        if dmax:
                            red = min(red, max(0.0, math.floor(tubes[m]["OD"] - beta_min * dmax)))
                can_reduce_OD.append(red)
            else:
                can_reduce_OD.append(0.0)

        # --- 2. THK reduction per member (cones included) ---
        can_reduce_THK = {}
        for member_id, d in data.items():
            if d["skip_thk"]:
                can_reduce_THK[member_id] = 0.0
                continue
            joints = set(joints_of(member_id))
            members = [(d["UC_max"], d["UC_FLS_max"])]
            members += [(b["brace_UCmax"], b["brace_FLS_UCmax"])
                        for j in joints for b in brace_member_data.get(j, [])]

            if ok(members):
                m_strength_allowance = 1.0 - max(val(uc) for uc, _ in members)
                m_fatigue_allowance = 1.0 - max(val(f) for _, f in members)
                ods = [d["OD_L"], d["OD_S"]] if d["is_cone"] else [d["OD"]]
                can_reduce_THK[member_id] = min(
                    compute_possible_reduction(od, d["THK"], m_strength_allowance,
                                               m_fatigue_allowance, "THK", d["group_id"],
                                               d["is_joint"])
                    for od in ods
                )
            else:
                can_reduce_THK[member_id] = 0.0

        # --- 3. Verification ---
        def new_section(d, od_red, thk_red):
            new_thk = max(d["THK"] - thk_red, MIN_THK)
            if is_leg:
                new_od = d["OD"]
            else:
                new_od = d["OD"] - 2 * d["THK"] - od_red + 2 * new_thk
            return tube_sec(new_od, new_thk, fy_at(d, new_thk))

        def cone_section(member_id, d, thk_red, tube_details):
            new_thk = max(d["THK"] - thk_red, MIN_THK)
            n1 = neighbour_id(member_id, member_id[:4])
            n2 = neighbour_id(member_id, member_id[-4:])

            if n1:
                s1 = tube_details[n1]
                od_l = (s1["new_OD"] - 2 * s1["new_THK"]) + 2 * new_thk
                thk1 = s1["new_THK"]
            else:
                od_l = (d["OD_L"] - 2 * d["THK"]) + 2 * new_thk
                thk1 = None

            if n2:
                s2 = tube_details[n2]
                od_s = s2["new_OD"]
                thk2 = s2["new_THK"]
            else:
                od_s = d["OD_S"]
                thk2 = None

            return cone_sec(od_l, od_s, new_thk, thk1, thk2, fy_at(d, new_thk))

        def is_changed(member_id, sec):
            d = data[member_id]
            if not d["is_cone"]:
                return (sec["new_OD"], sec["new_THK"]) != (d["OD"], d["THK"])
            if (sec["new_OD_L"], sec["new_OD_S"], sec["new_THK"]) != (d["OD_L"], d["OD_S"], d["THK"]):
                return True
            for key, j in (("new_THK1", member_id[:4]), ("new_THK2", member_id[-4:])):
                n = neighbour_id(member_id, j)
                if n and sec[key] != tubes[n]["THK"]:
                    return True
            return False

        def is_valid(member_id, sec):
            d = data[member_id]
            j1, j2 = joints_of(member_id)
            if d["is_cone"]:
                if sec["new_OD_L"] <= sec["new_OD_S"]:
                    return False
                od_at = {j1: sec["new_OD_L"], j2: sec["new_OD_S"]}
            else:
                od_at = {j1: sec["new_OD"], j2: sec["new_OD"]}
                d_t = sec["new_OD"] / sec["new_THK"]
                ratio_valid = d_t >= 18.0
                if d["is_joint"]:
                    eps = np.sqrt(23.5 / sec["new_FY"])
                    is_class2 = d_t <= 70 * eps**2
                else:
                    is_class2 = True
                if not (ratio_valid and is_class2):
                    return False

            for j, od in od_at.items():
                if j in end_ids:
                    # brace OD <= chord OD, and OD / chord OD >= beta_min
                    if any(od > c["OD"] for c in chord_member_data.get(j, []) if "OD" in c):
                        return False
                    dmax = max_chord_od(j)
                    if dmax and od / dmax < beta_min - 1e-9:
                        return False
                else:
                    if any(od < b["OD"] for b in brace_member_data.get(j, []) if "OD" in b):
                        return False
            return True

        def compute_verification(seg_red):
            details = {}
            for member_id, d in tubes.items():
                od_red = seg_red[seg_of[member_id]]
                thk_red = can_reduce_THK[member_id]
                sec = new_section(d, od_red, thk_red)
                if (od_red or thk_red) and not is_valid(member_id, sec):
                    return False, {}, {seg_of[member_id]}
                details[member_id] = sec

            tube_details = dict(details)
            for member_id, d in cones.items():
                for thk_red in (can_reduce_THK[member_id], 0.0):
                    sec = cone_section(member_id, d, thk_red, tube_details)
                    if not is_changed(member_id, sec) or is_valid(member_id, sec):
                        break
                else:
                    adjacent = {seg_of[n] for n in (neighbour_id(member_id, j)
                                                    for j in joints_of(member_id)) if n}
                    return False, {}, adjacent
                details[member_id] = sec

            return True, details, set()

        seg_red = list(can_reduce_OD)
        while True:
            passed, details, failed = compute_verification(seg_red)
            if passed:
                break
            to_drop = {i for i in failed if seg_red[i] > 0}
            if not to_drop:
                return {}
            for i in to_drop:
                seg_red[i] = 0.0

        return {member_id: sec for member_id, sec in details.items()
                if is_changed(member_id, sec)}

    def as_resize(result):
        return {m: {**sec, "action": "resize"} for m, sec in result.items()}

    # =====================================================================
    # Base sizing
    # =====================================================================
    base = solve(colinear_members_data)

    if is_leg or not chain_joint_ids or len(chain_joint_ids) < 3:
        return as_resize(base)

    data0 = colinear_members_data
    ids = list(chain_joint_ids)

    chain_members = []
    for a, b in zip(ids, ids[1:]):
        mid = next((m for m in data0 if set(joints_of(m)) == {a, b}), None)
        if mid is None:
            return as_resize(base)
        chain_members.append(mid)

    pos = {ids[0]: 0.0}
    for (a, b), mid in zip(zip(ids, ids[1:]), chain_members):
        pos[b] = pos[a] + data0[mid]["length"]

    def other_joint(member_id, j):
        a, b = joints_of(member_id)
        return b if a == j else a

    def span_of(a, b):
        return (min(pos[a], pos[b]), max(pos[a], pos[b]))

    def gap(s1, s2):
        return max(s2[0] - s1[1], s1[0] - s2[1])

    chain_name = f"{ids[0]}..{ids[-1]}"

    # =====================================================================
    # Cone insertion geometry check
    # =====================================================================
    def cone_geometry(end_id, e, data_base, taken_spans):
        start = other_joint(end_id, e)
        info = {"cone_start_joint": start, "next_member": None, "next_length": None}

        if start in end_ids:
            return None, "chain has a single member", info
        if brace_member_data.get(start):
            return None, "braces at cone start joint", info

        nexts = [m for m in data_base if start in joints_of(m) and m != end_id]
        if len(nexts) != 1:
            return None, "no single next member", info
        next_id = nexts[0]
        nd = data_base[next_id]
        info.update({"next_member": next_id, "next_length": nd["length"]})

        if nd["is_cone"]:
            return None, "next member is a cone", info
        if nd["skip_od"]:
            return None, "next member OD on avoid list", info

        far = other_joint(next_id, start)
        ods = [b["OD"] for b in brace_member_data.get(far, []) if b.get("OD")]
        ods += [c["OD"] for c in chord_member_data.get(far, []) if c.get("OD")]
        required = (clear_factor * max(ods) / 100 if ods else 0.0) + clear_allowance
        remainder = nd["length"] - cone_length
        if remainder < required:
            return None, (f"next member too short: {remainder:.2f} m after cone, "
                          f"{required:.2f} m needed"), info

        direction = 1.0 if pos[far] > pos[start] else -1.0
        span = tuple(sorted((pos[start], pos[start] + direction * cone_length)))
        if any(gap(span, o) < min_cone_spacing for o in taken_spans):
            return None, f"closer than {min_cone_spacing} m to another cone", info

        return {"end_id": end_id, "next_id": next_id, "start": start, "far": far,
                "span": span}, "", info

    # =====================================================================
    # 5. Leg end candidates (gateway)
    # =====================================================================
    ring_ends = {}
    for side, e in ((0, ids[0]), (1, ids[-1])):
        leg_chords = [c for c in chord_member_data.get(e, []) if c.get("is_leg")]
        leg_ods = [c["OD"] for c in leg_chords if c.get("OD")]
        if not leg_ods:
            continue

        seq = chain_members if side == 0 else chain_members[::-1]
        end_m = seq[0]
        ed = data0[end_m]

        rec = {"chain": chain_name, "leg_joint": e,
               "leg_members": " ".join(c["member_id"] for c in leg_chords),
               "leg_OD_min": min(leg_ods), "leg_OD_max": max(leg_ods),
               "leg_UC_SL": max((val(c.get("chord_UC_SL")) for c in leg_chords), default=0.0),
               "leg_FLS_UC": max((val(c.get("chord_FLS_UCmax")) for c in leg_chords), default=0.0),
               "end_member": end_m, "end_UC_SL": ed["UC_SL"],
               "end_FLS_UC": ed["UC_FLS_max"], "end_OD": ed["OD"],
               "OD_ratio_now": ed["OD"] / min(leg_ods) if ed["OD"] else None,
               "status": "skipped", "reason": ""}

        def ring_skip(reason):
            rec["reason"] = reason
            if ring_log is not None:
                ring_log.append(rec)

        if ed["is_cone"]:
            ring_skip("end member is a cone")
            continue

        k = next((i for i, m in enumerate(seq) if data0[m]["is_cone"]), None)
        if k is not None:
            # the cone belongs to this end only if its large end (joint1) faces this end
            cone_id = seq[k]
            joint_before = e if k == 0 else next(j for j in joints_of(cone_id)
                                                 if j in joints_of(seq[k - 1]))
            if cone_id[:4] != joint_before:
                k = None

        if k is None:
            stub, cone_m = [end_m], None
        else:
            stub, cone_m = seq[:k], seq[k]

        used = len(stub) + (1 if cone_m else 0)
        rec.update({"stub": " ".join(stub), "cone_member": cone_m,
                    "cone_type": "existing" if cone_m else "none"})

        if used >= len(chain_members):
            ring_skip("stub and cone cover the whole chain")
            continue
        if any(data0[m]["skip_od"] for m in stub):
            ring_skip("stub OD on avoid list")
            continue

        # strength UC gateway: stub AND leg
        too_high = [m for m in stub if not val(data0[m]["UC_SL"]) < ring_sl_limit]
        too_high += [c["member_id"] for c in leg_chords
                     if not val(c.get("chord_UC_SL")) < ring_sl_limit]
        if too_high:
            ring_skip(f"strength UC not below {ring_sl_limit}: {' '.join(too_high)}")
            continue

        if side == 1 and 0 in ring_ends and used + ring_ends[0]["used"] >= len(chain_members):
            ring_skip("overlaps the leg end at the other side")
            continue

        start = ids[len(stub)] if side == 0 else ids[-1 - len(stub)]
        boundary = ids[used] if side == 0 else ids[-1 - used]
        ring_ends[side] = {"end_joint": e, "stub": stub, "cone": cone_m, "used": used,
                           "start": start, "boundary": boundary,
                           "leg_od_min": min(leg_ods), "leg_od_max": max(leg_ods),
                           "leg_sl": rec["leg_UC_SL"], "leg_fls": rec["leg_FLS_UC"], "rec": rec}

    # =====================================================================
    # 6. Cone planning and sizing (ends that are not leg-end candidates)
    # =====================================================================
    def plan_and_size(data_base, ring_sides, pending_log):
        existing_cone_spans = [span_of(*joints_of(m)) for m, d in data_base.items() if d["is_cone"]]

        plans = []
        for side, e in ((0, ids[0]), (1, ids[-1])):
            if side in ring_sides:
                continue
            ends = [m for m in data_base if e in joints_of(m)]
            if len(ends) != 1:
                continue
            end_id = ends[0]
            ed = data_base[end_id]
            if ed["is_cone"] or ok([(ed["UC_max"], ed["UC_FLS_max"])]):
                continue

            rec = {"chain": chain_name, "end_joint": e, "end_member": end_id,
                   "end_UC": ed["UC_max"], "end_FLS_UC": ed["UC_FLS_max"], "end_OD": ed["OD"],
                   "status": "skipped", "reason": ""}

            plan, reason, info = cone_geometry(end_id, e, data_base,
                                               existing_cone_spans + [p["span"] for p in plans])
            rec.update(info)
            if plan is None:
                rec["reason"] = reason
                pending_log.append(rec)
                continue

            vj = f"~{side:03d}"
            plan.update({"rec": rec, "cone_id": f"{plan['start']}-{vj}",
                         "rem_id": f"{vj}-{plan['far']}"})
            plans.append(plan)

        result = None
        while plans:
            data = dict(data_base)
            for p in plans:
                nd = data_base[p["next_id"]]
                del data[p["next_id"]]
                data[p["cone_id"]] = {**nd, "is_cone": True, "OD": None,
                                      "OD_L": nd["OD"], "OD_S": nd["OD"],
                                      "length": cone_length, "is_joint": False}
                data[p["rem_id"]] = {**nd, "length": nd["length"] - cone_length}

            result = solve(data)

            dropped = []
            for p in plans:
                rem = result.get(p["rem_id"])
                reduction = data_base[p["next_id"]]["OD"] - rem["new_OD"] if rem else 0.0
                needed = min_cone_reduction * data_base[p["end_id"]]["OD"]
                p["reduction"] = reduction
                if reduction < needed or p["cone_id"] not in result:
                    p["rec"]["reason"] = (f"OD reduction {reduction:.1f} cm below "
                                          f"{needed:.1f} cm ({min_cone_reduction:.0%} of end OD)")
                    dropped.append(p)

            if not dropped:
                break
            for p in dropped:
                pending_log.append(p["rec"])
                plans.remove(p)

        if not plans:
            result = solve(data_base)

        return plans, result

    # =====================================================================
    # 7. Leg end sizing (after the middle has been sized without the stubs)
    # =====================================================================
    def size_ring_ends(result, data_base, plans):
        rows, failed = {}, {}
        taken_spans = [span_of(*joints_of(m)) for m, d in data_base.items() if d["is_cone"]]
        taken_spans += [p["span"] for p in plans]

        for side, r in ring_ends.items():
            rec = r["rec"]
            b = r["boundary"]
            end_m = r["stub"][0]
            end_d = data0[end_m]
            end_od = end_d["OD"]

            # middle member at the boundary joint: its new section, or its current one
            n = next((m for m, d in data_base.items()
                      if b in joints_of(m) and not d["is_cone"]), None)
            if n is None:
                failed[side] = "no tubular middle member at the boundary joint"
                continue
            nd = data_base[n]
            nsec = result.get(n) or tube_sec(nd["OD"], nd["THK"], fy_at(nd, nd["THK"]))
            target_id = nsec["new_OD"] - 2 * nsec["new_THK"]

            # --- strength floor (no ring credit): area rule, highest UC_SL of stub and leg ---
            sl_gov = max([val(data0[m]["UC_SL"]) for m in r["stub"]] + [r["leg_sl"]])
            strength_id = 0.0
            for m in r["stub"]:
                d = data0[m]
                allowed = compute_possible_reduction(d["OD"], d["THK"], 1.0 - sl_gov, 1.0,
                                                     "OD", d["group_id"], d["is_joint"])
                strength_id = max(strength_id, d["OD"] - allowed - 2 * d["THK"])

            # --- fatigue floor for a given cap: stub members and leg (via the end member) ---
            #   A_new >= A_old * (FLS_old / cap) ** (1 / fls_exponent);  ID = A / (pi * THK) - THK
            def fls_floor_id(cap):
                fid = 0.0
                for m in r["stub"]:
                    d = data0[m]
                    fls = val(d["UC_FLS_max"])
                    if fls > 0:
                        a_min = area(d["OD"], d["THK"]) * (fls / cap) ** (1 / fls_exponent)
                        fid = max(fid, a_min / (math.pi * d["THK"]) - d["THK"])
                if r["leg_fls"] > 0:
                    a_min = area(end_od, end_d["THK"]) * (r["leg_fls"] / cap) ** (1 / fls_exponent)
                    fid = max(fid, a_min / (math.pi * end_d["THK"]) - end_d["THK"])
                return fid

            def floor_id(beta):
                # ID that gives the end member OD_end / OD_leg = beta (largest leg OD)
                return beta * r["leg_od_max"] - 2 * end_d["THK"]

            def round_id(x):
                # stub reaching the middle keeps the middle's ID; otherwise round the
                # end member OD UP to a whole cm (all floors are minimums)
                if x <= target_id + 1e-6:
                    return target_id
                od = math.ceil(x + 2 * end_d["THK"] - 1e-6)
                return od - 2 * end_d["THK"]

            cone_thk = data0[r["cone"]]["THK"] if r["cone"] else nsec["new_THK"]

            def size_for(beta, cap):
                """Stub ID for a beta floor and fatigue cap; returns (new_id, governing)."""
                floors = {"middle_ID": target_id, "strength": strength_id,
                          "beta": floor_id(beta), "fatigue": fls_floor_id(cap)}
                governing = max(floors, key=floors.get)
                new_id = round_id(floors[governing])
                if new_id > target_id + 1e-6:
                    # a cone stays: keep a real step, OD_L >= cone_min_ratio * OD_S
                    cone_floor = cone_min_ratio * nsec["new_OD"] - 2 * cone_thk
                    if cone_floor > new_id:
                        new_id = round_id(cone_floor)
                        governing = "cone_step"
                return new_id, governing

            def stub_sizes(new_id):
                s_rows = {}
                for m in r["stub"]:
                    d = data0[m]
                    thk = d["THK"]
                    new_od = new_id + 2 * thk
                    new_fy = fy_at(d, thk)
                    d_t = new_od / thk
                    if d_t < 18.0:
                        return None, f"{m} D/t {d_t:.1f} < 18"
                    if d["is_joint"] and d_t > 70 * 23.5 / new_fy:
                        return None, f"{m} not Class 2 at D/t {d_t:.1f}"
                    for j in joints_of(m):
                        if j in end_ids:
                            if any(new_od > c["OD"] for c in chord_member_data.get(j, []) if c.get("OD")):
                                return None, f"{m} OD larger than chord at {j}"
                        elif any(new_od < bb["OD"] for bb in brace_member_data.get(j, []) if bb.get("OD")):
                            return None, f"{m} OD smaller than a brace at {j}"
                    s_rows[m] = {**tube_sec(new_od, thk, new_fy), "action": "resize"}
                return s_rows, None

            def fls_estimate(s_rows):
                fb = 0.0
                for m in r["stub"]:
                    d = data0[m]
                    factor = (area(d["OD"], d["THK"]) / area(s_rows[m]["new_OD"], d["THK"])) ** fls_exponent
                    fb = max(fb, val(d["UC_FLS_max"]) * factor)
                end_factor = (area(end_od, end_d["THK"]) /
                              area(s_rows[end_m]["new_OD"], end_d["THK"])) ** fls_exponent
                return fb, r["leg_fls"] * end_factor

            def build(new_id):
                """Stub rows plus the cone at this end. Returns (rows, reason, cone_action, cone_plan)."""
                s_rows, reason = stub_sizes(new_id)
                if reason:
                    return None, reason, None, None
                full = new_id <= target_id + 1e-6
                stub_thk = data0[r["stub"][-1]]["THK"]

                if full:
                    if r["cone"]:
                        cd = data0[r["cone"]]
                        c_od = target_id + 2 * cd["THK"]
                        if c_od / cd["THK"] < 18.0:
                            return None, f"{r['cone']} as a tube: D/t {c_od / cd['THK']:.1f} < 18", None, None
                        s_rows[r["cone"]] = {**tube_sec(c_od, cd["THK"], fy_at(cd, cd["THK"])),
                                             "action": "cone_to_tube"}
                        return s_rows, None, "existing cone converted to tube", None
                    return s_rows, None, "no cone needed", None

                if r["cone"]:
                    cd = data0[r["cone"]]
                    od_l = new_id + 2 * cd["THK"]
                    od_s = nsec["new_OD"]
                    if od_l <= od_s:
                        return None, f"{r['cone']} would flip ({od_l:.1f} <= {od_s:.1f} cm)", None, None
                    s_rows[r["cone"]] = {**cone_sec(od_l, od_s, cd["THK"], stub_thk,
                                                    nsec["new_THK"], fy_at(cd, cd["THK"])),
                                         "action": "resize"}
                    return s_rows, None, "existing cone kept (resized)", None

                plan, why, info = cone_geometry(end_m, r["end_joint"], data0, taken_spans)
                if plan is None:
                    return None, f"stub cannot reach middle ID and cone cannot be inserted: {why}", None, None
                c_thk = nsec["new_THK"]
                od_l = new_id + 2 * c_thk
                od_s = nsec["new_OD"]
                if od_l <= od_s:
                    return None, f"planned cone would flip ({od_l:.1f} <= {od_s:.1f} cm)", None, None
                cone_new = cone_sec(od_l, od_s, c_thk, stub_thk, nsec["new_THK"], fy_at(nd, c_thk))
                row = {**nsec, "action": "insert_cone",
                       "cone_start_joint": plan["start"], "cone_length": cone_length}
                row.update({"cone_" + key: v for key, v in cone_new.items()})
                s_rows[plan["next_id"]] = row
                plan.update({"info": info, "od_l": od_l, "od_s": od_s, "thk": c_thk})
                return s_rows, None, "cone inserted", plan

            # --- the two sizes ---
            id_plain, gov_plain = size_for(beta_min, ring_fls_trigger)     # without rings
            id_ring, gov_ring = size_for(ring_beta_min, fls_cap)            # with rings

            choice, ring_reason = None, None

            # rings: only if they allow a further reduction and pass the ring checks
            if id_ring < id_plain - 1e-6:
                rows_r, reason_r, act_r, plan_r = build(id_ring)
                if reason_r is None:
                    new_od_r = rows_r[end_m]["new_OD"]
                    red_r = end_od - new_od_r
                    beta_r = new_od_r / r["leg_od_min"]
                    if beta_r > ring_beta_max:
                        reason_r = f"OD_end / OD_leg = {beta_r:.2f} above {ring_beta_max}"
                    elif red_r < min_ring_reduction * end_od:
                        reason_r = (f"OD reduction {red_r:.1f} cm below "
                                    f"{min_ring_reduction * end_od:.1f} cm ({min_ring_reduction:.0%} of end OD)")
                if reason_r is None:
                    choice = ("rings", id_ring, gov_ring, rows_r, act_r, plan_r)
                else:
                    ring_reason = reason_r

            # without rings
            if choice is None:
                rows_p, reason_p, act_p, plan_p = build(id_plain)
                if reason_p is None and end_od - rows_p[end_m]["new_OD"] <= 0:
                    reason_p = "no reduction possible"
                if reason_p is None:
                    choice = ("reduced_no_IRS", id_plain, gov_plain, rows_p, act_p, plan_p)
                else:
                    failed[side] = reason_p + (f" (rings: {ring_reason})" if ring_reason else "")
                    continue

            status, new_id, governing, side_rows, cone_action, plan = choice

            if plan is not None:
                taken_spans.append(plan["span"])
                if cone_log is not None:
                    cone_log.append({"chain": chain_name, "end_joint": r["end_joint"],
                                     "end_member": end_m, "status": "inserted",
                                     "reason": "leg end: stub partly reduced",
                                     **plan["info"], "cone_OD_L": plan["od_l"],
                                     "cone_OD_S": plan["od_s"], "cone_THK": plan["thk"]})

            new_od = side_rows[end_m]["new_OD"]
            fls_brace, fls_leg = fls_estimate(side_rows)
            rec.update({"status": status, "reason": "",
                        "rings_rejected_because": ring_reason or "",
                        "UC_SL_governing": sl_gov, "size_governed_by": governing,
                        "middle_ID": target_id, "stub_new_ID": new_id, "end_new_OD": new_od,
                        "OD_ratio_final": new_od / r["leg_od_min"],
                        "reduction_cm": end_od - new_od,
                        "stub_reaches_middle_ID": new_id <= target_id + 1e-6,
                        "cone_action": cone_action,
                        "FLS_est_brace": fls_brace, "FLS_est_leg": fls_leg,
                        "end_UC_SL_estimated": val(end_d["UC_SL"]) * area(end_od, end_d["THK"])
                                               / area(new_od, end_d["THK"])})
            rows.update(side_rows)

        return rows, failed

    # =====================================================================
    # 8. Combine: take leg-end stubs out, plan cones elsewhere, size, then the stubs
    # =====================================================================
    while True:
        pending_cone_log = []
        removed = {m for r in ring_ends.values() for m in r["stub"] + ([r["cone"]] if r["cone"] else [])}
        data_ring = {m: d for m, d in data0.items() if m not in removed}

        plans, result = plan_and_size(data_ring, set(ring_ends), pending_cone_log)

        cone_log_before = len(cone_log) if cone_log is not None else 0
        ring_rows, failed = size_ring_ends(result, data_ring, plans)

        if not failed:
            break

        if cone_log is not None:
            del cone_log[cone_log_before:]
        for side, reason in failed.items():
            rec = ring_ends[side]["rec"]
            rec["reason"] = reason
            if ring_log is not None:
                ring_log.append(rec)
            del ring_ends[side]

    if cone_log is not None:
        cone_log.extend(pending_cone_log)
    if ring_log is not None:
        ring_log.extend(r["rec"] for r in ring_ends.values())

    # --- Output ---
    virtual = {p["cone_id"] for p in plans} | {p["rem_id"] for p in plans}
    out = {m: {**sec, "action": "resize"} for m, sec in result.items() if m not in virtual}

    for p in plans:
        rem = result[p["rem_id"]]
        cone = result[p["cone_id"]]
        row = {**rem, "action": "insert_cone",
               "cone_start_joint": p["start"], "cone_length": cone_length}
        row.update({"cone_" + key: v for key, v in cone.items()})
        out[p["next_id"]] = row

        p["rec"].update({"status": "inserted", "reason": "",
                         "remainder_new_OD": rem["new_OD"],
                         "reduction_cm": p["reduction"],
                         "cone_OD_L": cone["new_OD_L"], "cone_OD_S": cone["new_OD_S"],
                         "cone_THK": cone["new_THK"]})
        if cone_log is not None:
            cone_log.append(p["rec"])

    out.update(ring_rows)
    return out

ZONE_ORDER = ["below", "in", "above"]

def build_member_zones(model, upper, lower, csv_path="member_zones.csv",
                       correct_model=False):
    def classify(z1, z2):
        zmin, zmax = min(z1, z2), max(z1, z2)
        if zmax < lower:
            return "below", ""
        if zmin > upper:
            return "above", ""
        if zmin >= lower and zmax <= upper:
            return "in", ""
        return "in", "member straddles splash zone boundary"

    rows = []
    zone_by_key = {}
    for key, m in model.members.items():
        z1, z2 = m.coord1[2], m.coord2[2]
        zone, note = classify(z1, z2)
        zone_by_key[key] = zone
        rows.append({"member_id": m.Id, "group_id": m.group_id,
                     "z1": z1, "z2": z2, "zone": zone, "warning": note})
    member_zones = pd.DataFrame(rows)

    # group-level crossover check, copied back onto every member of the group
    zones_by_group = member_zones.groupby("group_id")["zone"].agg(lambda s: sorted(set(s)))
    crossing = zones_by_group[zones_by_group.map(len) > 1]

    def add_group_warning(row):
        parts = [row["warning"]] if row["warning"] else []
        if row["group_id"] in crossing.index:
            parts.append("group spans " + " + ".join(crossing[row["group_id"]]))
        return "; ".join(parts)

    member_zones["warning"] = member_zones.apply(add_group_warning, axis=1)

    # optional correction: make every group unique to one zone
    group_map = {}          # (old_group, zone) -> new_group
    model_out = model

    if correct_model:
        model_out = copy.deepcopy(model)

        # generate_unique_group adds each new id to this set as it goes
        existing_ids = set(model_out.groups)

        for old_group, zones in crossing.items():
            ordered = [z for z in ZONE_ORDER if z in zones]
            # the first zone keeps the original group id, the rest get clones
            for zone in ordered[1:]:
                new_id = s.generate_unique_group(existing_ids, old_group)
                model_out.groups[old_group].clone_group(model_out, new_id)
                group_map[(old_group, zone)] = new_id

        # point the affected members at their new groups
        for key, m in model_out.members.items():
            new_id = group_map.get((m.group_id, zone_by_key[key]))
            if new_id:
                m.group_id = new_id

    member_zones["new_group_id"] = [
        group_map.get((g, z), g)
        for g, z in zip(member_zones["group_id"], member_zones["zone"])
    ]

    if csv_path:
        member_zones.to_csv(csv_path, index=False)

    return member_zones, model_out, group_map


def build_section_zones(model, upper, lower, csv_path="section_zones.csv", correct_model=False):
    def classify(z1, z2):
        zmin, zmax = min(z1, z2), max(z1, z2)
        if zmax < lower:
            return "below", ""
        if zmin > upper:
            return "above", ""
        if zmin >= lower and zmax <= upper:
            return "in", ""
        return "in", "member straddles splash zone boundary"

    rows = []
    zone_by_key = {}
    for key, m in model.members.items():
        z1, z2 = m.coord1[2], m.coord2[2]
        zone, note = classify(z1, z2)
        zone_by_key[key] = zone
        rows.append({"member_id": m.Id, "group_id": m.group_id, "section_id": m.section_id,
                     "z1": z1, "z2": z2, "zone": zone, "warning": note})
    section_zones = pd.DataFrame(rows)

    # section-level crossover check, copied back onto every member using the section
    zones_by_section = section_zones.groupby("section_id")["zone"].agg(lambda s: sorted(set(s)))
    crossing = zones_by_section[zones_by_section.map(len) > 1]
    crossing = crossing[crossing.index != ""]

    def add_section_warning(row):
        parts = [row["warning"]] if row["warning"] else []
        if row["section_id"] in crossing.index:
            parts.append("section spans " + " + ".join(crossing[row["section_id"]]))
        return "; ".join(parts)

    section_zones["warning"] = section_zones.apply(add_section_warning, axis=1)

    # optional correction: make every section (and its group) unique to one zone
    group_map = {}          # (old_group, zone) -> new_group
    section_map = {}        # (old_group, zone, old_section) -> new_section
    new_ids = {}            # member_id -> (new_group_id, new_section_id)
    model_out = model

    if correct_model:
        model_out = copy.deepcopy(model)

        # the first zone (in below, in, above order) keeps the original section
        keep_zone = {sec: [z for z in ZONE_ORDER if z in zones][0]
                     for sec, zones in crossing.items()}

        # generate_unique_group adds each new group id to this set as it goes
        existing_group_ids = set(model_out.groups)

        for key, m in model_out.members.items():
            zone = zone_by_key[key]
            if m.section_id not in keep_zone or zone == keep_zone[m.section_id]:
                continue

            old_group, old_section = m.group_id, m.section_id

            # one cloned group (with its cloned sections) per old group and zone
            if (old_group, zone) not in group_map:
                new_group_id = s.generate_unique_group(existing_group_ids, old_group)
                old_g = model_out.groups[old_group]
                new_g = old_g.clone_group(model_out, new_group_id)   # also clones its sections
                group_map[(old_group, zone)] = new_group_id

                # record which new section replaced which old one
                for old_seg, new_seg in zip(old_g.segments, new_g.segments):
                    if old_seg.section_id != "":
                        section_map[(old_group, zone, old_seg.section_id)] = new_seg.section_id

            m.group_id = group_map[(old_group, zone)]
            m.section_id = section_map.get((old_group, zone, old_section), old_section)
            new_ids[m.Id] = (m.group_id, m.section_id)
            print(f"Member {m.Id} ({zone}): section {old_section} -> {m.section_id}, group {old_group} -> {m.group_id}")
            a = 1

    section_zones["new_group_id"] = [new_ids.get(mid, (g, s))[0]
                                     for mid, g, s in zip(section_zones["member_id"],
                                                          section_zones["group_id"],
                                                          section_zones["section_id"])]
    section_zones["new_section_id"] = [new_ids.get(mid, (g, s))[1]
                                       for mid, g, s in zip(section_zones["member_id"],
                                                            section_zones["group_id"],
                                                            section_zones["section_id"])]

    if csv_path:
        section_zones.to_csv(csv_path, index=False)

    return section_zones, model_out, section_map, group_map


def get_member_zones(model, upper, lower, z_max=21.5, csv_path="member_zones.csv"):
    rows = []
    for key, m in model.members.items():
        z1, z2 = m.coord1[2], m.coord2[2]
        zmin, zmax = min(z1, z2), max(z1, z2)

        if zmax > z_max or ((not m.is_tube) and (not m.is_cone)):
            continue

        if zmax < lower:
            zone = "below"
        elif zmin > upper:
            zone = "above"
        else:
            zone = "in"    # includes members straddling a boundary

        row = {"member_id": m.Id, "group_id": m.group_id, "section_id": m.section_id,
               "is_cone": m.is_cone,
               "OD": None if m.is_cone else m.OD,
               "THK": m.THK,
               "OD_L": None, "OD_S": None, "THK1": None, "THK2": None,
               "z1": z1, "z2": z2, "zone": zone}

        if m.is_cone:
            # THK1 / THK2: thickness of the tubulars towards joint1 / joint2
            sec = model.sections[m.section_id] if m.section_id != "" else None
            row.update({
                "OD_L": m.OD_L,
                "OD_S": m.OD_S,
                "THK1": sec.THK1 if sec is not None else None,
                "THK2": sec.THK2 if sec is not None else None,
            })

        rows.append(row)

    member_zones = pd.DataFrame(rows)

    if csv_path:
        member_zones.to_csv(csv_path, index=False)

    return member_zones

def apply_updated_sections(model_out, updated_members, member_zones, tol=0.01, fy_factor=1.0,
                           cone_log=None):
    """
    Applies the sizes from get_final_colinear_sections.
    - action "resize": reuse a matching group in the zone or clone and resize one.
    - action "insert_cone": split the member cone_length from cone_start_joint, make the
      piece at that joint a cone (large end at joint1) and resize the remainder.
    - action "cone_to_tube": turn an existing cone into a tube (ring ends).
    fy_factor: converts new_FY to SACS GRUP units (1.0 if already kN/cm2).
    """
    zone_by_id = dict(zip(member_zones["member_id"], member_zones["zone"]))

    def section_of(m):
        seg = model_out.groups[m.group_id].segments[0]
        return model_out.sections[seg.section_id] if seg.section_id != "" else None

    def cone_end_thks(m):
        # THK1 / THK2: thickness of the tubulars towards joint1 / joint2
        sec = section_of(m)
        if sec is not None:
            return float(sec.THK1), float(sec.THK2)
        return float(m.THK), float(m.THK)

    def to_fy(value, current):
        return float(current) if value is None or pd.isna(value) else round(float(value) * fy_factor, 2)

    def size_of(m):
        fy = float(m.FY)
        if m.is_cone:
            return ("CON", float(m.OD_L), float(m.OD_S), float(m.THK), *cone_end_thks(m), fy)
        return ("TUB", float(m.OD), float(m.THK), fy)

    def new_size_of(m, new):
        fy = to_fy(new.get("new_FY"), m.FY)
        if m.is_cone:
            cur_thk1, cur_thk2 = cone_end_thks(m)
            thk1 = cur_thk1 if new["new_THK1"] is None else float(new["new_THK1"])
            thk2 = cur_thk2 if new["new_THK2"] is None else float(new["new_THK2"])
            return ("CON", float(new["new_OD_L"]), float(new["new_OD_S"]),
                    float(new["new_THK"]), thk1, thk2, fy)
        return ("TUB", float(new["new_OD"]), float(new["new_THK"]), fy)

    def find_group(zone, size):
        for (z, sz), group_id in catalogue.items():
            if z == zone and sz[0] == size[0] and all(abs(a - b) <= tol for a, b in zip(sz[1:], size[1:])):
                return group_id
        return None

    # catalogue of what already exists: (zone, size) -> group_id
    catalogue = {}
    for member_id, zone in zone_by_id.items():
        m = model_out.members[member_id]
        catalogue.setdefault((zone, size_of(m)), m.group_id)

    existing_group_ids = set(model_out.groups)
    changes = {}   # member_id -> (old_group, new_group, created)

    def assign_size(m, zone, size):
        """Give member m the size: reuse a matching group in its zone or clone a new one."""
        old_group = m.group_id

        group_id = find_group(zone, size)
        if group_id is not None:
            m.group_id = group_id
            return old_group, group_id, False

        new_group_id = s.generate_unique_group(existing_group_ids, old_group)
        new_group = model_out.groups[old_group].clone_group(model_out, new_group_id)
        seg = new_group.segments[0]
        sec = model_out.sections[seg.section_id] if seg.section_id != "" else None

        if size[0] == "CON":
            _, od_l, od_s, thk, thk1, thk2, fy = size
            if sec is None or sec.stype != "CON":
                # group came from a tube: give it a new CON section
                sec = Section(section_id=model_out.make_unique_section_id("CONE00"), stype="CON",
                              OD_L=od_l, OD_S=od_s, THK=thk, THK1=thk1, THK2=thk2)
                model_out.add_section(sec)
                seg.section_id = sec.Id
                seg.OD, seg._THK = "", ""
            sec.OD_L, sec.OD_S, sec.THK = od_l, od_s, thk
            sec.THK1, sec.THK2 = thk1, thk2
        else:
            _, od, thk, fy = size
            if sec is not None and sec.stype == "CON":
                # group came from a cone: drop the CON section, define the tube on the GRUP card
                seg.section_id = ""
                sec = None
            seg.OD, seg._THK = od, thk
            if sec:
                sec.OD, sec.THK = od, thk

        seg.FY = fy
        catalogue[(zone, size)] = new_group_id
        m.group_id = new_group_id
        return old_group, new_group_id, True

    for member_id, new in updated_members.items():
        m = model_out.members.get(member_id)
        zone = zone_by_id.get(member_id)
        if m is None or zone is None:
            continue

        action = new.get("action", "resize")

        if action == "insert_cone":
            start = new["cone_start_joint"]
            cone_len = float(new["cone_length"])
            parent_j1 = m.joint1_id
            ratio = cone_len / m.length if parent_j1 == start else 1.0 - cone_len / m.length

            new_joint, m1, m2 = m.split_member_at_ratio(ratio)
            cone_m, rem_m = (m1, m2) if parent_j1 == start else (m2, m1)
            if cone_m.joint1_id != start:
                cone_m.invert()          # large end must be at joint1 (stub side)

            zone_by_id[cone_m.Id] = zone
            zone_by_id[rem_m.Id] = zone

            cone_thk = float(new["cone_new_THK"])
            thk1 = new.get("cone_new_THK1")
            thk2 = new.get("cone_new_THK2")
            cone_size = ("CON", float(new["cone_new_OD_L"]), float(new["cone_new_OD_S"]), cone_thk,
                         cone_thk if thk1 is None else float(thk1),
                         cone_thk if thk2 is None else float(thk2),
                         to_fy(new.get("cone_new_FY"), m.FY))
            rem_size = ("TUB", float(new["new_OD"]), float(new["new_THK"]),
                        to_fy(new.get("new_FY"), m.FY))

            changes[cone_m.Id] = assign_size(cone_m, zone, cone_size)
            changes[rem_m.Id] = assign_size(rem_m, zone, rem_size)

            if cone_log is not None:
                for rec in cone_log:
                    if rec.get("next_member") == member_id and rec.get("status") == "inserted":
                        rec.update({"new_joint": new_joint.Id,
                                    "cone_member": cone_m.Id, "cone_group": cone_m.group_id,
                                    "remainder_member": rem_m.Id, "remainder_group": rem_m.group_id})
            continue

        if action == "cone_to_tube":
            tube_size = ("TUB", float(new["new_OD"]), float(new["new_THK"]),
                         to_fy(new.get("new_FY"), m.FY))
            changes[member_id] = assign_size(m, zone, tube_size)
            continue

        changes[member_id] = assign_size(m, zone, new_size_of(m, new))

    return model_out, changes




