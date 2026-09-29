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
                                sl_threshold=0.5,
                                fls_threshold=0.3):
    """Returns a dictionary of only the colinear members that received section updates.

    - OD & THK are in cm, FY in kN/cm2.
    - Tubular members are grouped into OD segments: runs of tubular members connected
      through shared joints. Cones split the chain, so the tubulars on either side of a
      cone have no OD dependency on each other.
    - Tubular members are updated first:
        * OD reduction per segment, decided from the segment's tubular UCs, plus the end
          chords only where the segment reaches an extreme joint of the chain
        * own THK reduction, from own UCs plus braces at their joints
        * legs are outer-flushed (OD constant), other chains inner-flushed (ID shifts by od_red)
        * OD / THK >= 18 checked on tubular members; Class 2 only for members that are
          part of a joint (is_joint)
    - Cones are adjusted afterwards, from the final tubular sections:
        * large end (joint1) inner-flushed with its neighbour: OD_L = neighbour ID + 2 * cone THK
        * small end (joint2) outer-flushed with its neighbour: OD_S = neighbour OD
        * THK1 / THK2 = new THK of the tubular neighbour at joint1 / joint2
        * cone THK reduced from its own UCs (plus braces at its joints) only when below
          thresholds; otherwise left unchanged. Cones are never increased.
        * if an end has no tubular neighbour, it keeps its own ID (large end) / OD (small end)
          and its THK1 / THK2 is returned as None (leave unchanged)
        * a cone must not flip: new OD_L > new OD_S
    - new_FY = expected FY for the member's grade at its new THK.
    - Failures:
        * a failing tubular member drops its own segment's OD reduction
        * a failing cone first drops its own THK reduction, then the OD reduction of the
          segments on either side of it
        * if a failure remains with no OD reduction left to drop, no member is updated.
    - Returns only members whose section changed:
      tubulars -> new_OD, new_THK, new_FY
      cones    -> new_OD_L, new_OD_S, new_THK, new_THK1, new_THK2, new_FY
      (unused keys are None).
    """
    end_ids = set(chord_member_data.keys())
    MIN_THK = 1.5

    tubes = {m: d for m, d in colinear_members_data.items() if not d["is_cone"]}
    cones = {m: d for m, d in colinear_members_data.items() if d["is_cone"]}

    def val(x):
        return 0.0 if pd.isna(x) else x

    def ok(members):
        return all(val(uc) < sl_threshold and val(f) < fls_threshold for uc, f in members)

    def joints_of(member_id):
        return (member_id[:4], member_id[-4:])

    def neighbour_id(member_id, j):
        # Tubular chain member sharing joint j with member_id (None if there isn't one)
        return next((n for n in tubes if n != member_id and j in joints_of(n)), None)

    def fy_at(d, thk):
        # Expected FY (kN/cm2) for the member's grade at thickness thk (cm)
        return expected_FY(grade_from_group(d["group_id"]), thk)

    # --- 0. OD segments: tubular members connected through shared joints (cones split them) ---
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

    # --- 1. OD Reduction Evaluation, per segment (tubular members only) ---
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
            can_reduce_OD.append(min(
                compute_possible_reduction(tubes[m]["OD"], tubes[m]["THK"], strength_allowance,
                                           fatigue_allowance, "OD", tubes[m]["group_id"],
                                           tubes[m]["is_joint"])
                for m in seg
            ))
        else:
            can_reduce_OD.append(0.0)

    # --- 2. Individual Members (Local Thickness Reduction Evaluation, cones included) ---
    can_reduce_THK = {}
    for member_id, d in colinear_members_data.items():
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
                                           m_fatigue_allowance, "THK", d["group_id"], d["is_joint"])
                for od in ods
            )
        else:
            can_reduce_THK[member_id] = 0.0

    # --- 3. Final Verification Step ---
    def new_section(d, od_red, thk_red):
        # Tubular members: legs outer-flushed, others inner-flushed
        new_thk = max(d["THK"] - thk_red, MIN_THK)
        if is_leg:
            new_od = d["OD"]
        else:
            new_od = d["OD"] - 2 * d["THK"] - od_red + 2 * new_thk
        return {"new_OD": new_od, "new_OD_L": None, "new_OD_S": None,
                "new_THK": new_thk, "new_THK1": None, "new_THK2": None,
                "new_FY": fy_at(d, new_thk)}

    def cone_section(member_id, d, thk_red, tube_details):
        # Cones: large end (joint1) inner-flushed, small end (joint2) outer-flushed
        # with the neighbouring tubular members' new sections.
        new_thk = max(d["THK"] - thk_red, MIN_THK)
        n1 = neighbour_id(member_id, member_id[:4])
        n2 = neighbour_id(member_id, member_id[-4:])

        if n1:
            s1 = tube_details[n1]
            od_l = (s1["new_OD"] - 2 * s1["new_THK"]) + 2 * new_thk
            thk1 = s1["new_THK"]
        else:  # no tubular neighbour: keep the cone's own ID at the large end
            od_l = (d["OD_L"] - 2 * d["THK"]) + 2 * new_thk
            thk1 = None

        if n2:
            s2 = tube_details[n2]
            od_s = s2["new_OD"]
            thk2 = s2["new_THK"]
        else:  # no tubular neighbour: keep own OD at the small end
            od_s = d["OD_S"]
            thk2 = None

        return {"new_OD": None, "new_OD_L": od_l, "new_OD_S": od_s,
                "new_THK": new_thk, "new_THK1": thk1, "new_THK2": thk2,
                "new_FY": fy_at(d, new_thk)}

    def is_changed(member_id, sec):
        d = colinear_members_data[member_id]
        if not d["is_cone"]:
            return (sec["new_OD"], sec["new_THK"]) != (d["OD"], d["THK"])
        if (sec["new_OD_L"], sec["new_OD_S"], sec["new_THK"]) != (d["OD_L"], d["OD_S"], d["THK"]):
            return True
        # THK1 / THK2 change when the neighbouring tubular THK changed
        for key, j in (("new_THK1", member_id[:4]), ("new_THK2", member_id[-4:])):
            n = neighbour_id(member_id, j)
            if n and sec[key] != tubes[n]["THK"]:
                return True
        return False

    def is_valid(member_id, sec):
        d = colinear_members_data[member_id]
        j1, j2 = joints_of(member_id)
        if d["is_cone"]:
            # Cone must not flip: large end stays larger than small end
            if sec["new_OD_L"] <= sec["new_OD_S"]:
                return False
            od_at = {j1: sec["new_OD_L"], j2: sec["new_OD_S"]}
        else:
            od_at = {j1: sec["new_OD"], j2: sec["new_OD"]}
            d_t = sec["new_OD"] / sec["new_THK"]

            # Check 1a: OD / THK >= 18 (tubular members only)
            ratio_valid = d_t >= 18.0
            # Check 1b: section at least Class 2 (joint members only)
            if d["is_joint"]:
                eps = np.sqrt(23.5 / sec["new_FY"])
                is_class2 = d_t <= 70 * eps**2
            else:
                is_class2 = True

            if not (ratio_valid and is_class2):
                return False

        for j, od in od_at.items():
            if j in end_ids:
                # Check 2: at end joints, colinear OD must not exceed chord OD
                if any(od > c["OD"] for c in chord_member_data.get(j, []) if "OD" in c):
                    return False
            else:
                # Check 3: at middle joints, colinear OD must be >= attached brace OD
                if any(od < b["OD"] for b in brace_member_data.get(j, []) if "OD" in b):
                    return False
        return True

    def compute_verification(seg_red):
        """Returns (passed, details, failed_segments)."""
        details = {}

        # 3a. Tubular members first, each with its own segment's OD reduction
        for member_id, d in tubes.items():
            od_red = seg_red[seg_of[member_id]]
            thk_red = can_reduce_THK[member_id]
            sec = new_section(d, od_red, thk_red)
            if (od_red or thk_red) and not is_valid(member_id, sec):
                return False, {}, {seg_of[member_id]}
            details[member_id] = sec

        # 3b. Cones afterwards: ends and THK1 / THK2 follow the final tubular sections;
        #     drop the cone's own THK reduction if it fails
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

    # --- 4. Keep only members whose section actually changed ---
    return {member_id: sec for member_id, sec in details.items()
            if is_changed(member_id, sec)}


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

def apply_updated_sections(model_out, updated_members, member_zones, tol=0.01, fy_factor=1.0):
    """
    fy_factor: converts new_FY from expected_FY units to SACS GRUP units
               (e.g. 0.1 if expected_FY returns MPa and the model uses kN/cm2).
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

    def size_of(m):
        fy = float(m.FY)
        if m.is_cone:
            return ("CON", float(m.OD_L), float(m.OD_S), float(m.THK), *cone_end_thks(m), fy)
        return ("TUB", float(m.OD), float(m.THK), fy)

    def new_size_of(m, new):
        fy = float(m.FY) if new.get("new_FY") is None else round(float(new["new_FY"]) * fy_factor, 2)
        if m.is_cone:
            cur_thk1, cur_thk2 = cone_end_thks(m)
            thk1 = cur_thk1 if new["new_THK1"] is None else float(new["new_THK1"])
            thk2 = cur_thk2 if new["new_THK2"] is None else float(new["new_THK2"])
            return ("CON", float(new["new_OD_L"]), float(new["new_OD_S"]),
                    float(new["new_THK"]), thk1, thk2, fy)
        return ("TUB", float(new["new_OD"]), float(new["new_THK"]), fy)

    def find_group(zone, size):
        for (z, s), group_id in catalogue.items():
            if z == zone and s[0] == size[0] and all(abs(a - b) <= tol for a, b in zip(s[1:], size[1:])):
                return group_id
        return None

    # catalogue of what already exists: (zone, size) -> group_id
    catalogue = {}
    for member_id, zone in zone_by_id.items():
        m = model_out.members[member_id]
        catalogue.setdefault((zone, size_of(m)), m.group_id)

    existing_group_ids = set(model_out.groups)
    changes = {}   # member_id -> (old_group, new_group, created)

    for member_id, new in updated_members.items():
        m = model_out.members[member_id]
        zone = zone_by_id.get(member_id)
        if zone is None:
            continue
        size = new_size_of(m, new)
        old_group = m.group_id

        # a matching size (incl. FY) already exists in this zone: reuse its group
        group_id = find_group(zone, size)
        if group_id is not None:
            m.group_id = group_id
            changes[member_id] = (old_group, group_id, False)
            continue

        # otherwise clone the member's group and give the clone the new size and FY
        new_group_id = s.generate_unique_group(existing_group_ids, old_group)
        new_group = model_out.groups[old_group].clone_group(model_out, new_group_id)

        seg = new_group.segments[0]
        sec = model_out.sections[seg.section_id] if seg.section_id != "" else None

        if m.is_cone:
            _, od_l, od_s, thk, thk1, thk2, fy = size
            if sec:
                sec.OD_L, sec.OD_S, sec.THK = od_l, od_s, thk
                sec.THK1, sec.THK2 = thk1, thk2
            else:
                print(f"Cone {member_id}: no section, cone not written")
        else:
            _, od, thk, fy = size
            seg.OD, seg._THK = od, thk
            if sec:
                sec.OD, sec.THK = od, thk

        seg.FY = fy

        catalogue[(zone, size)] = new_group_id
        m.group_id = new_group_id
        changes[member_id] = (old_group, new_group_id, True)

    return model_out, changes




