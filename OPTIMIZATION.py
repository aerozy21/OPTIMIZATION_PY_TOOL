from SACS_API import *
from AUX_FUNCTIONS import *
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

# *********************** MAIN CODE ***************
# Use a recursive pattern to search in all subdirectories
patterns = ["**/*sac.inp", "**/sacinp.*"]
SACSInputFile = None

for pattern in patterns:
    for path in glob.glob(pattern, recursive=True):
        if "OUT" in os.path.basename(path).upper():
            continue  # skip output models written by previous runs
        SACSInputFile = path
        print(f'Found file: {SACSInputFile}')
        break  # Stop after finding the first file for this pattern
    if SACSInputFile:  # If a file was found, stop searching altogether
        break
if SACSInputFile is None:
    print("No matching file found.")

model = SACSModel(SACSInputFile)


z_max = 21.5
MEMBER_ZONES_CSV = "member_zones.csv"

# DOUBLE CHECKS IF GROUP ZONES ARE UNIQUE TO EACH ZONE:
correct_model_zones = False
if correct_model_zones:
    member_zones, model_out, group_map = s.build_member_zones( model, UPPER_SPLASH, LOWER_SPLASH, correct_model=correct_model_zones)
    section_zones, model_out, section_map, group_map_II = s.build_section_zones( model_out, UPPER_SPLASH, LOWER_SPLASH, correct_model=correct_model_zones)
    model_out_path = model.path.lower().replace("sac", "OUT_sac")
    model_out.write_model(model.path, "OUT")

df = pd.read_csv("governing_UCs_df.csv")


member_zones = get_member_zones(model, UPPER_SPLASH, LOWER_SPLASH, z_max, csv_path=MEMBER_ZONES_CSV)

# Create a working copy
working_df = df.copy()
# Create UC lookups from the dataframe
uc_df = working_df.dropna(subset=["member_id"]).set_index("member_id")
uc_lookup = uc_df["UC_max"].to_dict()
uc_FLS_lookup = uc_df["Max_Fatigue_UC"].to_dict()
# Non-fatigue UC: governing of strength and lift
uc_SL_lookup = uc_df[["Max_Strength_UC", "Max_Lift_UC"]].max(axis=1).to_dict()

# Identify members with UC_max < 1
members_to_optimize = (working_df.loc[working_df["UC_max"] < 1, "member_id"].dropna().unique())
print(f"Members to optimize: {len(members_to_optimize)}")


processed_member_ids = set()

# Initialize dictionary to collect all updated members across loops
updated_members = {}
# Logs of cone insertion and ring candidates (accepted and skipped)
cone_log = []
ring_log = []

avoid = pd.read_csv("avoid_groups.csv", dtype=str)
avoid_od_initials  = tuple(avoid["OD"].dropna().str.strip().loc[lambda s: s != ""])
avoid_thk_initials = tuple(avoid["THK"].dropna().str.strip().loc[lambda s: s != ""])
# Loop through each member
for member_id in members_to_optimize:
    if member_id in processed_member_ids:
        continue

    member = model.members[member_id]
    if not member.is_tube:
        continue

    # groups whose initials are on the avoid list
    skip_od  = member.group_id.startswith(avoid_od_initials)
    skip_thk = member.group_id.startswith(avoid_thk_initials)
    # nothing left to optimize for this member
    if skip_od and skip_thk:
        continue

    is_leg = member.is_leg
    colinear_members = s.get_colinear_members(model, member, z_max=21.5, include_cones=True)[0]
    colinear_member_ids = [colinear_member.Id for colinear_member in colinear_members]
    print(colinear_member_ids)
    processed_member_ids.update(colinear_member_ids)


    colinear_joints, colinear_joint_ids = s.get_colinear_joints_in_order(colinear_members)
    print(colinear_joint_ids)
    
    colinear_members_data = {}
    for cm in colinear_members:
        j1, j2 = cm.Id[:4], cm.Id[-4:]

        colinear_members_data[cm.Id] = {
            "UC_max": uc_lookup.get(cm.Id),
            "UC_FLS_max": uc_FLS_lookup.get(cm.Id),
            "UC_SL": uc_SL_lookup.get(cm.Id),                 # strength / lift only
            "is_cone": cm.is_cone,
            "OD": None if cm.is_cone else cm.OD,
            "THK": cm.THK,                                  # a cone's own thickness
            "OD_L": cm.OD_L if cm.is_cone else None,
            "OD_S": cm.OD_S if cm.is_cone else None,
            "OD_THK": (max(cm.OD_L, cm.OD_S) if cm.is_cone else cm.OD) / cm.THK,
            "skip_od": cm.group_id.startswith(avoid_od_initials),
            "skip_thk": cm.group_id.startswith(avoid_thk_initials),
            "group_id": cm.group_id,
            "is_joint": s.is_joint(cm.joint1, z_max) or s.is_joint(cm.joint2, z_max),   # Class 2 required
            "length": cm.length,                                                          # m
        }

    # Dictionary: {joint_id: [list of brace data dictionaries]}
    brace_member_data = {}

    for joint in colinear_joints[1:-1]:
        brace_member_data[joint.Id] = []
        
        for attached_member in joint.AttachedMembers:
            # Skip the colinear members themselves
            if attached_member.Id in colinear_member_ids:
                continue
                
            # Append each attached brace's data to this joint's list
            brace_member_data[joint.Id].append({
                "member_id": attached_member.Id,
                "OD": attached_member.OD,
                "brace_UCmax": uc_lookup.get(attached_member.Id),
                "brace_FLS_UCmax": uc_FLS_lookup.get(attached_member.Id)
            })

    # Dictionary: {joint_id: [list of chord data dictionaries]}
    # Only the extreme (first and last) joints of the chain
    chord_member_data = {}

    for joint in (colinear_joints[0], colinear_joints[-1]):
        chord_objects = s.get_joint_members(joint)["chords"]

        chord_member_data[joint.Id] = [
            {
                "member_id": chord.Id,
                "OD": chord.OD,
                "THK": chord.THK,
                "is_leg": chord.is_leg,
                "chord_UCmax": uc_lookup.get(chord.Id),
                "chord_UC_SL": uc_SL_lookup.get(chord.Id),       # strength / lift only
                "chord_FLS_UCmax": uc_FLS_lookup.get(chord.Id)
            }
            for chord in chord_objects
            if chord.Id not in colinear_member_ids
        ]
    final_colinear = get_final_colinear_sections(colinear_members_data, brace_member_data,
                                                 chord_member_data, is_leg,
                                                 chain_joint_ids=colinear_joint_ids,
                                                 cone_log=cone_log,
                                                 ring_log=ring_log)
    print("COLINEAR DATA:")
    print(colinear_members_data)
    print("BRACE MEMBER DATA:")
    print(brace_member_data)
    print("CHORD MEMBER DATA:")
    print(chord_member_data)
    print("UPDATED SECTIONS:")
    print(final_colinear)
    
    # Accumulate only the updated sections into the dictionary
    updated_members.update(final_colinear)
    
    a = 1



print(updated_members)

updated_members_df = pd.DataFrame.from_dict(updated_members, orient="index")
updated_members_df.index.name = "member_id"
updated_members_df.to_csv("updated_members.csv")

model_out, changes = apply_updated_sections(model, updated_members, member_zones, cone_log=cone_log)
model_out.write_model(model.path, "OUT")

# Logs (cone log written after apply, so it includes the new member / group IDs)
pd.DataFrame(cone_log).to_csv("cone_insertions.csv", index=False)

ring_df = pd.DataFrame(ring_log)
ring_df.to_csv("ring_log.csv", index=False)
if not ring_df.empty:
    ring_df[ring_df["status"] == "rings"].to_csv("irs_locations.csv", index=False)

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time} seconds")