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


start_time = time.time()


# Use a recursive pattern to search in all subdirectories
patterns = ["**/*sac.inp", "**/sacinp.*"]
SACSInputFile = None

for pattern in patterns:
    for path in glob.glob(pattern, recursive=True):
        SACSInputFile = path
        print(f'Found file: {SACSInputFile}')
        break  # Stop after finding the first file for this pattern
    if SACSInputFile:  # If a file was found, stop searching altogether
        break
if SACSInputFile is None:
    print("No matching file found.")

model = SACSModel(SACSInputFile)
z_max = 21.5

df = pd.read_csv("governing_UCs_df.csv")

# Create a working copy
working_df = df.copy()
# Create UC lookup from the dataframe
uc_lookup = ( working_df.dropna(subset=["member_id"]).set_index("member_id")["UC_max"].to_dict())
uc_FLS_lookup = ( working_df.dropna(subset=["member_id"]).set_index("member_id")["Max_Fatigue_UC"].to_dict())

# Identify members with UC_max < 0.4
members_to_optimize = (working_df.loc[working_df["UC_max"] < 1, "member_id"].dropna().unique())
print(f"Members to optimize: {len(members_to_optimize)}")


processed_member_ids = set()
# Thresholds (placeholders, set to your criteria)
UC_STRENGTH_THRESHOLD  = 0.50   # strength / lift
UC_FLS_THRESHOLD = 0.30   # fatigue

# Initialize dictionary to collect all updated members across loops
updated_members = {}

# Loop through each member
for member_id in members_to_optimize:
    if member_id in processed_member_ids:
        continue
    member = model.members[member_id]
    is_leg = member.is_leg
    colinear_members = s.get_colinear_members(model, member, z_max=21.5, include_cones=True)[0]
    colinear_member_ids = [colinear_member.Id for colinear_member in colinear_members]
    print(colinear_member_ids)
    processed_member_ids.update(colinear_member_ids)


    colinear_joints, colinear_joint_ids = s.get_colinear_joints_in_order(colinear_members)
    print(colinear_joint_ids)
    
    # Dictionary containing the relevant properties for each colinear member
    colinear_members_data = {
        colinear_member.Id: {
            "UC_max": uc_lookup.get(colinear_member.Id),
            "UC_FLS_max": uc_FLS_lookup.get(colinear_member.Id),
            "OD": colinear_member.OD,
            "THK": colinear_member.THK,
            "OD_THK": colinear_member.OD / colinear_member.THK
        }
        for colinear_member in colinear_members
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
                "chord_UCmax": uc_lookup.get(chord.Id),
                "chord_FLS_UCmax": uc_FLS_lookup.get(chord.Id)
            }
            for chord in chord_objects
            if chord.Id not in colinear_member_ids
        ]

    print("COLINEAR DATA:")
    print(colinear_members_data)
    print("BRACE MEMBER DATA:")
    print(brace_member_data)
    print("CHORD MEMBER DATA:")
    print(chord_member_data)

    final_colinear = s.get_final_colinear_sections(colinear_members_data, brace_member_data, chord_member_data, is_leg)
    print("UPDATED SECTIONS:")
    print(final_colinear)
    
    # Accumulate only the updated sections into the dictionary
    updated_members.update(final_colinear)
    
    a = 1

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time} seconds")