from SACS_API import *
import numpy as np
import pandas as pd
import math
# import SACS
from scipy.spatial import distance
import glob
import re
import string
import os
import itertools
import copy
from scipy.integrate import quad_vec
from scipy.integrate import simpson
import time
from collections import defaultdict

def read_group_params(SACSInputFile):
    # List to hold the matching lines
    groups = {}
    # Flag to indicate whether to start storing lines
    start_storing = False
    # Set of keywords to stop reading further
    stop_keywords = {"PGRUP", "PLATE", "JOINT", "MEMBER"}
    # Regex pattern to match the new MEMBER format
    pattern = re.compile(r'GRUP [A-Z0-9. ]{3}')
    # Open the file for reading
    with open(SACSInputFile, 'r') as file:
        for line in file:
            # Use a stripped version of the line for checks
            stripped_line = line.strip()
            # Stop reading if line starts with any stop keyword
            if any(stripped_line.startswith(keyword) for keyword in stop_keywords):
                break
            # Check if the stripped line contains the header "GROUP" and set the flag
            if stripped_line.startswith('GRUP')  and not start_storing:
                start_storing = True
                continue
            # Check if the stripped line matches the regex pattern after the header is found
            if start_storing and pattern.match(stripped_line):
                group_id = stripped_line[5:8].strip()
                fl = stripped_line[69:70].strip()
                if fl == "F":
                    fl = True
                else:
                    fl = False
                # print(group_id, fl)

                groups[group_id] = {"flooded": fl}  # Store the original line

    return groups


# >>>>>   READS EFFECT LENGTH PARAMETERS FOR EACH MEMBER FROM SACS FILE    ******************<<<<<<<<<<<<<<<<<<
# >>>>>   NOTE: SACS API DOES NOT HAVE EFFECTIVE LENGTH INSTANCE METHODS (MOST LIKELY WILL BE IMPLEMENTED)
def read_effective_length_params(SACSInputFile):
    # List to hold the matching lines
    memb = {}
    # Flag to indicate whether to start storing lines
    start_storing = False
    # Set of keywords to stop reading further
    stop_keywords = {"PGRUP", "PLATE", "JOINT"}
    # Regex pattern to match the new MEMBER format
    pattern = re.compile(r'MEMBER[ 12][A-Z0-9]{6}')
    # Open the file for reading
    with open(SACSInputFile, 'r') as file:
        for line in file:
            # Use a stripped version of the line for checks
            stripped_line = line.strip()
            # Stop reading if line starts with any stop keyword
            if any(stripped_line.startswith(keyword) for keyword in stop_keywords):
                break
            # Check if the stripped line contains the header "MEMBER" and set the flag
            if 'MEMBER' in stripped_line and not start_storing:
                start_storing = True
                continue
            # Check if the stripped line matches the regex pattern after the header is found
            if start_storing and pattern.match(stripped_line) and ("OFFSETS" not in stripped_line):
                joint1 = stripped_line[7:11]
                joint2 = stripped_line[11:15]
                member_id = joint1 + "-" + joint2
                KorL = stripped_line[46:47].strip()
                K_L_Y = stripped_line[51:55].strip()
                K_L_Z = stripped_line[55:59].strip()

                memb[member_id] = {"KorL": KorL, "K_L_Y": K_L_Y, "K_L_Z": K_L_Z}  # Store the original line
                # print(member, members[member]["KorL"], members[member]["K_L_Y"], members[member]["K_L_Z"])
    return memb

def calc_eff_lengths(length, eff_param):
    if eff_param["KorL"] == "":
        return [length, length]
    elif eff_param["KorL"] == "L":
        return [eff_param["K_L_Y"], eff_param["K_L_Z"]]
    elif eff_param["KorL"] == "K":
        return [eff_param["K_L_Y"]*length, eff_param["K_L_Z"]*length]
    
def process_load(joint_data, forces_data, target_columns):

    # ---------------------------------------------------------
    # 1. Member definitions
    # ---------------------------------------------------------
    member_keys = [
        "chord1", "chord2",
        "brace1", "brace2", "brace3",
        "x-brace1", "x-brace2", "x-brace3"
    ]

    # Number of output columns per member
    member_column_counts = {
        "chord1": 4,
        "chord2": 4,
        "brace1": 3,
        "brace2": 3,
        "brace3": 3,
        "x-brace1": 1,
        "x-brace2": 1,
        "x-brace3": 1
    }

    # ---------------------------------------------------------
    # 2. Build joint_data_output_columns dynamically
    # ---------------------------------------------------------
    joint_data_output_columns = {}
    idx = 0
    for member_key in member_keys:
        count = member_column_counts[member_key]
        joint_data_output_columns[member_key] = target_columns[idx:idx+count]
        idx += count

    # ---------------------------------------------------------
    # 3. Axis flag columns (IPB only)
    # ---------------------------------------------------------
    members_ipbs_columns = {}
    for member_key in member_keys:
        if "chord" in member_key:
            members_ipbs_columns[member_key] = "chd_IPBs"
        elif member_key.startswith("brace"):
            members_ipbs_columns[member_key] = "brc_IPB" + member_key[-1]
        else:
            members_ipbs_columns[member_key] = None

    # ---------------------------------------------------------
    # 4. Force column names in forces_data
    # ---------------------------------------------------------
    force_columns = [
        'FXA','MYA','MZA','FYA','FZA',
        'FXB','MYB','MZB','FYB','FZB'
    ]

    # ---------------------------------------------------------
    # 5. Process each member
    # ---------------------------------------------------------
    for member_key in member_keys:

        # Merge forces for this member
        temp_df = joint_data[[member_key]].merge(
            forces_data[['member'] + force_columns],
            left_on=member_key,
            right_on='member',
            how='left'
        ).drop(columns=['member'])

        # Split A and B forces
        temp_df_a = temp_df[['FXA','MYA','MZA','FYA','FZA']].values
        temp_df_b = temp_df[['FXB','MYB','MZB','FYB','FZB']].values

        # Determine A/B end
        mask_a = joint_data['joint'] == joint_data[member_key].str[:4]
        mask_b = joint_data['joint'] == joint_data[member_key].str[-4:]

        # Output columns for this member
        out_cols = joint_data_output_columns[member_key]

        # -----------------------------------------------------
        # 6. Write forces (different for 4/3/1-column members)
        # -----------------------------------------------------
        if len(out_cols) == 4:
            # Chord: FX, IPB, OPB, fYZ
            joint_data.loc[mask_a, out_cols[:3]] = temp_df_a[mask_a][:, :3]
            joint_data.loc[mask_b, out_cols[:3]] = temp_df_b[mask_b][:, :3]

        elif len(out_cols) == 3:
            # Brace: FX, IPB, OPB
            joint_data.loc[mask_a, out_cols] = temp_df_a[mask_a][:, :3]
            joint_data.loc[mask_b, out_cols] = temp_df_b[mask_b][:, :3]

        else:
            # x-brace: only FX
            joint_data.loc[mask_a, out_cols] = temp_df_a[mask_a][:, 0]
            joint_data.loc[mask_b, out_cols] = temp_df_b[mask_b][:, 0]

        # -----------------------------------------------------
        # 7. Swap IPB ↔ OPB if axis is Z (only for 3-column braces)
        # -----------------------------------------------------
        axis_col = members_ipbs_columns[member_key]

        if axis_col is not None:
            mask_c = joint_data[axis_col] == "Z"
            if mask_c.any():
                col_ipb = out_cols[1]
                col_opb = out_cols[2]
                joint_data.loc[mask_c, [col_ipb, col_opb]] = \
                    joint_data.loc[mask_c, [col_opb, col_ipb]].values

        # -----------------------------------------------------
        # 8. Chord shear extraction (FY or FZ depending on chd_IPBs)
        # -----------------------------------------------------
        if member_key in ["chord1", "chord2"]:

            axis_flag = joint_data["chd_IPBs"]

            shear_A = np.where(
            axis_flag == "Y",
            temp_df_a[:, 4], # FZA
            temp_df_a[:, 3] # FYA
            )
            
            shear_B = np.where(
            axis_flag == "Y",
            temp_df_b[:, 4], # FZB
            temp_df_b[:, 3] # FYB
            )

            # # A-end shear
            # shear_A = np.where(axis_flag == "Y", temp_df_a[:, 3], temp_df_a[:, 4])

            # # B-end shear
            # shear_B = np.where(axis_flag == "Y", temp_df_b[:, 3], temp_df_b[:, 4])

            shear_col = out_cols[3]  # chd1_fYZ or chd2_fYZ

            joint_data.loc[mask_a, shear_col] = shear_A[mask_a]
            joint_data.loc[mask_b, shear_col] = shear_B[mask_b]

    return joint_data



def process_load_ULS_old(joint_data, forces_data, target_columns):
    member_keys = ["chord1", "chord2", "brace1", "brace2", "brace3", 'x-brace1','x-brace2','x-brace3']
    # Use a loop to assign names of chord and brace force columns # Example >> member_forces_dict['brace1'] = ['brc1_FX', 'brc1_IPB', 'brc1_OPB']
    joint_data_output_columns = {member_key: target_columns[i*3:(i+1)*3] for i, member_key in enumerate(member_keys)}
    members_ipbs_columns = {member_key: ('chd_IPBs' if 'chord' in member_key else 'brc_IPB' + member_key[-1])  for member_key in member_keys} 
    # Produces a dictionary that relates each member_key to the target force_columns [FXA, MYA, MZA | FXB, MYB, MZB ]
    # forces_data_input_columns = {member_key: ForcesAorB(joint_data['joint'], joint_data[member_key]) for member_key in member_keys}
    force_columns = ['FXA', 'MYA', 'MZA', 'FXB', 'MYB', 'MZB']
    for member_key in member_keys:
        temp_df = joint_data[[member_key]].merge(
            forces_data[['member'] + force_columns], # The columns we want to retain from forces_data
            left_on=member_key,
            right_on='member',
            how='left'
        ).drop(columns=['member'])
        # Retrieve force labels for the current member_key
        # force_labels = forces_data_input_columns[member_key]
        # print(temp_df)
    # Split temp_df into two parts: one for 'A' forces and one for 'B' forces
        temp_df_a = temp_df[force_columns[:3]].values
        temp_df_b = temp_df[force_columns[3:]].values

        # Determine which rows correspond to 'A' forces and which to 'B' forces
        mask_a = joint_data['joint'] == joint_data[member_key].str[:4]
        mask_b = joint_data['joint'] == joint_data[member_key].str[-4:]
        # Mask for changing Y/Z local bending axis
        mask_c = joint_data[members_ipbs_columns[member_key]] == "Z"

        # Update joint_data with the new columns
        joint_data.loc[mask_a, joint_data_output_columns[member_key]] = temp_df_a[mask_a]
        joint_data.loc[mask_b, joint_data_output_columns[member_key]] = temp_df_b[mask_b]

        # Update the second and third columns based on mask_c
        if mask_c.any():
            # Get the column indices for the second and third columns
            col_2_index = joint_data_output_columns[member_key][1]
            col_3_index = joint_data_output_columns[member_key][2]

        # Swap values between the second and third columns where mask_c is True
        joint_data.loc[mask_c, [col_2_index, col_3_index]] = joint_data.loc[mask_c, [col_3_index, col_2_index]].values


    return joint_data
    


# ************************************>>>>>>          FORCE EXTRACTION               ******************<<<<<<<<<<<<<<<<<<
# NOTE: IN THE FOLLOWING FUNCTION ROW SERIES KEYS and COLUMN HANDLES ARE DYNMICALLY HANDLED. For a more striaght forward version please see superseeded version.
# This function for each joint row processes the forces into FX/IPB/OPB for each brace/chord.
def process_load_old(filtered_joints, filtered_forces, target_columns):
    # df here would be df_filtered passed from the loop
    def get_forces(row):
        # print(f"Index {row.name} " + f"Joint {row['joint']}" + f" Load {row['loads']}")
        # print(f"Index {row.name} " + f"Joint {row['joint']}" + f" Load {row['loads']}")
        # Define a {} that has members and key and [FXA, MYA, MZA | FXB, MYB, MZB ] as values
        member_keys = ['chord1', 'chord2', 'brace1', 'brace2', 'brace3']
        # Example >>  members_columns['brace1']=[FXA, MYA, MZA]
        members_columns_dict = {member_key: ForcesAorB(row['joint'], row[member_key]) for member_key in member_keys}
        # Initialize the dictionary
        member_forces_dict = {}
        # Use a loop to assign names of chord and brace force columns 
        # Example >> member_forces_dict['brace1'] = ['brc1_FX', 'brc1_IPB', 'brc1_OPB']
        member_forces_indexes = {member_key: target_columns[i*3:(i+1)*3] for i, member_key in enumerate(member_keys)}

        # Using isin to fetch all rows for these members and the specific load at once
        members_to_query = [row[member_key] for member_key in member_keys]
        filtered_forces_ = filtered_forces[(filtered_forces['member'].isin(members_to_query))]
        # Dictionary to hold force data series
        forces_data = {}

        # Process each member force data
        for member_key in member_keys:
            member_name = row[member_key]
            # Determine the correct IPB key based on whether the member is a chord or a brace
            ipb_key = 'chd_IPBs' if 'chord' in member_key else 'brc_IPB' + member_key[-1] 

            # if member_name in filtered_forces_['member'].values:
            member_forces = filtered_forces_[(filtered_forces_['member'] == member_name)]
            force_columns = members_columns_dict[member_key]
            # Select the required columns for this member
            member_force_data = member_forces[force_columns]
            if not member_force_data.empty:
                # Check and convert forces as needed
                processed_forces = check_IPB(row[ipb_key], member_force_data).iloc[0]
                # print(processed_forces)
                # Correctly update the series index to match the required output format
                processed_forces.index = member_forces_dict[member_key]
                forces_data[member_key] = processed_forces

        # Update the row with all the processed forces
        if forces_data:
            all_forces = pd.concat(list(forces_data.values()))
            # print(all_forces)
            row.update(all_forces)

        return row
    return filtered_joints.apply(get_forces, axis=1)


def import_joint_geometry(model, z_max, file_name):
    df = pd.DataFrame({})
    for joint_id, joint in model.joints.items():
        if is_joint(joint, z_max):
            #rint(joint.Id)
            jointprop = get_joint_prop(joint, True)
            jointprop = expand_keys(jointprop)
            df = pd.concat([df, pd.DataFrame(jointprop)], ignore_index=True)
    df = add_groups(df, model)
    # df = df.astype(str)
    df = df.replace({"nan": "", "None": "", "0.0": "", "0": "", "NA": ""})
    # Replace actual NaN values with empty strings
    df = df.fillna('')    
    # Replace string representations of NaN ('NA', 'na', 'nan', 'NaN') with empty strings
    df = df.replace(['nan', 'NaN', 'NA', 'na'], '', regex=False)
    df = add_X_braces_FLS(df)
    

    print(df)
    df.to_csv(f'{file_name}' + '.csv', index=False)
    return df

def get_group_id_vectorized(column, model):
    # Assuming column is a numpy array or list of objects
    print(column.iloc[0], type(column.iloc[0]))
    groups = np.array(['NA' if value == 'NA' else model.members[value].group.Id for value in column])
    print(groups)

    return groups

def add_groups(data, model):
    target_columns = ['chord1', 'chord2', 'brace1', 'brace2', 'brace3']

    for col in target_columns:
        print(col)
        groups = get_group_id_vectorized(data[col], model)
        data[f'{col}_group'] = groups
    
    return data

def get_member_prop(model, z_max, SACSInputFile, length_option = False ): 
    if length_option == False:
        eff_params = read_effective_length_params(SACSInputFile)
    else:
        eff_params = {}
        braces_to_avoid = avoid_groups("avoid_brace_groups.txt")

    group_params = read_group_params(model.path)
    avoid_list = avoid_groups("avoid_member_groups.txt")
    IDs, lengths, ODs, THKs, SYs, Lys, Lzs, Zs, flooded, groups, Es = [[] for _ in range(11)]
    cone_IDs, cone_lengths, L_ODs, S_ODs, cone_THKs, cone_SYs, cone_Lys, cone_Lzs, cone_L_Zs, cone_S_Zs, cone_flooded, cone_groups, Es_cone = [[] for _ in range(13)]
    member1_IDs, member1_ODs, member1_THKs, member1_SYs = [[] for _ in range(4)]
    member2_IDs, member2_ODs, member2_THKs, member2_SYs = [[] for _ in range(4)]
    for member_id, member in model.members.items():
        if (not member.is_tube) and (not member.is_cone):
            continue
        coords1, coords2 = [member.coord1, member.coord2]
        eff_lengths = []
        group = member.group.Id

   
        if (coords1[2] < z_max) and (coords2[2] < z_max) and (group not in avoid_list):
            # Identify if the member is a cone or a tube
            is_tube = member.is_tube
            is_cone = member.is_cone
            if is_cone:
                cone_section = member.group.section

            if is_tube:
                OD = member.OD * 10
                THK = member.THK * 10
                SY = float(member.group.FY) * 10
                E = member.E * 10000
                length = member.length

                if (length_option == True) and (member.Id not in eff_params):    
                    eff_params = calc_effective_length_params(model, member, eff_params, z_max, braces_to_avoid)

                eff_lengths = calc_eff_lengths(length, eff_params[member.Id])

                if is_numerical(OD): 
                    IDs.append(member.Id)
                    ODs.append(OD)
                    THKs.append(THK)
                    Es.append(E)
                    lengths.append(member.length)
                    SYs.append(SY)     
                    Lys.append(eff_lengths[0])  
                    Lzs.append(eff_lengths[1])    
                    Zs.append((coords1[2] + coords2[2])/2)
                    flooded.append(group_params[group]["flooded"])
                    groups.append(group)
            if is_cone:
                joint1, joint2 = [member.joint1, member.joint2]
                if (len(joint1.AttachedMembers) == 2) and (len(joint2.AttachedMembers) == 2):
                    
                    # Large Side
                    memb1, memb2 = joint1.AttachedMembers
                    if memb1.Id == member.Id:
                        member1 = memb2
                    else:
                        member1 = memb1
                    # Small Side
                    memb1, memb2 = joint2.AttachedMembers
                    if memb1.Id == member.Id:
                        member2 = memb2
                    else:
                        member2 = memb1
                    member1_OD = member1.OD * 10
                    member1_THK = member1.THK * 10
                    member1_SY = float(member1.group.FY) * 10
                    member2_OD = member2.OD * 10
                    member2_THK = member.THK * 10
                    member2_SY = float(member2.group.FY) * 10
                    cone_ODL = member.OD_L * 10
                    cone_ODS = member.OD_S * 10
                    cone_THK = member.THK * 10
                    E_cone = member.E * 10000
                    cone_SY = float(member.group.FY) * 10
                    cone_length = member.length


                    if is_numerical(cone_ODL): 
                        cone_IDs.append(member.Id)
                        L_ODs.append(cone_ODL)
                        S_ODs.append(cone_ODS)
                        Es_cone.append(E_cone)
                        cone_THKs.append(cone_THK)
                        cone_lengths.append(cone_length)
                        cone_SYs.append(cone_SY)
                        cone_L_Zs.append(coords1[2])      
                        cone_S_Zs.append(coords2[2])
                        cone_flooded.append(group_params[group]["flooded"])
                        cone_groups.append(group)

                        member1_IDs.append(member1.Id)
                        member1_ODs.append(member1_OD)
                        member1_THKs.append(member1_THK)
                        member1_SYs.append(member1_SY)
                        member2_IDs.append(member2.Id)
                        member2_ODs.append(member2_OD)
                        member2_THKs.append(member2_THK)
                        member2_SYs.append(member2_SY)

                        if (length_option == True) and (member.Id not in eff_params):    
                            eff_params = calc_effective_length_params(model, member, eff_params, z_max, braces_to_avoid)
                        eff_lengths = calc_eff_lengths(cone_length, eff_params[member.Id])
                        cone_Lys.append(eff_lengths[0])  
                        cone_Lzs.append(eff_lengths[1])

    member_dict = {"member": IDs, "OD": ODs, "THK": THKs, "groups": groups, "SY": SYs, "length": lengths, "Ly": Lys, "Lz": Lzs, "Z": Zs, "flooded": flooded, "E": Es}
    cone_dict = {"member": cone_IDs, "OD_L": L_ODs, "OD_S": S_ODs, "THK": cone_THKs, "groups": cone_groups, "E": Es_cone, "SY": cone_SYs, "length": cone_lengths, "Ly": cone_Lys, "Lz": cone_Lzs, "Z_L": cone_L_Zs, "Z_S": cone_S_Zs, "flooded": cone_flooded,
                 "member1": member1_IDs, "OD_1": member1_ODs, "THK_1": member1_THKs, "SY_1": member1_SYs,
                 "member2": member2_IDs, "OD_2": member2_ODs, "THK_2": member2_THKs, "SY_2": member2_SYs,}

    a = 1
    return cone_dict, member_dict


# RETURNS JOINT, CHORD, BRACES and GAPS instances
def get_joint_prop(joint, ID_option = False): 
    print(joint.Id)

    avoid_list = avoid_groups("avoid_joint_groups.txt")
    members = get_joint_members(joint, avoid_list)
    # print("MEMBERS:", dict_of_arrays_Ids(members))
    # if joint.Id == "60D1":
    #     a = 1
    chord = members["chords"][0]
    chord_v = outer_vect(joint, chord)
    all_braces = members["braces"]
    checked_list = ""
    brace_list = []
    gap_list = []
    angle_list = []
    plan_list = []
    braces_IPB = []
    for brace1 in all_braces:
        joint_braces = []
        brace_vectors = []
        if (brace1.Id not in checked_list) and (brace1.group.Id not in avoid_list):
            brace1_v = outer_vect(joint, brace1)      
            joint_braces.append(brace1)
            brace_vectors.append(brace1_v)
            checked_list += brace1.Id
            plan = plan_name(chord_v, brace1_v)
            plan_list.append(plan)
            braces_IPB.append(IPBaxis(plan, brace1_v))

            for brace2 in all_braces:
                    if (brace2.Id not in checked_list) and (brace2.group.Id not in avoid_list):
                        brace2_v = outer_vect(joint, brace2)
                        #same_joint = is_same_joint(chord_v, brace1_v, brace2_v) 
                        coplanar = are_coplanar(chord_v, brace1_v, brace2_v) 
                        same_side = are_on_same_side(chord_v, brace1_v, brace2_v) 
                        if (coplanar == True) and (same_side == True):
                            joint_braces.append(brace2)
                            brace_vectors.append(brace2_v)
                            checked_list += brace2.Id
           
            joint_braces = order_braces(chord_v, brace_vectors, joint_braces)
            gap_list.append(calculate_gaps(joint, chord, joint_braces))
            angle_list.append(list(map(lambda x: np.rad2deg(x), get_angles2(joint, chord_v, joint_braces))))
            brace_list.append(joint_braces)
            n_joints = len(brace_list)
  
    brc_ODs = [[get_geo(brace,"OD") for brace in braces] for braces in brace_list]
    brc_THKs = [[get_geo(brace,"THK") for brace in braces] for braces in brace_list]
    brc_SYs = [[get_geo(brace,"FY") for brace in braces] for braces in brace_list]
    braces_IPB = [[IPBaxis(plan_list[i], outer_vect(joint, brace)) for brace in braces] for i,braces in enumerate(brace_list)]
     

    chd_ODs = [get_geo(chord,"OD") for chord in members["chords"]]
    chd_THKs = [get_geo(chord,"THK") for chord in members["chords"]]
    chd_IPBs = [IPBaxis(plan, outer_vect(joint, chord)) for plan in plan_list]
    chd_SYs = [get_geo(chord,"FY") for chord in members["chords"]]
    
    #print(chd_SYs)

    if ID_option == False: # return SACS Objects
        data = { "joint": [joint] * n_joints , "type": joint_type2(brace_list), "chords": [members["chords"]] * n_joints, "braces": brace_list, "gaps": gap_list, "angles": angle_list,
                "chd_ODs": [chd_ODs] * n_joints, "chd_THKs": [chd_THKs] * n_joints, "brc_ODs": brc_ODs, "brc_THKs": brc_THKs, "plan": plan_list, "chd_IPBs": chd_IPBs, "brc_IPBs": braces_IPB#}
                ,"brc_SYs": brc_SYs, "chd_SYs": [chd_SYs]* n_joints }
    else: # turn all SACS objects to Id strings
        data = { "joint": obj_to_Id(n_joints, [joint]), "type": joint_type2(brace_list),"chords": obj_to_Id3(n_joints, members["chords"]), "braces": obj_to_Id2(brace_list), "gaps": gap_list, "angles": angle_list,
                "chd_ODs": [chd_ODs] * n_joints, "chd_THKs": [chd_THKs] * n_joints, "brc_ODs": brc_ODs, "brc_THKs": brc_THKs, "plan": plan_list, "chd_IPBs": chd_IPBs, "brc_IPBs": braces_IPB#}
                ,"brc_SYs": brc_SYs, "chd_SYs": [chd_SYs]* n_joints }
        return data 


def get_geo(member, method, joint_side=None):
    if member.is_tube and method != "FY":
        prop = member.group.section if member.group.section is not None else member.group
    elif member.is_cone and method != "FY":
        prop = member.group.section
    elif method == "FY":
        prop = member.group
    else:
        prop = None

    # Special case for cones: determine OD side based on joint_side
    if method == "OD" and (joint_side is not None) and member.is_cone:
        if joint_side == member.joint1.Id:
            method = "OD_L"
        else:
            method = "OD_S"

    value = float(getattr(prop, method, 0)) * 10
    return value

def get_geo_old(member, method):

    group = member.group
    is_cone = False
    is_tube = group.Segments[0].IsSimpleTube
    section_ = group.Segments[0].Section
    if section_:
        is_cone = group.Segments[0].Section.Shape == "CON"
        
    if (method == "OD" or method == "T") and is_tube:
        properties = group.Segments[0].Tube
    elif (method == "OD" or method == "T") and is_cone:
        properties = group.Segments[0].Section.XSect.Values
        if method == "OD":
            method = "OD1" 
    elif method == "FY":
        properties = group.Segments[0].Material
    if properties != None:
        if is_tube or method == "FY":
            value = getattr(properties, method, 0)
        elif is_cone:
            value = properties.get(method, 0)
        else:
            value = 0

        result = round(value * 10, 1)
        return result
    else:
        return None

# TRANSFORM TO DATAFRAME FRIENDLY FORMAT 
def expand_keys(jointprop):    
    new_dict = jointprop
    group_keys = [["chords", "chd_ODs", "chd_THKs", "chd_SYs"  ],["braces","brc_ODs", "brc_THKs", "brc_IPBs", "brc_SYs", "angles", "gaps"]]
    #group_keys = [["chords", "chd_ODs", "chd_THKs"  ],["braces","brc_ODs", "brc_THKs", "angles", "gaps"]]
    r = 1
    for target_keys in group_keys:
        r += 1
        for target_key in target_keys:
            original_arr = jointprop[target_key]
            new_arr = []
            for i in range(r):
                new_arr = []
                new_key = f'{target_key[:-1]}' + f'{i+1}'
                for j in range(len(original_arr)):
                    if len(original_arr[j]) >= i+1:
                        new_arr.append(original_arr[j][i])
                    else:
                        new_arr.append("NA")
                new_dict[new_key] = new_arr
            new_dict.pop(target_key)
    return new_dict


def is_joint(joint, z_max, get_from_list = True, avoid_list = []):
    if get_from_list:
        avoid_list = avoid_groups("avoid_joint_groups.txt")
    z = joint.Z
    n_joints = 0
    # if joint.Id == "K118":
    #     a = 1
    if z > z_max:
        return False
    else:
        for member in joint.AttachedMembers:
            if member.group.Id not in avoid_list:
                # print(member.Id, member.group.Id)
                n_joints += 1
    #print(n_joints)
    if z<= z_max and n_joints >= 3:
        return True
    else:
        return False
# List of SACS Objects
def obj_to_Id(i, arr):
    return [value.Id for value in arr] * i

def obj_to_Id3(i, arr):
    return [[value.Id for value in arr]] * i

#List with Sublists of SACS Objects
def obj_to_Id2(arr):
    return [[item.Id for item in sublist] for sublist in arr]

def calculate_gaps(joint, chord, braces):
    chord_v = outer_vect(joint, chord)
    braces_v = list(map(lambda brace: outer_vect(joint, brace), braces))
        
    gaps = []
    if len(braces) > 1:
        angles = get_angles(chord_v, braces_v)
        brace_ODs = list(map(lambda brace: get_geo(brace,"OD"), braces))        
        for i in range(len(braces)-1):
            #test = common_joint_coords(joint, braces[i])
            dst = distance.euclidean(common_joint_coords(joint,braces[i]), common_joint_coords(joint,braces[i+1]))
            brace_footprint1 = brace_ODs[i]/(2*np.sin(angles[i]))/1000
            brace_footprint2 = brace_ODs[i+1]/(2*np.sin(angles[i+1]))/1000
            gaps.append((dst - brace_footprint1 - brace_footprint2) * 1000)  
        if len(braces) == 3: # KT braceplan_ calcualte gap3
            dst = distance.euclidean(common_joint_coords(joint,braces[0]), common_joint_coords(joint,braces[2]))
            brace_footprint1 = brace_ODs[0]/(2*np.sin(angles[0]))/1000
            brace_footprint2 = brace_ODs[2]/(2*np.sin(angles[2]))/1000
            gaps.append((dst - brace_footprint1 - brace_footprint2) * 1000)              
    return gaps

def correct_gap(joint, chord, brace1, brace2, target_gap, gap_tolerance, model, z_max):
    chord_v = chord.outer_vector(joint.Id)
    brace1_v = brace1.outer_vector(joint.Id)
    brace2_v = brace2.outer_vector(joint.Id)
    angle1 = vect_angle(chord_v, brace1_v)
    angle2 = vect_angle(chord_v, brace2_v)
    dst = distance.euclidean(common_joint_coords(joint,brace1), common_joint_coords(joint,brace2))
    brace_footprint1 = brace1.OD/(2*np.sin(angle1))/100
    brace_footprint2 = brace2.OD/(2*np.sin(angle2))/100
    gap = (dst - brace_footprint1 - brace_footprint2) * 1000
    delta = gap - target_gap    
    move1, move2 = decide_brace_movement(angle1, angle2, delta, ortho_tol_deg=2)
    moves = [move1, move2]
    brace_vs = [brace1_v, brace2_v]
    braces   = [brace1, brace2]
    for move_mm, brace_v, brace in zip(moves, brace_vs, braces):
        # Correct chord direction for this brace
        if abs(move_mm) > gap_tolerance:
            colinear_members = get_colinear_members(model, brace, include_cones=True)[0]
            joints, joints_id = get_colinear_joints(brace, colinear_members)
            print(joints_id)
            direction = chord_direction_for_brace(chord_v, brace_v)
            # mm → m
            move_m = move_mm / 1000.0
            # Apply offset directly to the 3-element list
            if joint.Id == brace.joint1_id:
                brace.offset1[0] += direction[0] * move_m
                brace.offset1[1] += direction[1] * move_m
                brace.offset1[2] += direction[2] * move_m
                print(brace.offset1)
            else:
                brace.offset2[0] += direction[0] * move_m
                brace.offset2[1] += direction[1] * move_m
                brace.offset2[2] += direction[2] * move_m
                print(f"Brace Offset {brace.offset2}")
            brace_end1 = common_joint_coords(joint, brace)
            brace_end2 = get_far_end_coordinate(brace, colinear_members)

            P1 = np.array(brace_end1)
            P2 = np.array(brace_end2)
            v = P2 - P1
            v_norm_sq = np.dot(v, v)
            for j in joints[1:-1]:   # skip first and last
                J = np.array([j.X, j.Y, j.Z])
                print(J)
                t = np.dot(J - P1, v) / v_norm_sq
                J_new = P1 + t * v
                print(J_new)
                j.X = J_new[0]
                j.Y = J_new[1]
                j.Z = J_new[2]

    return model


def get_colinear_joints_in_order(colinear_members):
    if not colinear_members:
        return [[], []]

    if len(colinear_members) == 1:
        member = colinear_members[0]
        return [
            [member.joint1, member.joint2],
            [member.joint1.Id, member.joint2.Id]
        ]

    joints = []
    joint_ids = []

    for i in range(len(colinear_members) - 1):
        current_member = colinear_members[i]
        next_member = colinear_members[i + 1]

        common_joint_id = get_common_joint(current_member, next_member)
        other_joint_id = other_item(current_member.Id, common_joint_id )

        if current_member.joint1.Id == other_joint_id:
            other_joint = current_member.joint1
        else:
            other_joint = current_member.joint2

        if current_member.joint1.Id == common_joint_id:
            common_joint = current_member.joint1
        else:
            common_joint = current_member.joint2

        # Add first joint
        if not joints:
            joints.append(other_joint)
            joint_ids.append(other_joint_id)

        joints.append(common_joint)
        joint_ids.append(common_joint_id)

    # Add last joint
    last_member = colinear_members[-1]
    final_joint_id = other_item(last_member.Id, joint_ids[-1])
                                
    if last_member.joint1.Id == final_joint_id:
        final_joint = last_member.joint1
    else:
        final_joint = last_member.joint2

    joints.append(final_joint)
    joint_ids.append(final_joint_id)

    return [joints, joint_ids]

def get_colinear_joints(brace, colinear_members):
    # Determine iteration direction
    if colinear_members[0].Id == brace.Id:
        ordered = colinear_members[:]          # forward
    elif colinear_members[-1].Id == brace.Id:
        ordered = colinear_members[::-1]       # backward
    else:
        raise ValueError("Brace is not at either extreme of colinear list.")

    joints = []
    joint_ids = []

    # First member: add its non-common joint
    first = ordered[0]
    second = ordered[1]

    if first.joint1_id in second.Id:
        j = first.joint2
    else:
        j = first.joint1

    joints.append(j)
    joint_ids.append(j.Id)

    # Middle: add common joints
    for m_prev, m_next in zip(ordered[:-1], ordered[1:]):
        if m_prev.joint1_id in m_next.Id:
            j = m_prev.joint1
        else:
            j = m_prev.joint2

        joints.append(j)
        joint_ids.append(j.Id)

    # Last member: add its non-common joint
    last = ordered[-1]
    second_last = ordered[-2]

    if last.joint1_id in second_last.Id:
        j = last.joint2
    else:
        j = last.joint1

    joints.append(j)
    joint_ids.append(j.Id)

    return joints, joint_ids



def get_far_end_coordinate(brace, colinear_members):
    # If brace is first in the list → far end is last member
    if colinear_members[0].Id == brace.Id:
        far_member = colinear_members[-1]
        prev_member = colinear_members[-2]

    # If brace is last in the list → far end is first member
    elif colinear_members[-1].Id == brace.Id:
        far_member = colinear_members[0]
        prev_member = colinear_members[1]

    else:
        raise ValueError("Brace is not at either extreme of the colinear list.")

    # Identify the far-end joint (the one NOT shared with prev_member)
    if far_member.joint1_id in prev_member.Id:
        return np.array(far_member.coord2)
    else:
        return np.array(far_member.coord1)

    

def chord_direction_for_brace(chord_v, brace_v):
    """
    Returns a unit chord vector oriented so that it forms a closed angle with the brace.
    """
    dot = np.dot(chord_v, brace_v)
    unit_chord = chord_v / np.linalg.norm(chord_v)
    if dot >= 0:
        return unit_chord      # chord points towards brace → OK
    else:
        return -unit_chord     # chord points away → flip

def decide_brace_movement(angle1, angle2, delta_mm, ortho_tol_deg=5):
    """
    Returns:
        move1_mm : movement for brace1 (mm)
        move2_mm : movement for brace2 (mm)
    """
    # Classify brace type
    def classify(angle_rad):
        ang = np.degrees(angle_rad)
        return "ORTHO" if abs(ang - 90) <= ortho_tol_deg else "DIAG"
    type1 = classify(angle1)
    type2 = classify(angle2)
    # Movement logic
    if type1 == "DIAG" and type2 == "DIAG":
        # K-brace → split correction
        move1_mm = -delta_mm / 2.0
        move2_mm = -delta_mm / 2.0
    elif type1 == "DIAG" and type2 == "ORTHO":
        # KT → move diagonal only (brace1)
        move1_mm = -delta_mm
        move2_mm = 0.0
    elif type1 == "ORTHO" and type2 == "DIAG":
        # KT → move diagonal only (brace2)
        move1_mm = 0.0
        move2_mm = -delta_mm
    else:
        # Both orthogonal → no movement
        move1_mm = 0.0
        move2_mm = 0.0
    return move1_mm, move2_mm

def order_braces(chord_v, braces_v, braces):
    angles= list(map(lambda brace_v: vect_angle(chord_v, brace_v), braces_v))
    #print(angles)
    return [x for _, x in sorted(zip(angles, braces))]

def get_angles(chord_v, braces_v):
    return list(map(lambda brace_v: vect_angle(chord_v, brace_v), braces_v))

def get_angles2(joint, chord_v, braces):
    braces_v = list(map(lambda brace: outer_vect(joint, brace), braces))
    return list(map(lambda brace_v: vect_angle(chord_v, brace_v), braces_v))

def are_coplanar(chord, brace1, brace2):
    # Calculate the scalar triple product (a . (b x c))
    scalarTripleProduct = np.dot(chord, np.cross(brace1, brace2))
    # print(scalarTripleProduct, np.isclose(scalarTripleProduct, 0))
    # If the scalar triple product is close to zero, the vectors are coplanar
    return np.isclose(scalarTripleProduct, 0, atol=1e-4, rtol=1e-3)

def are_on_same_side(chord, brace1, brace2):
    # Calculate cross product of chord with each brace
    crossProduct1 = np.cross(chord, brace1)
    crossProduct2 = np.cross(chord, brace2)
    # Check if the dot product of the cross products is positive
    return np.dot(crossProduct1, crossProduct2) > 0

def are_vectors_colinear(vec1, vec2, angle_tol_deg=5):
    """Check if two vectors are colinear within a small angular tolerance."""
    unit1 = vec1 / np.linalg.norm(vec1)
    unit2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(abs(dot_product)))
    return angle_deg < angle_tol_deg

def get_member_coord_at_joint(member, joint_id):
    """Return the coordinate of the member at the side connected to joint.id."""
    if member.joint1.Id == joint_id:
        return np.array(member.coord1)
    elif member.joint2.Id == joint_id:
        return np.array(member.coord2)
    else:
        raise ValueError(f"Member not connected to joint {joint_id}")

def get_joint_members(joint, avoid_list=[], avoid_prefix = [],angle_tol_deg=5, dist_tol=0.01):
    chords = []
    braces = []   
    att_members = [ m for m in joint.AttachedMembers if (m.group.Id not in avoid_list) and (not any(m.group.Id.startswith(pref) for pref in avoid_prefix))
]

    # Track which members have already been classified as chords
    classified_as_chord = set()

    for i, m1 in enumerate(att_members):
        coord1 = get_member_coord_at_joint(m1, joint.Id)
        for j, m2 in enumerate(att_members):
            if j <= i:
                continue
            coord2 = get_member_coord_at_joint(m2, joint.Id)

            if are_vectors_colinear(m1.ref_vec, m2.ref_vec):
                if distance.euclidean(coord1, coord2) < dist_tol:
                    classified_as_chord.update({m1, m2})

    for m in att_members:
        if not m.is_tube:
            continue  # skip non‑tubes completely
        if m in classified_as_chord:
            chords.append(m)
        else:
            braces.append(m)

    return {"chords": chords, "braces": braces}



# From Joint object get {chord / member} objects
def get_joint_members_old(joint, avoid_list):
    chords = []
    braces = []
    att_members = joint.AttachedMembers
    for member in att_members:
        if member.group.Id not in avoid_list:
            coord = common_joint_coords(joint, member)
            dst = distance.euclidean(joint.coord, coord)
            if dst < 0.1:
                chords.append(member)
            else:
                braces.append(member)
    return { "chords": chords, "braces": braces}

# Applies method ID to items of array in dictionary 
def dict_of_arrays_Ids(dictionary):
    return {k: list(map(lambda i: i.Id, v)) for k, v in dictionary.items()}


# return member vector (w/ outwards direction)
def outer_vect(joint, member):
    if joint.Id == member.joint1.Id:
        return memb_vect(member.coord1, member.coord2)
    else:
        return memb_vect(member.coord2, member.coord1)  
    
    # get common joint object between joint/member
def common_joint_coords(common_joint, member):
    if common_joint.Id == member.joint1.Id:
        return member.coord1
    else:
        return member.coord2

# Function that returns actual member coordinates (including offsets)
def memb_coords(memb):
    return [memb.coord1, memb.coord2]

# Function that returns actual member coordinates (including offsets)
def average_elevation(memb):
    return (memb.coord1[2]+memb.coord2[2])/2

# Function that returns member vector
def memb_vect(memb_coord1, memb_coord2):
    return list(map(lambda i, j: j - i, memb_coord1, memb_coord2))


# Function that returns angle between two SACS member objects
def member_angle(m1, m2, radians=True):
    v1 = memb_vect(m1.coord1, m1.coord2)
    v2 = memb_vect(m2.coord1, m2.coord2)
    angle = vect_angle(v1, v2)
    
    if not radians:
        angle = math.degrees(angle)
    
    return angle

# Returns angle with members defined vectorily outwardly
def member_vectored_angle(joint, m1, m2):
    v1 = outer_vect(joint, m1)
    v2 = outer_vect(joint, m2)
    return vect_angle(v1, v2)

def member_angle_min(m1, m2):
    m1_coords1, m1_coords2 = memb_coords(m1)
    m2_coords1, m2_coords2 = memb_coords(m2)
    v1 = memb_vect(m1_coords1, m1_coords2)
    v2 = memb_vect(m2_coords1, m2_coords2)
    angle = vect_angle(v1, v2)
    if angle > math.pi / 2:
        return math.pi - angle
    else:
        return angle
     

# ************** AUX MATH FUNCTIONS ***********************
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def vect_angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vect_angle_min(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if angle > math.pi / 2:
        return math.pi - angle
    else:
        return angle

def plan_name(v1, v2):
    # Calculate the cross product to find the normal vector
    nx = (v1[1]*v2[2] - v1[2]*v2[1])
    ny = (v1[2]*v2[0] - v1[0]*v2[2])
    nz = (v1[0]*v2[1] - v1[1]*v2[0])
    # Analyze the dominant component of the normal vector to name the plane
    # The logic assumes a perfect alignment for simplicity
    dominant_component = max(abs(nx), abs(ny), abs(nz))
    if np.isclose(abs(nx)+abs(ny), 0, 1e-03, 1e-03):
        plan = "XY"
    elif np.isclose(abs(nx)+abs(nz), 0, 1e-03, 1e-03):
        plan = "XZ"
    elif np.isclose(abs(ny)+abs(nz), 0, 1e-03, 1e-03):
        plan = "YZ"
    elif dominant_component == abs(nx):
        plan = "YZ"
    elif dominant_component == abs(ny):
        plan = "XZ"
    elif dominant_component == abs(nz):
        plan = "XY"
    return plan



def IPBaxis(plan, v1):
    #print(plan, is_vertical(v1))
    if plan == "XZ" and is_vertical(v1):
        return "Z"
    elif plan == "XZ" and (not is_vertical(v1)):
        return "Y"
    elif plan == "YZ":
        return "Y"
    elif plan == "XY":
        return "Z"  
    else:
        return "NA"


def is_vertical(v1):
    if np.isclose(abs(v1[0])+abs(v1[1]), 0, 1e-02, 1e-02):
        return True
    else:
        return False

def is_numerical(value):
    # Check if value is a direct instance of numeric types
    if isinstance(value, (int, float, complex)):
        return True

    # Check if value is a string and if it can be converted to a float
    if isinstance(value, str):
        if value.strip().lower() == "nan":  # Handle 'NaN'
            return True
        if value.strip() == "":
            return False  # Empty string is not numeric
        try:
            float(value)  # Try to convert string to a float
            return True
        except ValueError:
            return False

    # If value is neither a numeric type nor a string
    return False
# ************** TESTING FUNCTIONS ***********************
def joint_type(braces):
    if len(braces) == 3:
        return "KT"
    elif len(braces) == 2: 
        return "K"
    elif len(braces) == 1:
        return "T"
    else:
        return "NA"

def joint_type2(braces_list):
  return [joint_type(sublist) for sublist in braces_list]
    
def avoid_groups(filename):
    avoid_path = filename
    with open(avoid_path, 'r') as file:
        avoid_list = [line.strip() for line in file]
    return avoid_list
      

def divide_string(prefix, original, max_width=81):
    prefix_length = len(prefix)
    usable_length = max_width - prefix_length
    segments = []

    # Start at the beginning of the string, step by the usable_length
    for start in range(0, len(original), usable_length):
        # Slice the original string for the next segment
        end = start + usable_length
        segment = original[start:end]
        
        # Add the prefix and the sliced segment to the list
        segments.append(prefix + segment)

    return segments


def save_list_to_file(list_to_save, file_name):
    # Open the file in write mode
    with open(file_name, 'w') as file:
        # Iterate through each element in the list
        for item in list_to_save:
            # Write each item to a new line in the file
            file.write(str(item) + '\n')



# # ******************>>>>>>>>>>>>>              Read MEMBER END FORCES DIRECTLY FROM SACS DATABASE                         ******************
# def DB_MEF_to_CSV(prefix, members):
#     Forces = []
#     k=0

#     for path in glob.glob('**/**' + f'{prefix}' + '**', recursive=True):
#         k += 1
#         SACSResultsFile = path
#         print("READING " + SACSResultsFile)
#         postviewDB = SACS.OpenResultDB(SACSResultsFile)
#         dbresults = postviewDB.AnalysisResults
#         memberendforces = dbresults.GetMemberEndForces(Members = members)
#         paths = [path for m in memberendforces]
#         members = [m.Member.Id for m in memberendforces]
#         loads = [m.Load.Id for m in memberendforces]
#         forcesA = [m.ValuesA for m in memberendforces]
#         forcesB = [m.ValuesB for m in memberendforces]
#         df1 = pd.DataFrame({"member": members, "loads": loads, "paths": paths} )
#         df2 = pd.DataFrame(forcesA, columns=["FXA", "FYA", "FZA", "MXA", "MYA", "MZA"])
#         df3 = pd.DataFrame(forcesB, columns=["FXB", "FYB", "FZB", "MXB", "MYB", "MZB"])
#         force_df_list = [df1, df2, df3]
#         forces_df = pd.concat(force_df_list, axis=1)
#         forces_df.to_csv('DBforces_df' + f'{k}' '.csv')
#         Forces.append(forces_df)



# CHCEK IF LAST 6X COLUMNS ARE NUMBERS
def checkIfForceRow(lst):
    # Check if the list has more than six elements
    if len(lst) > 6:
        # Get the last six elements of the list
        last_six = lst[-6:]
        # Check if all these elements are strings convertible to numbers
        for item in last_six:
            try:
                float(item)  # Attempt to convert each to a float
            except ValueError:
                return False  # Return False if conversion fails
        return True  # Return True if all conversions are successful
    else:
        return False  # Return False if there aren't more than six elements

# READ FORCES FROM REPORT SACS OUTPUT FILE
def read_report_forces(prefix):
    k = 0
    member_pattern = r'^[A-Z0-9]{4}-[A-Z0-9]{4}$'
    joint_pattern  = r'^[A-Z0-9]{4}'
    Forces = []

    for path in glob.glob(f'**/**{prefix}.lst', recursive=True):
        k += 1
        SACSReportFile = path
        
        paths = []
        members = []
        loads = []
        forcesA = []
        forcesB = []

        j, joint = ("", "")
        m, member = ("", "")
        l, load = ("", "")

        # NEW: lists of load cases for A and B
        loadsA = []
        loadsB = []
        joint_A = ""   # A-end joint = joint read from A-end force row

        filename = 'forces_df' + f'{k}' '.csv'

        if not os.path.exists(filename):
            with open(SACSReportFile, 'r') as file:
                print(k, f"WRITING {SACSReportFile} to {filename}")
                lines = [line for line in file]

                for line in lines:

                    if len(line) > 8:  m = line[0:9].strip()
                    if len(line) > 15: j = line[11:15].strip()
                    if len(line) > 27: l = line[23:27].strip()

                    # NEW MEMBER → reset A/B load lists
                    if re.fullmatch(member_pattern, m):
                        member = m
                        loadsA = []
                        loadsB = []
                        joint_A = ""   # reset joint_A

                    # JOINT
                    if re.fullmatch(joint_pattern, j):
                        joint = j

                    # LOAD CASE
                    if re.fullmatch(joint_pattern, l):
                        load = l
                        substring = re.sub(r'\s+', ' ', line)
                        line_arr = substring.split()

                        if checkIfForceRow(line_arr):

                            # A-END: first time this load appears
                            if load not in loadsA:
                                loadsA.append(load)
                                joint_A = joint   # EXACTLY AS YOU SAID

                                forcesA.append(line_arr[-6:])
                                members.append(member)
                                loads.append(load)
                                paths.append(SACSReportFile)

                            # B-END: joint changed AND load matches next expected A-load
                            else:
                                # Validate joint change
                                if joint == joint_A:
                                    raise ValueError(
                                        f"B-end joint did not change in {SACSReportFile}: "
                                        f"joint={joint}, joint_A={joint_A}"
                                    )

                                # Validate B-load matches the corresponding A-load
                                expected_load = loadsA[len(loadsB)]
                                if load != expected_load:
                                    raise ValueError(
                                        f"B-end load sequence mismatch in {SACSReportFile}: "
                                        f"expected B-load={expected_load}, got B-load={load}"
                                    )

                                loadsB.append(load)
                                forcesB.append(line_arr[-6:])

                # AFTER FINISHING FILE: validate A/B load lists
                if loadsA != loadsB:
                    raise ValueError(
                        f"A/B load case mismatch in {SACSReportFile}: "
                        f"A-loads={loadsA}, B-loads={loadsB}"
                    )

            # BUILD DATAFRAMES
            df1 = pd.DataFrame({"member": members, "loads": loads, "paths": paths})
            df2 = pd.DataFrame(forcesA, columns=["FXA", "FYA", "FZA", "MXA", "MYA", "MZA"])
            df3 = pd.DataFrame(forcesB, columns=["FXB", "FYB", "FZB", "MXB", "MYB", "MZB"])

            # LENGTH CHECK
            if len(df2) != len(df3):
                raise ValueError(
                    f"A/B mismatch in file {SACSReportFile}: A={len(df2)}, B={len(df3)}"
                )

            forces_df = pd.concat([df1, df2, df3], axis=1)

            # SIMPLE INDEPENDENT SUFFIX
            forces_df["loads"] = forces_df["loads"].astype(str) + f"_{k}"

            forces_df.to_csv(filename, index=False)
            Forces.append(forces_df)

        else:
            print(k, "WRITING " + filename)
            forces_df = pd.read_csv(filename)

    return pd.concat(Forces, ignore_index=True)





# READ FORCES FROM REPORT SACS OUTPUT FILE
def read_report_forces_original(prefix):
    k = 0
    member_pattern = r'^[A-Z0-9]{4}-[A-Z0-9]{4}$'
    joint_pattern = r'^[A-Z0-9]{4}'
    Forces = []

    for path in glob.glob(f'**/**{prefix}.lst', recursive=True):
        k += 1
        SACSReportFile = path
        
        paths = []
        members = []
        loads = []
        forcesA = []
        forcesB = []
        j,joint = ("","")
        m, member = ("","")
        l, load = ("","")
        filename = 'forces_df' + f'{k}' '.csv'
        if not os.path.exists(filename):
            with open(SACSReportFile, 'r') as file:
                print(k, f"WRITING {SACSReportFile} to {filename} ")
                #lines = [line.strip() for line in file]
                lines = [line for line in file]

                substring = ""
                for line in lines:
                    
                    if len(line) > 8: m = line[0:9]
                    if len(line) > 15: j = line[11:15]
                    if len(line) > 27: l = line[23:27]

                    if re.fullmatch(member_pattern, m):
                        member = m
                    if re.fullmatch(joint_pattern, j):
                        joint = j
                    if re.fullmatch(joint_pattern, l):
                        load = l
                        substring = re.sub(r'\s+', ' ', line)   # replaces n" " for a single space
                        line_arr =  substring.split()

                        if checkIfForceRow(line_arr):
                            if start_end(joint, member):
                                forcesA.append(line_arr[-6:])
                                members.append(member)
                                loads.append(load)
                                paths.append(SACSReportFile)
                            else:
                                forcesB.append(line_arr[-6:])

            df1 = pd.DataFrame({"member": members, "loads": loads, "paths": paths} )
            df2 = pd.DataFrame(forcesA, columns=["FXA", "FYA", "FZA", "MXA", "MYA", "MZA"])
            df3 = pd.DataFrame(forcesB, columns=["FXB", "FYB", "FZB", "MXB", "MYB", "MZB"])
            force_df_list = [df1, df2, df3]
            forces_df = pd.concat(force_df_list, axis = 1)
            
            
            # Iterate through each DataFrame in the forces list
            for i, df in enumerate(Forces):
                # Convert the 'loads' column of the current DataFrame to a set
                df_loads = set(df['loads'])
                
                # Find intersection between the loads in forces_df and the current DataFrame
                repeated_loads = df_loads.intersection(set(forces_df["loads"]))
                if repeated_loads:
                    forces_df["loads"] += f"_{k}"

            forces_df.to_csv(filename, index=False)


            Forces.append(forces_df)
        else:
            print(k, "WRITING " + filename)
            forces_df = pd.read_csv(filename)
    return pd.concat(Forces, ignore_index=True)




def start_end(joint, member):
    if member[0:4] == joint:
        return True
    else:
        return False

def adjust_angle(angle):
    try:
        angle = float(angle)
    except ValueError:
        # If angle is not numerical, return it as is
        return angle
    
    if angle > 90:
        return 180 - angle
    else:
        return angle

def path_to_case(str):
    str = str.replace("\\REPORT\\MEF_rpt.lst", "")
    str = str.split("\\")[-1]
    return str

# Constants
FORCES = ['FX', 'MY', 'MZ']

def ForcesAorB(joint_col, member_col):
    # Create empty arrays to store the results with 'A' and 'B' appended
    forces_a = np.array([f + 'A' for f in FORCES])
    forces_b = np.array([f + 'B' for f in FORCES])
    
    # Create boolean masks to check the conditions
    mask_a = (joint_col.values == member_col.str[:4].values)[:, None]
    mask_b = (joint_col.values == member_col.str[-4:].values)[:, None]
    
    # Initialize an array to store the forces
    force_column_keys = np.empty((len(joint_col), len(FORCES)), dtype=object)
    
    # Use np.where() to assign forces vectorially
    force_column_keys = np.where(mask_a, forces_a, force_column_keys)
    force_column_keys = np.where(mask_b, forces_b, force_column_keys)
    
    return force_column_keys

def ForcesAorB_old(joint: str, member: str):
    if pd.notna(member) and len(member) >= 4:
        if joint == member[:4]:
            return [x + "A" for x in FORCES]
        elif joint == member[-4:]:
            return [x + "B" for x in FORCES]
    return []

def force_columns_myversion(joint, member):
    forces = ['FX', 'MY', 'MZ']
    if pd.notna(member):
        if joint == member[:4]:
            forces = [x + "A" for x in forces]
        elif joint == member[-4:]:
            forces = [x + "B" for x in forces]
    else:
        forces = []
    return forces

def check_IPB(axis, forces):
    # Check the condition with the string / Note that forces shoudl be a single row DataFrame
    if (axis == "Z"):  # Example condition, adjust as necessary
        # Swap MYA with MZA if these columns exist
        if 'MYA' in forces.columns and 'MZA' in forces.columns:
            forces['MYA'], forces['MZA'] = forces['MZA'].copy(), forces['MYA'].copy()
        # Swap MYB with MZB if these columns exist
        if 'MYB' in forces.columns and 'MZB' in forces.columns:
            forces['MYB'], forces['MZB'] = forces['MZB'].copy(), forces['MYB'].copy()
    if forces.empty:
        return pd.DataFrame({"a": [0], "b": [0], "c": [0]})
    return forces

def check_IPB_old(axis, forces):

    if (axis == "Z"):
        return [forces[0], forces[2], forces[1]]
    elif (len(forces)==0):
        return ["", "", ""]
    else:
        return forces
    
def left(s, n):
    return s[:n]
    
def right(s, n):
    return s[-n:]    

#************* SUPERSEEDED  ***************************
# Tests if 2x braces are part of same joint
def is_same_joint(chord_vector, brace1_vector, brace2_vector):
    perpendicular_vector1 = np.cross(chord_vector, brace1_vector)
    perpendicular_vector2 = np.cross(chord_vector, brace2_vector)
    planar = np.dot(brace2_vector, perpendicular_vector1) # zero if planar

    direction1 = np.dot(perpendicular_vector1, perpendicular_vector1)
    direction2 = np.dot(perpendicular_vector1, perpendicular_vector2)
    same_direction = direction1 / direction2

    if (abs(planar) < 0.001) and same_direction > 0: 
        return True
    else:
        return False
    
    # get other joint object between joint/member
def other_joint_coords(common_joint, member):
    joints = member.Joints
    m_coords = memb_coords(member)
    if common_joint.Id == joints[0].Id:
        return m_coords[1]
    else:
        return m_coords[0]

def get_common_joint(first_member, second_member):
    joint1_ids = [first_member.joint1_id, first_member.joint2_id]
    joint2_ids = [second_member.joint1_id, second_member.joint2_id]
    common_joint = list(set(joint1_ids) & set(joint2_ids))[0]
    return common_joint


def ordered_joints(first_member, second_member):
    common_joint = get_common_joint(first_member, second_member)
    other_joint = other_item(second_member.Id, common_joint)
    return [common_joint, other_joint]



def split_member_add_cones(row, model, z_max, final_diff):
    central_member_id = row['member']
    # Check if the target central member has been previously split
    condition = final_diff['check_type'].str.contains("Split", case=False, na=False) & final_diff['check_type'].str.contains(str(central_member_id), na=False)
    print(central_member_id)
    if condition.any(): # If so apply the changes to the second member
        central_member_id = final_diff.loc[condition, 'member'].iloc[0]
    central_member = model.members[central_member_id]
    central_member_OD = row['OD_after']
    central_member_THK = row['THK_after']
    
    group_id = row['group']
    Lengths = []
    Members = []
    before_cone_Members = [[], []]
    all_joints = model.joints.values()
    all_joints = [joint.Id for joint in all_joints]
    all_joints = list(set(all_joints))
    all_sections = model.sections.values()
    all_sections = [section.Id for section in all_sections]
    all_sections = list(set(all_sections))
    all_groups = model.members.values() 
    all_groups = [group.Id for group in all_groups]
    all_groups = list(set(all_groups))
    
    # Get the colinear members for the current member
    colinear_members = get_colinear_members(model, central_member, z_max)
    cones = colinear_members[1].copy()
    colinear_members = colinear_members[0]

    member_ids = [member.Id for member in colinear_members]
    print(central_member_id, member_ids)

    central_index = member_ids.index(central_member_id)
    # Get member objects to the left and right of the central member
    # Calculate total lengths using list comprehensions
    Members.append(colinear_members[:central_index])
    Members.append(colinear_members[central_index + 1:])

    Lengths.append(sum(member.length for member in colinear_members[:central_index]))
    Lengths.append(sum(member.length for member in colinear_members[central_index + 1:]))
    cone_added = False
    member_split = False
    
    ID_inner = central_member_OD - 2 * central_member_THK

    for i in [0, 1]:
        reached_end = False
        if Members[i]:       
            # Correct usage for maximum outer diameter calculation
            max_OD = max(get_geo(member, "OD") for member in Members[i] if member is not None)
            ΔOD = central_member_OD - max_OD
            if ΔOD > 50 and Lengths[i] > 2:
                j = 1 - i
                current_member = central_member
                # <<< add a test tp see of target_member has any joints, if so move on to next member and add that member to the before_cone_Members[i] list
                n = 1
                n_joints = 3
                target_member_length = 0
                # and (n < len(Members[i])) 
                while ((target_member_length) <= 1) :
                    if i == 0: # members towards left
                        target_member = Members[i][-n]
                        if len(Members[i]) >= n+1:
                            next_member = Members[i][-n-1]
                        else:
                            next_member = target_member
                    else: # members towards right
                        target_member = Members[i][n-1]
                        if len(Members[i]) >= n+1:
                            next_member = Members[i][n]
                        else:
                            next_member = target_member
                    target_member_length = target_member.length   
                    joint_Ids = ordered_joints(current_member, target_member)  
                    joint1 = model.joints[joint_Ids[0]] # common joint
                    joint2 = model.joints[joint_Ids[1]] # other joint

                    n_joints = len(joint2.AttachedMembers)
                    if (n_joints > 2) or target_member_length <= 1:
                        n += 1
                        if n == len(Members[i]):
                            reached_end = True
                            break
                        current_member = target_member
                        before_cone_Members[i].insert(0, target_member)
    
                istube = target_member.is_tube and next_member.is_tube
                if istube and not reached_end:
                    if target_member.length > 1.5:
                        original_Id = target_member.Id
                        condition = final_diff["member"] == original_Id
                        if target_member.joint2.Id == joint_Ids[0]: # cone is counter-wise
                            target_member.invert()

                        ratio = 1.0 / target_member.length    
                        new_joint, memb1, memb2 = target_member.split_member_at_ratio(ratio)

                        # test = model.members[original_Id]
                        target_member = memb1
                        next_member = memb2
                        member_split = True

                        if target_member.Id == "2261-Y225" or next_member.Id == "2261-Y225":
                            a = 1
                        

                        if condition.any():
                            final_diff.loc[condition, "member"] = memb2.Id
                            final_diff.loc[condition, "check_type"] = f"{original_Id} Split " + final_diff.loc[condition, "check_type"].iloc[0]

                        if (not member_split) and ((f"{joint_Ids[0]}-{joint_Ids[1]}") != target_member.Id):
                            target_member.delete
                            target_member = Member( f"{joint_Ids[0]}-{joint_Ids[1]}", model,joint_Ids[0], joint_Ids[1],target_member.group_id,target_member.offset_type,target_member.offset2,target_member.offset1, target_member.fixity2, target_member.fixity1, target_member.chord_angle,target_member.ref_joint_id, target_member.length_option, target_member.Ly,target_member.Lz,target_member.flooded, target_member.memb2  )
                            model.add_member(target_member)
                        
                        # Transform to Cone
                        new_section_Id = generate_new_section("C50", all_sections) 
                        new_group_Id = model.make_unique_group_id(group_id)
                        all_sections += new_section_Id
                        Members[i] = before_cone_Members[i]
                        original_thk = get_geo(target_member, "THK")
                        cone_thk = original_thk + 5
                        OD_S = get_geo(next_member, "OD")
                        OD_L = ID_inner + 2 * cone_thk
                        new_section_values = {"section_id": new_section_Id, "stype": "CON", "OD": None, "THK": cone_thk/10, "OD_L": OD_L/10, "OD_S": OD_S/10, "THK1": None, "THK2": None}
                        new_section = Section(**new_section_values)
                        model.add_section(new_section)
                        grp_seg = target_member.group.segments[0]

                        mat_ = material(cone_thk, grp_seg.FY)
                        FY_new = yield_strength(cone_thk, mat_)
                        new_segment = GroupSegment(1, new_section_Id,"","", E=grp_seg.E, G=grp_seg.G, FY= FY_new/10, Ky=grp_seg.Ky, Kz=grp_seg.Kz, flooded=grp_seg.flooded, density=grp_seg.density, segment_ratio="1")
                        new_group = Group(new_group_Id, model)
                        new_group.add_segment(new_segment)
                        model.add_group(new_group)

                        target_member.group_id = new_group.Id
                        if memb1.Id == "105M-100B" or memb2.Id == "105M-100B":
                            test = model.members["105M-100B"]  
                            a = 1
                        cone_added = True
                        if not member_split: 
                            original_Id = target_member.Id
                            member_Id = original_Id
                        else:
                            member_Id = memb1.Id
                        
                        condition = final_diff["member"] == original_Id
                        final_diff = final_diff[~condition]
                        new_row = pd.DataFrame([{
                        'member': member_Id,
                        'group': target_member.group.Id,
                        'check_type': "Cone Introduced",
                        'OD_after': OD_L,
                        'OD_before': OD_S,
                        'THK_after': cone_thk,
                        'THK_before': original_thk,
                        'SY_after': yield_strength(cone_thk, material(cone_thk, get_geo(target_member, "FY"))),
                        'SY_before': get_geo(target_member, "FY"),                    
                        }])
                        # Concatenate the new row with the final_diff DataFrame
                        final_diff = pd.concat([final_diff, new_row], ignore_index=True)  

   
    return final_diff

def update_col_members(row, model, z_max, final_diff):
    central_member_id = row['member']
    central_member = model.members[central_member_id]
    central_member_OD = row['OD_after']

    # Get the colinear members for the current member
    colinear_members = get_colinear_members(model, central_member, z_max)
    cones = colinear_members[1].copy()
    colinear_members = colinear_members[0]

    member_ids = [member.Id for member in colinear_members]
    print(central_member_id, member_ids)

    member_IDs = {member.Id: get_geo(member, "OD") - 2 * get_geo(member, "THK") for member in colinear_members}
    
    for member_id in member_IDs:
        # Filter final_diff to find rows matching the current member_id
        matching_rows = final_diff[final_diff['member'] == member_id]
        if not matching_rows.empty:
            # Calculate the maximum ID from final_diff for this member
            df_ID = (matching_rows['OD_after'] - 2 * matching_rows['THK_after']).iloc[0]
            member_IDs[member_id] = max(member_IDs[member_id], df_ID)
            optimized_ID = row['OD_after'] - 2 * row['THK_after']
            member_IDs[central_member_id] = max(optimized_ID, member_IDs[central_member_id])

    target_ID = max(member_IDs.values())
    print(member_IDs)

    # Iterate over each member in colinear_members and guarantee inner flushed members
    for colinear_member in colinear_members:
        colinear_ID = member_IDs[colinear_member.Id]
        # ------------------------------------------------------------
        # FIX: check final_diff first and adjust OD_after if needed
        # ------------------------------------------------------------
        if colinear_member.Id in final_diff['member'].values:
            fd_row = final_diff.loc[final_diff['member'] == colinear_member.Id].iloc[0]
            fd_inner_ID = fd_row['OD_after'] - 2 * fd_row['THK_after']

            # If updated inner ID is below target → fix OD_before comparing anything else
            if fd_inner_ID < target_ID:
                target_THK = fd_row['THK_after']
                new_OD_after = target_ID + 2 * target_THK
                final_diff.loc[final_diff['member'] == colinear_member.Id, 'OD_after'] = new_OD_after

                # Update colinear_ID so the next check uses the corrected value
                colinear_ID = target_ID
        # Check Inner Diameter´s
        if colinear_ID != target_ID:

            # Check if the member exists in final_diff
            if colinear_member.Id in final_diff['member'].values:
                target_THK = final_diff.loc[final_diff['member'] == colinear_member.Id, 'THK_after']
                # Update OD_after
                new_OD_after = target_ID + 2 * target_THK
                final_diff.loc[final_diff['member'] == colinear_member.Id, 'OD_after'] = new_OD_after
            else:
                target_THK = get_geo(colinear_member, "THK")
                new_OD_after = target_ID + 2 * target_THK
                # Add new row to final_diff
                print(get_geo(colinear_member, "FY")) 
                new_row = pd.DataFrame([{
                    'member': colinear_member.Id,
                    'group': colinear_member.group.Id,
                    'check_type': "ID Adjustment",
                    'OD_after': new_OD_after,
                    'OD_before': get_geo(colinear_member, "OD"),
                    'THK_after': target_THK,
                    'THK_before': get_geo(colinear_member, "THK"),
                    'SY_after': yield_strength(target_THK, material(target_THK, get_geo(colinear_member, "FY"))),
                    'SY_before': get_geo(colinear_member, "FY"),                    
                }])
                # Concatenate the new row with the final_diff DataFrame
                final_diff = pd.concat([final_diff, new_row], ignore_index=True)
    for cone_member_Id, cone in cones.items():
            if cone_member_Id == "105M-100B":
                a = 1
            final_diff = update_cones(final_diff, cone, target_ID) 

    return final_diff


def update_cones(final_diff, cone, target_ID):
    joint, cone_member, attached_member, right_side = cone
    original_THK = cone_member.THK * 10
    original_L_OD = cone_member.OD_L * 10
    original_S_OD = cone_member.OD_S * 10

    original_memb_OD = get_geo(attached_member, "OD")

    # Check if final_diff already contains cone updated sections
    condition1 = final_diff['member'] == cone_member.Id
    filtered_rows = final_diff[condition1]
    THK_after = filtered_rows['THK_after'].max()
    THK_after = np.nanmax([THK_after, original_THK])

    condition2 = (final_diff['member'] == cone_member.Id) & (final_diff['check_type'] == "Cone L Adjustment")
    filtered_rows = final_diff[condition2]
    target_L_OD = filtered_rows['OD_after'].max()
    target_L_OD = np.nanmax([target_L_OD, original_L_OD])

    condition3 = (final_diff['member'] == cone_member.Id) & (final_diff['check_type'] == "Cone S Adjustment")
    filtered_rows = final_diff[condition3]
    target_S_OD = filtered_rows['OD_after'].max()
    target_S_OD = np.nanmax([target_S_OD, original_S_OD])
    
    condition4 = (final_diff['member'] == attached_member.Id)
    filtered_rows = final_diff[condition4]
    target_memb_OD = filtered_rows['OD_after'].max()

    if joint == left(cone_member.Id, 4):   # Large Side
        OD_before = original_L_OD
        inner_aligned_OD = target_ID + 2 * THK_after # Inner Aligned
        OD_after = np.nanmax([inner_aligned_OD, target_L_OD]) 
        check_type = "Cone L Adjustment"
    else: # Small Side
        OD_before = original_S_OD
        OD_after = np.nanmax([target_ID,target_S_OD, target_memb_OD])  # Outer Aligned
        check_type = "Cone S Adjustment"
    print(get_geo(cone_member, "FY"))
    new_row = pd.DataFrame([{
        'member': cone_member.Id,
        'group': cone_member.group.Id,
        'check_type': check_type,
        'OD_after': OD_after,
        'OD_before': OD_before,
        'THK_after': THK_after,
        'THK_before': THK_after,
        'SY_after': yield_strength(THK_after, material(THK_after, get_geo(cone_member, "FY"))),
        'SY_before': get_geo(cone_member, "FY"),    
            }])
    # Remove existing rows with same member and check_type
    final_diff = final_diff[~((final_diff['member'] == cone_member.Id) & (final_diff['check_type'] == check_type))]
    # Concatenate new row
    final_diff = pd.concat([final_diff, new_row], ignore_index=True)
    return final_diff


def correct_model_IDs(model, final_diff, z_max):
    # Filter out LEG rows where group starts with 'L' and 'V'
    filtered_diff = final_diff[~final_diff['group'].str.startswith(('L', 'V'))]  # <<<<<<<<<<<<<<<<<<<<<<<< OD ALIGNED GROUPS
    filtered_diff.to_csv('PRE_ADDED_CONES.csv', index=False)

    # Iterate dynamically
    for idx in range(len(filtered_diff)):
        row = filtered_diff.iloc[idx]
        final_diff = split_member_add_cones(row, model, z_max, final_diff)

        # refresh filtered_diff because final_diff changed
        filtered_diff = final_diff[~final_diff['group'].str.startswith(('L', 'V'))]

    final_diff.to_csv('ADDED_CONES.csv', index=False)
    # Step 1: Remove rows where check_type contains "Introduced"
    final_diff = final_diff[~final_diff['check_type'].str.contains("Introduced", case=False, na=False)]
    # Step 2: Modify rows where check_type contains "Strip"
    mask_strip = final_diff['check_type'].str.contains("Strip", case=False, na=False)
    final_diff.loc[mask_strip, 'check_type'] = final_diff.loc[mask_strip, 'check_type'].str[15:]    
    final_diff.to_csv('BEFORE_INNER_ID.csv', index=False)
    filtered_diff = final_diff[~final_diff['group'].str.startswith(('L', 'V'))]  # <<<<<<<<<<<<<<<<<<<<<<<< OD ALIGNED GROUPS    

    for idx, row in filtered_diff.iterrows():
        final_diff = update_col_members(row, model, z_max, final_diff)
    final_diff.to_csv('AFTER_INNER_ID.csv', index=False)
    print(final_diff)

    return final_diff
    
def get_colinear_members(model, member, z_max=21.5, include_cones=False):
    member_id = member.Id
    central_member = member

    member_list = [central_member]
    joint_list = []
    cone_list = {}

    angle_tolerance = 1
    dist_tolerance = 0.1

    # True = right, False = left
    direction_flags = [True, False]

    for is_right_direction in direction_flags:

        current_member = central_member
        current_joint_id = right(central_member.Id, 4) if is_right_direction else left(central_member.Id, 4)
        current_joint = model.joints[current_joint_id]

        if is_right_direction:
            joint_list.append(current_joint)
        else:
            joint_list.insert(0, current_joint)

        angle_ = 0
        dist = 0

        # IMPORTANT: allow cones → remove istube from while condition
        while angle_ < angle_tolerance and dist < dist_tolerance:

            found_colinear = False
            cone_found = False
            n_members = len(current_joint.AttachedMembers)

            if n_members > 1 and current_joint.Z <= z_max:

                for member_i in current_joint.AttachedMembers:

                    if member_i.Id == current_member.Id:
                        continue

                    # Exclude legs unless central member is a leg
                    if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                        angle_ = 10
                        break

                    # Compute angle + distance
                    angle_ = math.degrees(member_angle_min(current_member, member_i))
                    dist = distance.euclidean(current_joint.coord, common_joint_coords(current_joint, member_i))

                    # Tube / cone flags
                    istube = member_i.is_tube
                    iscone = member_i.is_cone

                    # --- Cone handling ---
                    if iscone:
                        cone_found = True
                        i_direction = 1 if is_right_direction else 0
                        cone_list[member_i.Id] = [current_joint.Id, member_i, current_member, i_direction]

                        if include_cones:
                            # treat cone as colinear
                            found_colinear = True
                            next_member = member_i
                            next_joint_id = other_item(next_member.Id, current_joint.Id)
                            next_joint = model.joints[next_joint_id]
                            break
                        else:
                            # stop at cone
                            break

                    # --- Normal tubular colinear ---
                    if angle_ < angle_tolerance and dist < dist_tolerance and istube:
                        found_colinear = True
                        next_member = member_i
                        next_joint_id = other_item(next_member.Id, current_joint.Id)
                        next_joint = model.joints[next_joint_id]
                        break

                # --- Append next member/joint ---
                if found_colinear:
                    current_member = next_member
                    current_joint = next_joint

                    if is_right_direction:
                        member_list.append(current_member)
                        joint_list.append(current_joint)
                    else:
                        member_list.insert(0, current_member)
                        joint_list.insert(0, current_joint)

                else:
                    break

            else:
                break

    return [member_list, cone_list]

    
def get_colinear_members_backup_30072026(model, member, z_max=21.5, include_cones=False):
    member_id = member.Id
    is_right_direction  = True
    central_member = member
    current_joint = ""

    member_list = [central_member]  
    joint_list = []
    cone_list = {}

    angle_tolerance = 1
    dist_tolerance = 0.1

    direction_flags = [True, False]  # True = right, False = left

    for is_right_direction in direction_flags:
        current_member = central_member
        current_joint_id = right(central_member.Id, 4) if is_right_direction else left(central_member.Id, 4)
        current_joint = model.joints[current_joint_id]

        if is_right_direction:
            joint_list.append(current_joint)
        else:
            joint_list.insert(0, current_joint)

        dist = 0
        angle_ = 0
        istube = current_member.is_tube
        cone_found = False

        while angle_ < angle_tolerance and dist < dist_tolerance and istube:
            found_colinear = False
            skip_member = False
            n_members = len(current_joint.AttachedMembers)

            if n_members > 1 and current_joint.Z <= z_max:
                for member_i in current_joint.AttachedMembers:
                    shape_ = False

                    if member_i.Id != current_member.Id:

                        # Exclude legs unless central member is a leg
                        if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                            angle_ = 10
                            break

                        angle_ = math.degrees(member_angle_min(current_member, member_i))
                        dist = distance.euclidean(current_joint.coord, common_joint_coords(current_joint, member_i))
                        istube = member_i.is_tube

                        section_ = member_i.group.section
                        if section_:
                            shape_ = section_.stype

                        # Cone detection
                        if member_i.is_cone:
                            i_direction = 1 if is_right_direction else 0
                            cone_list[member_i.Id] = [current_joint.Id, member_i, current_member, i_direction]
                            cone_found = True

                            # NEW: include_cones option
                            if include_cones:
                                # treat cone as colinear and continue
                                found_colinear = True
                                next_member = member_i
                                next_joint_id = other_item(next_member.Id, current_joint.Id)
                                next_joint = model.joints[next_joint_id]
                                break
                            else:
                                # stop at cone (old behaviour)
                                break

                        # Normal colinear member
                        if angle_ < angle_tolerance and dist < dist_tolerance and istube:
                            found_colinear = True
                            next_member = member_i
                            next_joint_id = other_item(next_member.Id, current_joint.Id)
                            next_joint = model.joints[next_joint_id]
                            break

                # Append next member/joint if valid
                if found_colinear and not skip_member and not cone_found:
                    current_member = next_member
                    current_joint = next_joint

                    if is_right_direction:
                        member_list.append(current_member)
                        joint_list.append(current_joint)
                    else:
                        member_list.insert(0, current_member)
                        joint_list.insert(0, current_joint)

                # If cone found and include_cones=True → continue chain
                elif found_colinear and include_cones and cone_found:
                    current_member = next_member
                    current_joint = next_joint

                    if is_right_direction:
                        member_list.append(current_member)
                        joint_list.append(current_joint)
                    else:
                        member_list.insert(0, current_member)
                        joint_list.insert(0, current_joint)

                else:
                    break

            else:
                break

    return [member_list, cone_list]


# Update this function to find colinear members up to cone 
def get_colinear_members_old(model, member, z_max=21.5):
    member_id = member.Id
    is_right_direction  = True
    central_member = member
    current_joint = ""
   # Initialize lists to hold member and joint objects for clarity and explicit handling
    member_list = [central_member]  # Start with central member included
    joint_list = []
    cone_list = {}
    angle_tolerance = 1
    dist_tolerance = 0.1
  # Determine the direction flags for iteration
    direction_flags = [True, False]  # True for right, False for left
    

# CYCLE THROUGH COLINEAR MEMBERS TO THE RIGHT AND THEN TO THE LEFT "STOP AT CONES / JOINTS / ANGLES CHANGES"
    for is_right_direction in direction_flags:
        current_member = central_member
        current_joint_id = right(central_member.Id, 4) if is_right_direction else left(central_member.Id, 4)
        current_joint = model.joints[current_joint_id]
        joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)
        dist = 0
        angle_ = 0  # Reset angle for each direction
        istube = current_member.is_tube
        cone_found = False

        while angle_ < angle_tolerance and dist < dist_tolerance and istube:
            found_colinear = False
            skip_member = False
            n_members = len(current_joint.AttachedMembers)
            if n_members > 1 and current_joint.Z <= z_max:
                for member_i in current_joint.AttachedMembers:
                    shape_ = False
                    if member_i.Id != current_member.Id:
                        if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                            angle_ = 10  # Reset angle to exit loop
                            break
                        angle_ = math.degrees(member_angle_min(current_member, member_i))
                        dist = distance.euclidean(current_joint.coord , common_joint_coords(current_joint, member_i))
                        istube = member_i.is_tube
                        section_ = member_i.group.section
                        if section_:
                            shape_ = section_.stype
                        if member_i.is_cone:
                            i_direction = 1 if is_right_direction else 0 # is the cone on the left [0] or right side [1] of central member
                            cone_list[member_i.Id] = [current_joint.Id, member_i, current_member, i_direction]
                            cone_found = True
                        if angle_ < angle_tolerance and dist < dist_tolerance and istube:
                            found_colinear = True
                            next_member = member_i
                            next_joint_id = other_item(next_member.Id, current_joint.Id)
                            next_joint = model.joints[next_joint_id]
                            break
                if found_colinear and not skip_member and not cone_found:
                    current_member = next_member
                    current_joint = next_joint
                    member_list.append(current_member) if is_right_direction else member_list.insert(0, current_member)
                    joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)

                else:
                    # End the current direction iteration and break to switch direction or end loop
                    break
            else:
                # If no valid joint or member found, force exit from the loop
                break

    return [member_list, cone_list]

def calc_effective_length_params(model, member, eff_params, z_max, braces_to_avoid):
    is_right_direction  = True
    central_member = member
    current_joint = ""
   # Initialize lists to hold member and joint objects for clarity and explicit handling
    member_list = [central_member]  # Start with central member included
    joint_list = []
    angle_tolerance = 1
  # Determine the direction flags for iteration
    direction_flags = [True, False]  # True for right, False for left

# CYCLE THROUGH COLINEAR MEMBERS TO THE RIGHT AND THEN TO THE LEFT
    for is_right_direction in direction_flags:
        current_member = central_member
        current_joint_id = right(central_member.Id, 4) if is_right_direction else left(central_member.Id, 4)
        current_joint = model.joints[current_joint_id]
        joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)

        angle_ = 0  # Reset angle for each direction
        while angle_ < angle_tolerance:
            found_colinear = False
            skip_member = False
            if len(current_joint.AttachedMembers) > 1 and current_joint.Z <= z_max:
                for member_i in current_joint.AttachedMembers:
                    if member_i.Id != current_member.Id and (member_i.group.Id not in braces_to_avoid):
                        if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                            angle_ = 10  # Reset angle to exit loop
                            break
                        angle_ = math.degrees(member_angle_min(current_member, member_i))
                        if angle_ < angle_tolerance:
                            found_colinear = True
                            next_member = member_i
                            next_joint_id = other_item(next_member.Id, current_joint.Id)
                            next_joint = model.joints[next_joint_id]
                            break
                if found_colinear and not skip_member:
                    current_member = next_member
                    current_joint = next_joint
                    member_list.append(current_member) if is_right_direction else member_list.insert(0, current_member)
                    joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)
                else:
                    # End the current direction iteration and break to switch direction or end loop
                    break
            else:
                # If no valid joint or member found, force exit from the loop
                break

    memb_plan = get_plan(joint_list[0].coord, joint_list[-1].coord)

    # print(member.Id)
    # print([memb.Id for memb in member_list])
    # print([joint.Id for joint in joint_list])
    # print(memb_plan)

    if left(member.group.Id, 1) == "L" and ("_X_" not in memb_plan) and ("_Y_" not in memb_plan):
        is_leg = True
        target_ratio = 0.35
    else:
        is_leg = False
        target_ratio = 0.4
    
    My_constraints = []
    Mz_constraints = []
    #' Find hard point constraints only mid joints
    for joint in joint_list[1:-1]: 
        if len(joint.AttachedMembers) > 2:
            chdbrcs = get_joint_members(joint, braces_to_avoid)
            braces = chdbrcs['braces']
            chords = chdbrcs.get('chords', braces[:1] if braces else [])
            if chords:
                group, CHD_OD = chords[0].group.Id, get_geo(chords[0], "OD")
                n_xy_braces, is_plan_elevation = 0, False
                if is_leg:
                    for brace in braces:
                        member_plan = get_plan(*memb_coords(brace))
                        BRC_OD = get_geo(brace, "OD")
                        BRC_to_CHD = BRC_OD / CHD_OD if is_numerical(BRC_OD) else 0

                        if ("XY" in member_plan) and (BRC_to_CHD > target_ratio) and (left(brace.group.Id, 1) != "P") and is_leg:
                            n_xy_braces += 1
                    if n_xy_braces >= 2:
                        is_plan_elevation = True
       
                for brace in braces + chords:
                    brace_plan = common_plan(get_plan(*memb_coords(brace)), memb_plan)
                    BRC_OD = get_geo(brace, "OD")
                    BRC_to_CHD = BRC_OD / CHD_OD if is_numerical(BRC_OD) else 0

                    if (group not in braces_to_avoid):
                        # 'LEG CONNECTIONS
                        if is_leg and is_plan_elevation and (joint.Id not in Mz_constraints+My_constraints):
                            # print(joint.Id)
                            My_constraints.append(joint.Id)
                            Mz_constraints.append(joint.Id)
                        elif (BRC_to_CHD > target_ratio) and (not is_leg): 
                            IP_proj_angle = projection_angle(brace, member, get_planes(memb_plan, 0))
                            OP1_proj_angle = projection_angle(brace, member, get_planes(memb_plan, 1))
                            OP2_proj_angle = projection_angle(brace, member, get_planes(memb_plan, 2))
                            # NOTE: The bellow code relfects the default SACS local CSYS orientation! 
                            if right(memb_plan, 2) == "XY": 
                                if (IP_proj_angle > 30) and (joint.Id not in Mz_constraints):
                                    Mz_constraints.append(joint.Id)
                                if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in My_constraints):
                                    My_constraints.append(joint.Id)
                            else:
                                if (IP_proj_angle > 30) and (joint.Id not in My_constraints):
                                    My_constraints.append(joint.Id)
                                if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in Mz_constraints):
                                    Mz_constraints.append(joint.Id)
    # ******************* CALCULATE ACTUAL LENGTHS BETWEEN CONSTRAINTS *************************
    # print(Mz_constraints)
    # print(Mz_constraints)
    # ' >>>>>>>>>> K Factors! <<<<<<<<<<<
    kfactor = 1 if is_leg else 0.8

    # Initialize starting positions and lengths
    initial_joint = joint_list[0].coord
    left_joint_y = left_joint_z = initial_joint
    Ly = Lz = 0
    my_left = mz_left = 0
    canti_f = 1
    # Initiate member dictionary
    for memb in member_list:
        eff_params[memb.Id] = {"KorL": "L", "K_L_Y": Ly, "K_L_Z": Lz} 

    # Loop through joints to calculate effective lengths
    for j, joint in enumerate(joint_list):
        m= max(j-1,0)
        # Update cantilever factor unless previously set to 2
        if canti_f != 2: canti_f = cantilever_factor(model, member_list[m]) 
        # Handle effective length in Z-direction
        if (joint.Id in Mz_constraints) or (joint.Id == joint_list[-1].Id):
            right_joint_z = joint.coord
            mz_right = m
            Lz = distance.euclidean(left_joint_z, right_joint_z) * kfactor * canti_f
            for k in range(mz_left, mz_right+1):
                eff_params[member_list[k].Id]["K_L_Z"] = Lz
            left_joint_z, mz_left = right_joint_z, mz_right+1
        # Handle effective length in Y-direction
        if joint.Id in My_constraints or (joint.Id == joint_list[-1].Id):
            right_joint_y = joint.coord
            my_right = m
            Ly = distance.euclidean(left_joint_y, right_joint_y) * kfactor * canti_f
            for k in range(my_left, my_right+1):
                eff_params[member_list[k].Id]["K_L_Y"] = Ly
            left_joint_y, my_left = right_joint_y, my_right+1

    return eff_params

def calc_effective_length_scan(model, member, z_max, skip_exact, skip_prefix):
    is_right_direction  = True
    
    central_member = member

   # Initialize lists to hold member and joint objects for clarity and explicit handling
    member_list = [central_member]  # Start with central member included
    joint_list = []
    angle_tolerance = 1
  # Determine the direction flags for iteration
    direction_flags = [True, False]  # True for right, False for left

# CYCLE THROUGH COLINEAR MEMBERS TO THE RIGHT AND THEN TO THE LEFT
    for is_right_direction in direction_flags:
        current_member = central_member
        current_joint_id = right(central_member.Id, 4) if is_right_direction else left(central_member.Id, 4)
        current_joint = model.joints[current_joint_id]
        joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)

        angle_ = 0  # Reset angle for each direction
        while angle_ < angle_tolerance:
            found_colinear = False
            skip_member = False
            if len(current_joint.AttachedMembers) > 1 and current_joint.Z <= z_max:
                for member_i in current_joint.AttachedMembers:
                    ignore_brace = member_i.group_id in skip_exact and any(member_i.group_id.startswith(pref) for pref in skip_prefix)    
                    if member_i.Id != current_member.Id and (not ignore_brace):
                        if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                            angle_ = 10  # Reset angle to exit loop
                            break
                        angle_ = math.degrees(member_angle_min(current_member, member_i))
                        if angle_ < angle_tolerance:
                            found_colinear = True
                            next_member = member_i
                            next_joint_id = other_item(next_member.Id, current_joint.Id)
                            next_joint = model.joints[next_joint_id]
                            break
                if found_colinear and not skip_member:
                    current_member = next_member
                    current_joint = next_joint
                    member_list.append(current_member) if is_right_direction else member_list.insert(0, current_member)
                    joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)
                else:
                    # End the current direction iteration and break to switch direction or end loop
                    break
            else:
                # If no valid joint or member found, force exit from the loop
                break

    memb_plan = get_plan(joint_list[0].coord, joint_list[-1].coord)

    group_id = member.group.Id
    if group_id[1] == "L" and ("_X_" not in memb_plan) and ("_Y_" not in memb_plan):
        is_leg = True
        target_ratio = 0.35
    else:
        is_leg = False
        target_ratio = 0.4
    
    is_jtube = group_id.startswith("JT")
    is_xbrace = group_id.startswith("X")

    joint_ids = [joint.Id for joint in joint_list]
    
    My_constraints = []
    Mz_constraints = []
    #' Find hard point constraints only mid joints
    for joint in joint_list[1:-1]: 
        if len(joint.AttachedMembers) > 2:
            chdbrcs = get_joint_members(joint, skip_exact, skip_prefix)
            braces = chdbrcs['braces']
            chords = chdbrcs.get('chords', braces[:1] if braces else [])
            if chords:
                group, CHD_OD = chords[0].group.Id, get_geo(chords[0], "OD")
                n_xy_braces, is_plan_elevation = 0, False
                if is_leg:
                    for brace in braces:
                        member_plan = get_plan(*memb_coords(brace))
                        BRC_OD = get_geo(brace, "OD")
                        BRC_to_CHD = BRC_OD / CHD_OD if is_numerical(BRC_OD) else 0

                        if ("XY" in member_plan) and (BRC_to_CHD > target_ratio) and (left(brace.group.Id, 1) != "P") and is_leg:
                            n_xy_braces += 1
                    if n_xy_braces >= 2:
                        is_plan_elevation = True
       
                for brace in braces + chords:
                    brace_plan = common_plan(get_plan(*memb_coords(brace)), memb_plan)
                    BRC_OD = get_geo(brace, "OD")
                    BRC_to_CHD = BRC_OD / CHD_OD if is_numerical(BRC_OD) else 0
                    ignore_member = group in skip_exact and any(group.startswith(pref) for pref in skip_prefix)    
                    if not ignore_member:
                        # 'LEG CONNECTIONS
                        if is_leg and is_plan_elevation and (joint.Id not in Mz_constraints+My_constraints):
                            # print(joint.Id)
                            My_constraints.append(joint.Id)
                            Mz_constraints.append(joint.Id)
                        elif (BRC_to_CHD > target_ratio) and (not is_leg): 
                            IP_proj_angle = projection_angle(brace, member, get_planes(memb_plan, 0))
                            OP1_proj_angle = projection_angle(brace, member, get_planes(memb_plan, 1))
                            OP2_proj_angle = projection_angle(brace, member, get_planes(memb_plan, 2))
                            # NOTE: The bellow code relfects the default SACS local CSYS orientation! 
                            if right(memb_plan, 2) == "XY": 
                                if (IP_proj_angle > 30) and (joint.Id not in Mz_constraints):
                                    Mz_constraints.append(joint.Id)
                                if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in My_constraints):
                                    My_constraints.append(joint.Id)
                            else:
                                if (IP_proj_angle > 30) and (joint.Id not in My_constraints):
                                    My_constraints.append(joint.Id)
                                if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in Mz_constraints):
                                    Mz_constraints.append(joint.Id)
    # ******************* CALCULATE ACTUAL LENGTHS BETWEEN CONSTRAINTS *************************
    if member.Id == "3575-3512":
        a = 1
    My_constraints_ = get_effective_BCs(model, central_member, joint_ids, My_constraints)
    Mz_constraints_ = get_effective_BCs(model, central_member, joint_ids, Mz_constraints)

    # Left Side
    canti_y = model.joints[Mz_constraints_[0]].is_cantilever
    canti_z = model.joints[Mz_constraints_[1]].is_cantilever
    canti_f_y = 2 if canti_y else 1
    canti_f_z = 2 if canti_z else 1
    # Handle effective length in Z-direction
    left_joint_coord_y = model.joints[My_constraints_[0]].coord
    left_joint_coord_z = model.joints[Mz_constraints_[0]].coord
    right_joint_coord_y = model.joints[My_constraints_[1]].coord
    right_joint_coord_z = model.joints[Mz_constraints_[1]].coord
    
    Ly = distance.euclidean(left_joint_coord_y, right_joint_coord_y)  * canti_f_y
    Lz = distance.euclidean(left_joint_coord_z, right_joint_coord_z)  * canti_f_z

  
    


    if is_leg or is_jtube: 
        ky, kz = 0.9, 0.9
    elif is_xbrace:
        ky, kz = 0.8, 0.8
    else:
        ky, kz = 0.75, 0.75                 

    Ly_final = Ly * ky
    Ly_final = Lz * kz

    return Ly_final, Ly_final

# def calc_effective_length_params_original(model, member, eff_params, z_max, braces_to_avoid):
#     # member = model.FindMember("2113-2143")
#     right_iteration = True
#     left_iteration = False
#     central_member = member
#     current_joint = ""
#     member_list = []
#     joint_list = []
#     angle_tolerance = 1


#     # CYCLE THROUGH COLINEAR MEMBERS TO THE RIGHT AND THNE TO THE LEFT
#     for k in range(2):
#         angle_ = 0
#         current_member = member
#         # Initiate right or left iteration flag variables
#         if right_iteration:
#             current_joint = model.FindJoint(right(central_member.Id, 4))
#             member_list.append(central_member)
#             joint_list.append(current_joint)
#         elif left_iteration:
#             current_joint = model.FindJoint(left(central_member.Id, 4))
#             joint_list.insert(0, current_joint) 
        
#         # Main Loop
#         while (angle_ < angle_tolerance) and (right_iteration or left_iteration):
#             found_colinear = False
#             skip_member = False
#             angle_ = 10
#             if len(current_joint.AttachedMembers) > 1 and current_joint.Coordinate.Z <= z_max:
#                 for member_i in current_joint.AttachedMembers:
#                     if member_i.Id != current_member.Id:
#                         if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L": # Exit colinear selection if leg element found
#                             angle_ = 10
#                         else:
#                             angle_ = math.degrees(member_angle_min(current_member, member_i))
        
#                     if angle_ < angle_tolerance: 
#                         found_colinear = True
#                         next_member = member_i
#                         next_joint = model.FindJoint(other_item(next_member.Id, current_joint.Id))
#                         break
#             # ' MEMBER ENDS AS "CANTILEVER"
#             else: 
#                 angle_ = 10 # force loop exit
#                 skip_member = True

#             if found_colinear and not skip_member:
#                 current_member = next_member
#                 current_joint = next_joint
#                 if right_iteration:
#                     member_list.append(current_member)
#                     joint_list.append(current_joint)
#                 else:
#                     member_list.insert(0, current_member)
#                     joint_list.insert(0, current_joint)
#             else:
#                 if right_iteration and left_iteration == False:
#                     right_iteration = False
#                     left_iteration = True
#                     break
#                 else:
#                     left_iteration = False 

#     memb_plan = get_plan(joint_list[0].Coordinate.Values, joint_list[-1].Coordinate.Values)

#     print(member.Id)
#     print([memb.Id for memb in member_list])
#     print([joint.Id for joint in joint_list])
#     print(memb_plan)

#     if left(member.group.Id, 1) == "L" and ("_X_" not in memb_plan) and ("_Y_" not in memb_plan):
#         is_leg = True
#         target_ratio = 0.35
#     else:
#         is_leg = False
#         target_ratio = 0.4

#     # total_length = distance.euclidean(joint_list[0].Coordinate.Values, joint_list[-1].Coordinate.Values)
    
#     My_constraints = []
#     Mz_constraints = []
#     #' Find hard point constraints only mid joints
#     for joint in joint_list[1:-1]: 
#         if len(joint.AttachedMembers) > 2:
#             chdbrcs = get_joint_members(joint, braces_to_avoid)
#             braces = chdbrcs["braces"]
#             chords = chdbrcs["chords"]
#             if len(chords) == 0 and len(braces) > 0:
#                 chords = [braces[0]]
#             if len(chords) != 0:    
#                 group = chords[0].group.Id
#                 CHD_OD = get_geo(chords[0], "OD")
#                 # Main Plan Test
#                 if is_leg:
#                     n_xy_braces = 0
#                     is_plan_elevation = False
#                     for brace in braces:
#                         member_plan = get_plan(*memb_coords(brace))
#                         group = brace.group.Id
#                         BRC_OD = get_geo(brace, "OD")
#                         if is_numerical(BRC_OD):  
#                             BRC_to_CHD = BRC_OD / CHD_OD
#                         else:
#                             BRC_to_CHD = 0
#                         #  ' <<<<<<<< Last test to avoid pile cluster "braces"
#                         if ("XY" in member_plan) and (BRC_to_CHD > target_ratio) and (left(group, 1) != "P"): 
#                             # print(member.Id)
#                             n_xy_braces += 1
#                     if n_xy_braces >= 2: is_plan_elevation = True 
#                 for brace in (braces + chords):
#                     brace_plan = get_plan(*memb_coords(brace))
#                     brace_plan = common_plan(brace_plan, memb_plan)
#                     BRC_OD = get_geo(brace, "OD")
#                     if is_numerical(BRC_OD):  
#                         BRC_to_CHD = BRC_OD / CHD_OD
#                     else:
#                         BRC_to_CHD = 0
                    
#                     if (group not in braces_to_avoid):
#                         # 'LEG CONNECTIONS
#                         if is_leg and is_plan_elevation and (joint.Id not in Mz_constraints) and (joint.Id not in My_constraints) :
#                             # print(joint.Id)
#                             My_constraints.append(joint.Id)
#                             Mz_constraints.append(joint.Id)
#                         elif (BRC_to_CHD > target_ratio) and (not is_leg):
#                             IP_proj_angle = projection_angle(brace, member, get_planes(brace_plan, 0), brace_plan)
#                             OP1_proj_angle = projection_angle(brace, member, get_planes(brace_plan, 1), brace_plan)
#                             OP2_proj_angle = projection_angle(brace, member, get_planes(brace_plan, 2), brace_plan)
#                             if right(memb_plan, 2) == "XY":
#                                 if (IP_proj_angle > 30) and (joint.Id not in Mz_constraints):
#                                     Mz_constraints.append(joint.Id)
#                                 if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in My_constraints):
#                                     My_constraints.append(joint.Id)
#                             else:
#                                 if (IP_proj_angle > 30) and (joint.Id not in My_constraints):
#                                     My_constraints.append(joint.Id)
#                                 if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in Mz_constraints):
#                                     Mz_constraints.append(joint.Id)
#     # ******************* CALCULATE ACTUAL LENGTHS BETWEEN CONSTRAINTS *************************
#     print(Mz_constraints)
#     print(Mz_constraints)
#     # ' >>>>>>>>>> K Factors! <<<<<<<<<<<
#     if is_leg == True:
#         kfactor = 1
#     else:
#         kfactor = 0.8
#     # Initiate variables
#     left_joint_y = joint_list[0].Coordinate.Values
#     left_joint_z = joint_list[0].Coordinate.Values
#     Ly = 0
#     Lz = 0
#     my_left = 0
#     mz_left = 0
#     canti_f = 1
#     # Initiate member dictionary
#     for memb in member_list:
#         if memb.Id in eff_params: print(memb.Id + "already in list!!!")
#         eff_params[memb.Id] = {"KorL": "L", "K_L_Y": Ly, "K_L_Z": Lz} 

#     # Loop through joints to calculate effective lengths
#     for j, joint in enumerate(joint_list):
#         m= max(j-1,0)
#         # Update cantilever factor unless previously set to 2
#         if canti_f != 2: canti_f = cantilever_factor(model, member_list[m]) 
#         # Handle effective length in Z-direction
#         if (joint.Id in Mz_constraints) or (joint.Id == joint_list[-1].Id):
#             right_joint_z = joint.coord
#             mz_right = m
#             Lz = distance.euclidean(left_joint_z, right_joint_z) * kfactor * canti_f
#             for k in range(mz_left, mz_right+1):
#                 eff_params[member_list[k].Id]["K_L_Z"] = Lz
#             left_joint_z, mz_left = right_joint_z, mz_right
#         # Handle effective length in Y-direction
#         if joint.Id in My_constraints or (joint.Id == joint_list[-1].Id):
#             right_joint_y = joint.coord
#             my_right = m
#             Ly = distance.euclidean(left_joint_y, right_joint_y) * kfactor * canti_f
#             for k in range(my_left, my_right+1):
#                 eff_params[member_list[k].Id]["K_L_Y"] = Ly
#             left_joint_y, my_left = right_joint_y, my_right
    
#     return eff_params

def cantilever_factor(model, member):
    end1_cantilever = model.joints[left(member.Id, 4)].AttachedMembers == 1
    end2_cantilever = model.joints[right(member.Id, 4)].AttachedMembers == 1
    if end1_cantilever or end2_cantilever:
        cantilever_factor = 2
    else:
        cantilever_factor = 1
    return cantilever_factor

def cantilever_factor_joint(model, joint_id):
    end_cantilever = len(model.joints[joint_id].AttachedMembers) == 1
    if end_cantilever:
        cantilever_factor = 2
    else:
        cantilever_factor = 1
    return cantilever_factor

def projection_angle(member1, member2, plan):
    if (right(plan, 2) == "YZ"):
        k0 = 0
    elif right(plan, 2) == "XZ":
        k0 = 1
    elif right(plan, 2) == "XY":
        k0 = 2
    memb1_v = memb_vect(*memb_coords(member1))
    memb2_v = memb_vect(*memb_coords(member2))

    memb1_v[k0] = 0
    memb2_v[k0] = 0
    projection_angle = math.degrees(vect_angle_min(memb1_v, memb2_v))
    return projection_angle

def projection_angle_old(member1, member2, plan, chord_plan):
    if ("DD" in chord_plan):
        # 'Guarantee plan assumption if member diagonal
        if (right(chord_plan, 2) == "YZ"):
            m0 = 0
        elif right(chord_plan, 2) == "XZ":
            m0 = 1
        elif right(chord_plan, 2) == "XY":
            m0 = 2
    if (right(plan, 2) == "YZ"):
        k0 = 0
    elif right(plan, 2) == "XZ":
        k0 = 1
    elif right(plan, 2) == "XY":
        k0 = 2
    memb1_v = memb_vect(*memb_coords(member1))
    memb2_v = memb_vect(*memb_coords(member2))
  
    #   'PROJECT ON TO PLANE
    if ("DD" in chord_plan):
        memb1_v[m0] = 0
        memb2_v[m0] = 0
    memb1_v[k0] = 0
    memb2_v[k0] = 0
    projection_angle_old = math.degrees(vect_angle_min(memb1_v, memb2_v))
    return projection_angle_old   

def get_plan(coords1, coords2):
    get_plan = ""
    dx = abs(coords1[0] - coords2[0])
    dy = abs(coords1[1] - coords2[1])
    dz = abs(coords1[2] - coords2[2])
    
    if dx >= 0.01: get_plan = get_plan + "_X"
    if dy >= 0.01: get_plan = get_plan + "_Y"
    if dz >= 0.01: get_plan = get_plan + "_Z"
    
    if dz < 0.01: get_plan = get_plan + "_XY"
    if dy < 0.01: get_plan = get_plan + "_XZ"
    if dx < 0.01: get_plan = get_plan + "_YZ"

    if (dx >= 0.01 and dy >= 0.01 and dz >= 0.01): 
        get_plan = get_plan + "_DD"
        # '        ' Approximate plane
        if min(dx, dy, dz) == dx: get_plan = get_plan + "_YZ"
        elif min(dx, dy, dz) == dy:
            get_plan = get_plan + "_XZ"
        elif min(dx, dy, dz) == dz:
            get_plan = get_plan + "_XY"
    return get_plan

def common_plan(plan1, plan2):
    common_plan = ""
    found_common_plan = False     
    for val in plan1.split("_"):
        if len(val) == 1:
            common_plan = common_plan + "_" + val
        if len(val) == 2 and (val in plan2):
            common_plan = common_plan + "_" + val
            found_common_plan = True
    if (not found_common_plan):
        common_plan = common_plan + "_" + right(plan1, 2)
    return common_plan

def get_planes(plan, k):
    # Define the dictionary to map plan names to their corresponding lists
    planes = {
        "XY": ["XY", "XZ", "YZ"],
        "XZ": ["XZ", "XY", "YZ"],
        "YZ": ["YZ", "XY", "XZ"]
    }
    return planes[right(plan,2)][k]

def other_item(key, sub_key):
    length = len(sub_key)
    if left(key, length) == sub_key:
        return right(key, length)
    else:
        return left(key, length)


def add_X_braces(df):
    # Step 1: Group by 'joint'
    grouped = df.groupby('joint')

    # Initialize a list to store new rows
    new_rows = []

    # Step 2: Process each joint group
    for joint, group in grouped:
        # Step 2.1: Group by 'type' and 'plan' to find pairs
        print(f'X-Brace Testing {joint}')
        type_plan_groups = group.groupby(['type', 'plan', 'loads'])

        for (type, plan, loadcase), type_plan_group in type_plan_groups:
            # print(type, plan, loadcase, len(type_plan_group))
            if len(type_plan_group) == 2:
                # Step 3: Combine the two rows, making sure to copy them
                row1, row2 = type_plan_group.iloc[0].copy(), type_plan_group.iloc[1].copy()
                if row1["type"] == "T" and row2["type"] == "T":
                    row1["type"], row2["type"] = "XT", "XT"
                else:
                    row1["type"], row2["type"] = "XK", "XK"
                # Add the new row to the list
                new_rows.append(row1)
                new_rows.append(row2)
    
    # Step 4: Create a new DataFrame from the new rows
    new_rows_df = pd.DataFrame(new_rows)

    # Step 5: Concatenate the new rows DataFrame with the original DataFrame
    df = pd.concat([df, new_rows_df], ignore_index=True)

    return df

def generate_new_group_vectorized(all_groups, final_diff):
    import string
    import numpy as np

    # Convert all_groups to a set for fast lookup and ensure all are exactly 3 characters
    all_groups_set = {group[:3] for group in all_groups}  # Ensure existing groups are 3 characters

    # Extract the first 3 characters from the 'group' column of final_diff
    base_names = final_diff['group']# Ensure all base names are exactly 3 characters

    # Ensure all base names are unique and valid
    new_groups = []
    alphanum_chars = string.ascii_uppercase + string.digits

    for base_name in base_names:
        new_group_id = None

        for char1 in alphanum_chars:
            for char2 in alphanum_chars:
                candidate = f"{base_name[:1]}{char1}{char2}"  # Generate exactly 3-character IDs
                if candidate not in all_groups_set:
                    new_group_id = candidate
                    all_groups_set.add(candidate)  # Add the candidate to avoid future duplicates
                    break
            if new_group_id is not None:
                break

        if new_group_id is None:
            raise ValueError(f"Could not generate a unique group ID for base name {base_name}")

        new_groups.append(new_group_id)

    # Assign new_groups to the new column in final_diff
    final_diff['new_group'] = np.array(new_groups)

    # final_diff.to_csv('test3.csv', index=False) 
    return np.array(new_groups)



    
def generate_new_cone_group_vectorized(all_groups, final_diff):
    import string
    import numpy as np

    all_groups_set = set(all_groups)
    base_names = final_diff['group'].str[0]  # Extract base names from the group column

    # Determine which groups need updating
    update_group = (
        (final_diff['OD_L_after'] != final_diff['OD_L_before']) |
        (final_diff['OD_S_after'] != final_diff['OD_S_before']) |
        (final_diff['THK_after'] != final_diff['THK_before'])
    )

    new_groups = []
    alphanum_chars = string.ascii_uppercase + string.digits

    for i, base_name in enumerate(base_names):
        if update_group.iloc[i]:
            new_group_id = None

            for char1 in alphanum_chars:
                for char2 in alphanum_chars:
                    candidate = f"{base_name}{char1}{char2}"  # Generate candidate ID like L00, L01, ..., L99, LA0, ..., LZZ
                    if candidate not in all_groups_set:
                        new_group_id = candidate
                        all_groups_set.add(candidate)  # Add the candidate to the set to avoid future duplicates
                        break
                if new_group_id is not None:
                    break

            if new_group_id is None:
                raise ValueError(f"Could not generate a unique group ID for base name {base_name}")

            new_groups.append(new_group_id)
        else:
            # Retain the existing group name if no update is needed
            new_groups.append(final_diff['group'].iloc[i])

    # Assign new_groups to the new column in final_diff
    final_diff['new_group'] = np.array(new_groups)

    return final_diff


# def generate_new_group_vectorized_original(groups, all_groups):
#     all_groups_set = set(all_groups)
#     base_names = groups.str[0]

#     new_groups = []
#     alphanum_chars = string.ascii_uppercase + string.digits

#     for base_name in base_names:
#         new_group_id = None

#         for char1 in alphanum_chars:
#             for char2 in alphanum_chars:
#                 candidate = f"{base_name}{char1}{char2}"  # Generate candidate ID like L00, L01, ..., L99, LA0, ..., LZZ
#                 if candidate not in all_groups_set:
#                     new_group_id = candidate
#                     all_groups_set.add(candidate)  # Add the candidate to the set to avoid future duplicates
#                     break
#             if new_group_id is not None:
#                 break
#             # If no unique ID could be generated, move to the next base name
#             if new_group_id is None:
#                 next_base_name = chr((ord(base_name) + 1 - 65) % 26 + 65)  # Get the next base name (A-Z)
#                 for i in range(100):
#                     candidate = f"{next_base_name}{i:02d}"  # Generate candidate ID with the new base name
#                     if candidate not in all_groups_set:
#                         new_group_id = candidate
#                         all_groups_set.add(candidate)  # Add the candidate to the set to avoid future duplicates
#                         break

#         if new_group_id is None:
#             raise ValueError(f"Could not generate a unique group ID for base name {base_name}")

#         new_groups.append(new_group_id)

#     return np.array(new_groups)


def generate_new_joint(joint_Id, all_joints):
    base_char = joint_Id[0]
    suffix_number = 1  # Start with numerical suffix
    suffix_letter = 'A'  # Start with alphabetical suffix if needed

    while True:
        # Generate a numeric joint name first
        new_joint = f"{base_char}{suffix_number:03d}"
        if new_joint not in all_joints:
            all_joints.append(new_joint)  # Modify the list directly
            return new_joint

        # Switch to alphabetic suffix if numeric suffixes are exhausted
        if suffix_number >= 999:  # Arbitrary limit for numbers
            new_joint = f"{base_char}{suffix_letter}"
            if new_joint not in all_joints:
                all_joints.append(new_joint)  # Modify the list directly
                return new_joint
            suffix_letter = chr(ord(suffix_letter) + 1)  # Increment to the next letter
        else:
            suffix_number += 1  # Increment the numeric suffix

def generate_new_group(group_Id, all_groups):
    base_char = group_Id[0]
    suffix_number = 1  # Start with numerical suffix
    suffix_letter = 'A'  # Start with alphabetical suffix if needed
    iteration = 0  # Track iterations to avoid infinite loops
    
    while True:
        # Generate a numeric group name first
        new_group = f"{base_char}{suffix_number:02d}"
        if new_group not in all_groups:
            return new_group
        
        # Switch to alphabetic suffix if numeric suffixes are exhausted
        if suffix_number >= 99:  # Arbitrary limit for numbers
            new_group = f"{base_char}{suffix_letter}{iteration if iteration > 0 else ''}"
            if new_group not in all_groups:
                return new_group
            
            suffix_letter = chr(ord(suffix_letter) + 1)  # Increment to the next letter
            
            # Reset to 'A' and increment iteration if beyond 'Z'
            if suffix_letter > 'Z':
                suffix_letter = 'A'
                iteration += 1
        else:
            suffix_number += 1  # Increment the numeric suffix


def generate_new_section(old_section, all_sections):
    # Remove any non-alphabetic or non-numeric characters
    clean_section = re.sub(r'[^a-zA-Z0-9]', '', old_section)
    # If the cleaned section is empty, start with a default base
    if not clean_section:
        clean_section = 'Section'
    # Base part of the section name
    base_section = clean_section[:-1] if len(clean_section) > 1 else clean_section
    counter = ord('A')
    # Track attempts to ensure uniqueness
    iteration = 0
    while True:
        # Generate a new section name
        new_section = f"{base_section}{chr(counter)}{iteration if iteration > 0 else ''}"
        # Check if the new section name is unique
        if new_section not in all_sections:
            break
        # Increment the counter
        counter += 1
        # If counter exceeds 'Z', reset it and increment the iteration
        if counter > ord('Z'):
            counter = ord('A')
            iteration += 1
    return new_section


def update_flooded_groups(file_path, final_diff):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Function to find and update the line
    def find_and_update_line(lines, group, new_group):
        # Find the original group line and check the 70th column
        for i, line in enumerate(lines):
            
            if line.startswith(f'GRUP {group}'):
                if len(line) > 69 and line[69] == 'F':
                    # Find the new group line and update the 70th column
                    for j, new_line in enumerate(lines):
                        if new_line.startswith(f'GRUP {new_group}'):
                            # Ensure the line is long enough
                            if len(new_line) > 69:
                                lines[j] = new_line[:69] + 'F' + new_line[70:]
                            else:
                                lines[j] = new_line.rstrip('\n') + ' ' * (69 - len(new_line)) + 'F\n'
                            break
                break

    # Iterate over the DataFrame and update lines
    for index, row in final_diff.iterrows():
        group = row['group']
        new_group = row['new_group']
        find_and_update_line(lines, group, new_group)
    
    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)


def update_color_in_file(file_path, final_diff):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determine the color codes
    # color_code_1 = "255   0   0"   # Strong Bright Red for ULS failures
    # color_code_2 = "255   0   0"   # Strong Deep Red for FLS failures
    # color_code_3 = "255  10   0"   # Deep Blue for ID adjustments
    # color_code_4 = "255   0  10"   # Deep Green for secondary ID adjustments
    color_code_1 = "255   0   0"   # Strong Bright Red for ULS failures
    color_code_2 = "200   0   0"   # Strong Deep Red for FLS failures
    color_code_3 = "128   0 128"   # Deep Purple for ID adjustments
    color_code_4 = "255  20 147"   # Deep Pink for secondary ID adjustments

    # Function to create the color line
    def create_color_line(group, color_code):
        return f" **GCOL** {group}M{color_code}"

    # 2nd Last line
    end_index = len(lines) - 1

    # Iterate over the DataFrame and create color lines
    for index, row in final_diff.iterrows():
        new_group = row['new_group']
        color_code = None

        if "ULS" in row["check_type"]:
            color_code = color_code_1
        elif "FLS" in row["check_type"]:
            color_code = color_code_2
        elif row["check_type"] == "ID Adjustment":
            color_code = color_code_3
        elif "Cone" in row["check_type"]:
            color_code = color_code_4


        if color_code:
            color_line = create_color_line(new_group, color_code)
            lines.insert(end_index, color_line + "\n")
            end_index += 1  # Adjust the end_index to account for the newly inserted line
    
    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def color_groups_and_splash(model, file_path):
    """
    Colors:
      - All groups whose ID starts with 'G'      → Pink
      - All groups whose ID starts with 'JT'     → Light Blue
      - All groups intersecting splash zone      → Green

    Uses:
      - group IDs directly from model.groups keys
      - member.group_id for splash zone
    """

    # --- Color codes ---
    pink        = "255  20 147"   # For G*
    light_blue  = "135 206 250"   # For JT*
    green       = "  0 255   0"   # Splash zone

    # Splash zone limits
    z_min = -4.4
    z_max = 8.3

    # Read file
    with open(file_path, "r") as f:
        lines = f.readlines()

    insert_index = len(lines) - 1

    def create_color_line(group_id, color_code):
        return f" **GCOL** {group_id}M{color_code}"

    # ---------------------------------------------------------
    # STEP 1 — Groups starting with G (using dict keys)
    # ---------------------------------------------------------
    g_groups = {gid for gid in model.groups.keys() if gid.startswith("G")}

    # ---------------------------------------------------------
    # STEP 2 — Groups starting with JT (using dict keys)
    # ---------------------------------------------------------
    jt_groups = {gid for gid in model.groups.keys() if gid.startswith("JT")}

    # ---------------------------------------------------------
    # STEP 3 — Splash-zone groups (using member.group_id)
    # ---------------------------------------------------------
    splash_groups = set()

    for member in model.members.values():
        z1 = member.coord1[2]
        z2 = member.coord2[2]

        if (z_min <= z1 <= z_max) and (z_min <= z2 <= z_max):
            splash_groups.add(member.group_id)

    # ---------------------------------------------------------
    # STEP 4 — Insert colors in correct order
    # ---------------------------------------------------------


    # 3) Splash zone → Green
    for gid in sorted(splash_groups):
        lines.insert(insert_index, create_color_line(gid, green) + "\n")
        insert_index += 1

        # 1) G groups → Pink
    for gid in sorted(g_groups):
        lines.insert(insert_index, create_color_line(gid, pink) + "\n")
        insert_index += 1

    # 2) JT groups → Light Blue
    for gid in sorted(jt_groups):
        lines.insert(insert_index, create_color_line(gid, light_blue) + "\n")
        insert_index += 1

    # Write updated file
    with open(file_path, "w") as f:
        f.writelines(lines)



def color_by_plan(file_path, model, plans_df, tolerance=0.01):
    members = model.AllMembers()
    
    # Ensure coordinates are treated as numeric values
    plans_df[['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']] = plans_df[['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']].apply(pd.to_numeric, errors='coerce')

    # Dictionary to store unique groups using sets (prevents duplicates)
    plan_groups = {index: set() for index in plans_df.index}

    # Iterate over members first
    for memb in members:
        m_coord = memb_coords(memb)  # Expected as [[x1, y1, z1], [x2, y2, z2]]

        # Iterate over plan rows (less frequent loop)
        for index, plan in plans_df.iterrows():
            xmin, ymin, zmin = plan['xmin'], plan['ymin'], plan['zmin']
            xmax, ymax, zmax = plan['xmax'], plan['ymax'], plan['zmax']

            # Compact match conditions
            x_match = pd.isna(xmin) or pd.isna(xmax) or all((xmin - tolerance if pd.notna(xmin) else float('-inf')) <= x <= (xmax + tolerance if pd.notna(xmax) else float('inf')) for x in [m_coord[0][0], m_coord[1][0]])
            y_match = pd.isna(ymin) or pd.isna(ymax) or all((ymin - tolerance if pd.notna(ymin) else float('-inf')) <= y <= (ymax + tolerance if pd.notna(ymax) else float('inf')) for y in [m_coord[0][1], m_coord[1][1]])
            z_match = pd.isna(zmin) or pd.isna(zmax) or all((zmin - tolerance if pd.notna(zmin) else float('-inf')) <= z <= (zmax + tolerance if pd.notna(zmax) else float('inf')) for z in [m_coord[0][2], m_coord[1][2]])

            if x_match and y_match and z_match:
                plan_groups[index].add(memb.group.Id)  # Add to set to prevent duplicates

    # Iterate over plan_groups and apply colors
    for index, group_list in plan_groups.items():
        if group_list:  # Only process non-empty groups
            color_code = plans_df.at[index, 'color']  # Retrieve the corresponding color
            color_by_group_list(file_path, list(group_list), color_code)  # Convert set back to list for processing

SUFFIX_CHARS = string.digits + string.ascii_uppercase  # 0-9 then A-Z

def generate_unique_group(existing_groups, original_id):
    """
    Generate a unique 3-character group id based on original_id.

    1. Keep the first two characters and try a third of 0-9, then A-Z.
    2. If all of those are taken, keep only the first character and try
       every two-character combination of 0-9 and A-Z.

    The new id is added to existing_groups (a set) before it is returned.
    """
    if not original_id:
        raise ValueError("original_id must not be empty")

    # Stage 1: preserve the first two characters
    if len(original_id) >= 2:
        prefix2 = original_id[:2]
        for c in SUFFIX_CHARS:
            new_id = prefix2 + c
            if new_id not in existing_groups:
                existing_groups.add(new_id)
                return new_id

    # Stage 2: preserve only the first character
    prefix1 = original_id[0]
    for a, b in itertools.product(SUFFIX_CHARS, repeat=2):
        new_id = prefix1 + a + b
        if new_id not in existing_groups:
            existing_groups.add(new_id)
            return new_id

    raise ValueError(f"Ran out of unique group names for '{original_id}'")

def generate_unique_group_old2(existing_groups, original_id):
    """Generate a unique group name with max 3 alphanumeric characters, keeping TW prefix when possible."""

    # Check if the original_id starts with "TW"
    if original_id.startswith("TW") and len(original_id) >= 3:
        base_prefix = "TW"
        numeric_part = original_id[2:]
    elif original_id[0].isalpha():
        base_prefix = original_id[0].upper()
        numeric_part = original_id[1:]
    else:
        base_prefix = "A"  # Default to 'A' if no valid prefix exists
        numeric_part = ""

    # Generate unique ID
    while True:
        for num in range(100):  # 00-99
            new_id = f"{base_prefix}{num:02d}"[-3:]  # Ensure max 3 characters
            if new_id not in existing_groups:
                existing_groups.add(new_id)
                return new_id

        # If we reach X99 or TW99, increment the letter(s)
        if base_prefix == "TW":
            raise ValueError("Ran out of TW group names!")  # Limited to TW00-TW99
        elif len(base_prefix) == 1 and base_prefix in string.ascii_uppercase:
            base_prefix = chr(ord(base_prefix) + 1)  # Move to next letter
        else:
            raise ValueError("Ran out of unique group names!")  # Should never happen


def generate_unique_group_old(existing_groups, original_id):
    """Generate a unique group name with max 3 alphanumeric characters, rolling over letters when needed."""
    base_letter = original_id[0].upper() if original_id and original_id[0].isalpha() else 'A'
    letters = string.ascii_uppercase  # A-Z
    
    while base_letter <= 'Z':  # Ensure we don’t go beyond 'Z'
        for num in range(100):  # 00-99
            new_id = f"{base_letter}{num:02d}"  # Ensure it's exactly 3 characters
            if new_id not in existing_groups:
                existing_groups.add(new_id)
                return new_id
        base_letter = chr(ord(base_letter) + 1)  # Move to the next letter

    raise ValueError("Ran out of unique group names!")  # Should never happen


def rename_groups(SACSInputFile, model, rename_df, tolerance=0.01):
    members = model.AllMembers()
    
    # Ensure `rename_df` has a valid index
    rename_df = rename_df.reset_index(drop=True)

    # Convert coordinate columns to numeric
    rename_df[['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']] = rename_df[['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']].apply(pd.to_numeric, errors='coerce')

    # Retrieve all existing group IDs
    existing_groups = {group.Id for group in model.AllMemberGroups()}

    # Dictionary to track renamed groups {original_group_id: new_group_id}
    renamed_groups = {}

    # Store members to rename
    rename_targets = {index: set() for index in rename_df.index}

    # Iterate over members
    for memb in members:
        m_coord = memb_coords(memb)  # Expected as [[x1, y1, z1], [x2, y2, z2]]

        for index, rename in rename_df.iterrows():
            xmin, ymin, zmin, xmax, ymax, zmax = rename['xmin'], rename['ymin'], rename['zmin'], rename['xmax'], rename['ymax'], rename['zmax']

            # Compact match conditions
            x_match = pd.isna(xmin) or pd.isna(xmax) or all((xmin - tolerance if pd.notna(xmin) else float('-inf')) <= x <= (xmax + tolerance if pd.notna(xmax) else float('inf')) for x in [m_coord[0][0], m_coord[1][0]])
            y_match = pd.isna(ymin) or pd.isna(ymax) or all((ymin - tolerance if pd.notna(ymin) else float('-inf')) <= y <= (ymax + tolerance if pd.notna(ymax) else float('inf')) for y in [m_coord[0][1], m_coord[1][1]])
            z_match = pd.isna(zmin) or pd.isna(zmax) or all((zmin - tolerance if pd.notna(zmin) else float('-inf')) <= z <= (zmax + tolerance if pd.notna(zmax) else float('inf')) for z in [m_coord[0][2], m_coord[1][2]])

            if memb.Id == "32A3-33A3":
                    a = 1

            if x_match and y_match and z_match:
                rename_targets[index].add(memb)

    # Process renaming
    for index, members_to_rename in rename_targets.items():
        for memb in members_to_rename:
            original_group_id = memb.group.Id
            old_group = model.FindMemberGroup(original_group_id)
            group_members = old_group.Members

            # 🔹 Check if ALL members of the group fall inside the defined bounds
            all_inside = True
            for group_memb in group_members:
                g_coord = memb_coords(group_memb)  # [[x1, y1, z1], [x2, y2, z2]]

                inside_x = pd.isna(xmin) or pd.isna(xmax) or all((xmin - tolerance if pd.notna(xmin) else float('-inf')) <= x <= (xmax + tolerance if pd.notna(xmax) else float('inf')) for x in [g_coord[0][0], g_coord[1][0]])
                inside_y = pd.isna(ymin) or pd.isna(ymax) or all((ymin - tolerance if pd.notna(ymin) else float('-inf')) <= y <= (ymax + tolerance if pd.notna(ymax) else float('inf')) for y in [g_coord[0][1], g_coord[1][1]])
                inside_z = pd.isna(zmin) or pd.isna(zmax) or all((zmin - tolerance if pd.notna(zmin) else float('-inf')) <= z <= (zmax + tolerance if pd.notna(zmax) else float('inf')) for z in [g_coord[0][2], g_coord[1][2]])
                

                if not (inside_x and inside_y and inside_z):
                    all_inside = False
                    break  # If one member is outside, no need to check further

            # 🔹 If all group members are inside, SKIP renaming
            if all_inside:
                continue

            # Identify if the member is a cone or a tube
            is_tube = memb.group.Segments[0].IsSimpleTube
            if not is_tube:
                shape_ = memb.group.Segments[0].Section.Shape
                is_cone = shape_ == "CON"
            else:
                is_cone = False

            if is_cone or is_tube:
                # Check if this group has already been renamed
                if original_group_id in renamed_groups:
                    new_group_id = renamed_groups[original_group_id]
                    new_group = model.FindMemberGroup(new_group_id)
                else:
                    # Generate a new group ID
                    new_group_id = generate_unique_group(existing_groups, original_group_id)
                    existing_groups.add(new_group_id)
                    # Create a new group instance
                    new_group = model.AddMemberGroup(new_group_id)

                    # Copy material properties
                    material_properties = old_group.Segments[0].Material.Values

                    # Add segment to the group
                    if is_cone:
                        all_sections = {section.Id for section in model.AllMemberSections()}
                        old_section_values = old_group.Segments[0].Section.XSect.Values
                        old_section_id = old_group.Segments[0].Section.Id
                        new_section_id = generate_unique_group(all_sections, old_section_id)  # Generate unique section ID
                        all_sections.add(new_section_id)

                        new_group.AddSegment()
                        new_section_values = old_section_values.copy()

                        print(new_section_values)

                        # Create new section for cone
                        new_section = model.AddMemberSection(id=new_section_id, Shape="CON", XSect=new_section_values)
                        new_group.Segments[0].Section = new_section

                    elif is_tube:
                        # Standard Tube
                        OD = get_geo(memb, "OD")
                        THK = get_geo(memb, "T")
                        new_group.AddSegment(Tube={'OD': OD/10, 'T': THK/10})
                        material_properties["FY"] = get_geo(memb, "FY")

                    # Assign material properties
                    new_group.Segments[0].Material = material_properties

                    # Store new group mapping
                    renamed_groups[original_group_id] = new_group_id

                # Assign the new group to the member
                memb.group = new_group

    return model

def color_by_plan_old(file_path, model, plans_df, tolerance=0.01):
    members = model.AllMembers()
    
    # Dictionary to store groups corresponding to each plan index
    plan_groups = {index: [] for index in plans_df.index}

    # Iterate over members first
    for memb in members:
        m_coord = memb_coords(memb)  # Now expected as [[x1, y1, z1], [x2, y2, z2]]

        # Iterate over plan rows (less frequent loop)
        for index, plan in plans_df.iterrows():
            x_target, y_target, z_target = plan['x'], plan['y'], plan['z']
            
            # Check if the member matches this plan row
            x_match = pd.isna(x_target) or all(abs(x_target - x) <= tolerance for x in [m_coord[0][0], m_coord[1][0]])
            y_match = pd.isna(y_target) or all(abs(y_target - y) <= tolerance for y in [m_coord[0][1], m_coord[1][1]])
            z_match = pd.isna(z_target) or all(abs(z_target - z) <= tolerance for z in [m_coord[0][2], m_coord[1][2]])

            if x_match and y_match and z_match:
                plan_groups[index].append(memb.group.Id)  # Store the group in the corresponding plan index

    # Iterate over plan_groups and apply colors
    for index, group_list in plan_groups.items():
        if group_list:  # Only process non-empty groups
            color_code = plans_df.at[index, 'color']  # Retrieve the corresponding color
            color_by_group_list(file_path, group_list, color_code)  # Apply coloring





        

def color_by_group_list(file_path, group_list, color_code):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Function to create the color line
    def create_color_line(group, color_code):
        return f" **GCOL** {group}M{color_code}"

    # 2nd Last line
    end_index = len(lines) - 1

    # Iterate over the DataFrame and create color lines
    for group in group_list:

        color_line = create_color_line(group, color_code)
        lines.insert(end_index, color_line + "\n")
        end_index += 1  # Adjust the end_index to account for the newly inserted line
    
    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def get_group_list_from_members(member_list, model):
    group_list = []

    for member in member_list:
        memb = model.FindMember(member)
        group_list.append(memb.group.Id)

    group_list = list(set(group_list))

    return group_list

import numpy as np

def material(thk, Sy):
    # Detect whether inputs are scalar
    thk_scalar = np.isscalar(thk)
    Sy_scalar = np.isscalar(Sy)
    # Convert to arrays for vectorized operations
    thk = np.asarray(thk, dtype=float)
    Sy = np.asarray(Sy, dtype=float)
    # S420 table
    S420 = np.where(thk < 16, 420,
           np.where(thk <= 25, 400,
           np.where(thk <= 40, 390,
           np.where(thk <= 63, 380,
           np.where(thk <= 80, 370,
           np.where(thk <= 100, 360, 340))))))
    # S460 table
    S460 = np.where(thk < 16, 460,
           np.where(thk <= 25, 440,
           np.where(thk <= 40, 420,
           np.where(thk <= 63, 415,
           np.where(thk <= 80, 400,
           np.where(thk <= 100, 390, 380))))))
    # Compare which yield strength is closer
    diff_460 = np.abs(Sy - S460)
    diff_420 = np.abs(Sy - S420)

    result = np.where(diff_460 < diff_420, 460, 420)

    # Return scalar if input was scalar
    if thk_scalar and Sy_scalar:
        return result.item()

    return result


import numpy as np

def yield_strength(thk, material):
    # Detect scalar inputs
    thk_scalar = np.isscalar(thk)
    mat_scalar = np.isscalar(material)
    # Convert to arrays for vectorized operations
    thk = np.asarray(thk, dtype=float)
    material = np.asarray(material, dtype=float)
    # Conditions for S420
    cond_420 = (material == 420)
    cond_460 = (material == 460)

    conditions = [
        (thk < 16) & cond_420,
        (thk <= 25) & cond_420,
        (thk <= 40) & cond_420,
        (thk <= 63) & cond_420,
        (thk <= 80) & cond_420,
        (thk <= 100) & cond_420,
        cond_420,  # fallback for S420
        (thk < 16) & cond_460,
        (thk <= 25) & cond_460,
        (thk <= 40) & cond_460,
        (thk <= 63) & cond_460,
        (thk <= 80) & cond_460,
        (thk <= 100) & cond_460,
        cond_460   # fallback for S460
    ]

    choices = [420, 400, 390, 380, 370, 360, 340,
               460, 440, 420, 415, 400, 390, 380]
    result = np.select(conditions, choices, default=np.nan)
    # Return scalar if both inputs were scalar
    if thk_scalar and mat_scalar:
        return result.item()
    return result


def UC_color(file_path, reduced_df):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Define the RGB values for green and red
    green = np.array([0, 255, 0])
    red = np.array([255, 0, 0])
    # Number of colors to generate
    num_colors = 10
    # Interpolate between green and red
    colors = np.linspace(green, red, num_colors).astype(int)
    # Convert to the desired format
    formatted_colors = ["{:3} {:3} {:3}".format(r, g, b) for r, g, b in colors]
    # Color names
    color_names = [
        "bright_green", "green_yellow", "yellow_green", "olive_green",
        "yellow_olive", "golden_yellow", "orange_gold", "orange_red",
        "red_orange", "bright_red"
    ]
    # Create a dictionary mapping UC_max ranges to colors
    uc_max_ranges = np.linspace(0.3, 1.0, num_colors)
    uc_max_colors = {uc_max_ranges[i]: formatted_colors[i] for i in range(num_colors)}
    # Function to get color based on UC_max
    def get_color_by_uc_max(uc_max):
        for key in uc_max_colors.keys():
            if uc_max <= key:
                return uc_max_colors[key]
        return uc_max_colors[max(uc_max_colors.keys())]
    # Function to create the color line
    def create_color_line(member, color_code):
        return f" **GCOL** {member}M{color_code}"
    # 2nd Last line
    end_index = len(lines) - 1
    # Iterate over the DataFrame and create color lines
    for index, row in reduced_df.iterrows():
        members = [row[col] for col in ["chord1", "chord2", "brace1", "brace2", "brace3"] if pd.notna(row[col]) and row[col] != '']
        uc_max = row['UC_Max']
        color_code = get_color_by_uc_max(uc_max)
        for member in members:
            color_line = create_color_line(member, color_code)
            lines.insert(end_index, color_line + "\n")
            end_index += 1  # Adjust the end_index to account for the newly inserted line
 # Construct the new file name
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_UC{ext}"

    # Write the updated lines back to the new file
    with open(new_file_path, 'w') as file:
        file.writelines(lines)
# Helper function to convert values to float where possible
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value
# ************************************>>>>>>>> FLS FUNCTIONS START ******************<<<<<<<<<<<<  



def import_fls_joints(model, joints, z_max):
    tubular_df = pd.DataFrame({})
    inline_df = pd.DataFrame({})
    cone_df = pd.DataFrame({})
    for joint_id, joint in model.joints.items():
        print(joint_id)
        if is_joint(joint, z_max):
            print(joint.Id)
            tubular_joint = get_fls_joint_geometry(joint, True)
            tubular_joint = expand_keys_fls(tubular_joint)
            tubular_df = pd.concat([tubular_df, pd.DataFrame(tubular_joint)], ignore_index=True)
            print(tubular_df)
            tubular_df = add_groups(tubular_df, model)
            
        elif is_inline(joint, z_max):
            inline_joint = get_fls_inline_geometry(joint)
            inline_df = pd.concat([inline_df, pd.DataFrame(inline_joint)], ignore_index=True)
        elif is_cone(joint, z_max):
            cone_joint = get_fls_cone_geometry(joint)
            cone_df = pd.concat([cone_df, pd.DataFrame(cone_joint)], ignore_index=True)
 
    
    # Replace actual NaN values with empty strings
    tubular_df = tubular_df.fillna('')    
    # Replace string representations of NaN ('NA', 'na', 'nan', 'NaN') with empty strings
    tubular_df = tubular_df.replace(['nan', 'NaN', 'NA', 'na'], '', regex=False)


    tubular_df = add_X_braces_FLS(tubular_df)
    tubular_df.to_csv(f'joint_data.csv', index=False)
    inline_df.to_csv(f'inline_data.csv', index=False)
    cone_df.to_csv(f'cone_data.csv', index=False)
    return [tubular_df, inline_df, cone_df]

def is_cone(joint, z_max):
    avoid_list = avoid_groups("avoid_joint_groups.txt")
    z = joint.Z
    n_joints = 0
    ang_tolerance = 5 
    members = joint.AttachedMembers
   
    if z > z_max:
        return False
    else:
        is_any_cone = any(member.is_cone for member in members)
        n_joints = len(members)

    if z<= z_max and n_joints == 2 and is_any_cone and members[0].group.Id not in avoid_list and members[1].group.Id not in avoid_list:
        angle = member_angle(members[0], members[1], False)
        return True if (math.isclose(angle, 0, abs_tol=ang_tolerance) or math.isclose(angle, 180, abs_tol=ang_tolerance)) else False
    else:
        return False

def is_inline(joint, z_max):
    avoid_list = avoid_groups("avoid_joint_groups.txt")
    z = joint.Z
    n_joints = 0
    ang_tolerance = 1 
    members = joint.AttachedMembers

    if z > z_max:
        return False
    else:
        are_tubes = all(member.is_tube for member in members)
        n_joints = len(members)
                
    if z<= z_max and n_joints == 2 and are_tubes and members[0].group.Id not in avoid_list and members[1].group.Id not in avoid_list:
        is_continuous_ = is_continuous(members[0], members[1], joint.Id)
        angle = member_angle(members[0], members[1], False)
        return members if (is_continuous_ and ((math.isclose(angle, 0, abs_tol=ang_tolerance) or math.isclose(angle, 180, abs_tol=ang_tolerance)))) else False
    else:
        return False

def is_continuous(m1, m2, joint_id, dist_tol = 0.01):
    coord1 = get_member_coord_at_joint(m1, joint_id)
    coord2 = get_member_coord_at_joint(m2, joint_id)
    dist = distance.euclidean(coord1, coord2)
    if dist < dist_tol:
        return True
    else:
        return False

def is_straight(joint, skip_exact = [], skip_prefix = []):
    n_joints = 0
    ang_tolerance = 1
    attached_members = joint.AttachedMembers
    
    members = [m for m in attached_members if m.group_id not in skip_exact and not any(m.group_id.startswith(pref) for pref in skip_prefix)]
    are_tubes = all(member.is_tube for member in members)
    n_members = len(members)
           
    if n_members == 2 and are_tubes:
        is_continuous_ = is_continuous(members[0], members[1], joint.Id)
        angle = member_angle(members[0], members[1], False)
        return members if (is_continuous_ and ((math.isclose(angle, 0, abs_tol=ang_tolerance) or math.isclose(angle, 180, abs_tol=ang_tolerance)))) else False
    else:
        return False
    


# RETURNS JOINT, CHORD, BRACES and GAPS instances
def get_fls_joint_geometry(joint, ID_option = False): 
    print(joint.Id)

    avoid_list = avoid_groups("avoid_joint_groups.txt")
    members = get_joint_members(joint, avoid_list)
    # print("MEMBERS:", dict_of_arrays_Ids(members))
    # if joint.Id == "60D1":
    #     a = 1
    chord = members["chords"][0]
    chord_v = outer_vect(joint, chord)
    all_braces = members["braces"]
    checked_list = ""
    brace_list = []
    gap_list = []
    angle_list = []
    plan_list = []
    braces_IPB = []
    for brace1 in all_braces:
        joint_braces = []
        brace_vectors = []
        if (brace1.Id not in checked_list) and (brace1.group.Id not in avoid_list):
            brace1_v = outer_vect(joint, brace1)      
            joint_braces.append(brace1)
            brace_vectors.append(brace1_v)
            checked_list += brace1.Id
            plan = plan_name(chord_v, brace1_v)
            plan_list.append(plan)
            braces_IPB.append(IPBaxis(plan, brace1_v))

            for brace2 in all_braces:
                    if (brace2.Id not in checked_list) and (brace2.group.Id not in avoid_list):
                        brace2_v = outer_vect(joint, brace2)
                        #same_joint = is_same_joint(chord_v, brace1_v, brace2_v) 
                        coplanar = are_coplanar(chord_v, brace1_v, brace2_v) 
                        same_side = are_on_same_side(chord_v, brace1_v, brace2_v) 
                        if (coplanar == True) and (same_side == True):
                            joint_braces.append(brace2)
                            brace_vectors.append(brace2_v)
                            checked_list += brace2.Id
           
            joint_braces = order_braces(chord_v, brace_vectors, joint_braces)
            gap_list.append(calculate_gaps(joint, chord, joint_braces))
            angle_list.append(list(map(lambda x: np.rad2deg(x), get_angles2(joint, chord_v, joint_braces))))
            brace_list.append(joint_braces)
            n_joints = len(brace_list)
  
    brc_ODs = [[get_geo(brace,"OD", joint.Id) for brace in braces] for braces in brace_list]
    brc_THKs = [[get_geo(brace,"THK") for brace in braces] for braces in brace_list]
    brc_SYs = [[get_geo(brace,"FY") for brace in braces] for braces in brace_list]
    braces_IPB = [[IPBaxis(plan_list[i], outer_vect(joint, brace)) for brace in braces] for i,braces in enumerate(brace_list)]

    chd_ODs = [get_geo(chord,"OD", joint.Id) for chord in members["chords"]]
    chd_THKs = [get_geo(chord,"THK") for chord in members["chords"]]
    chd_SYs = [get_geo(chord,"FY") for chord in members["chords"]]
    chd_IPBs = [IPBaxis(plan, outer_vect(joint, chord)) for plan in plan_list]
    
    elevation_list = [[common_joint_coords(joint, brace)[2] for brace in braces] for braces in brace_list]

    if ID_option == False: # return SACS Objects
        data = { "joint": [joint] * n_joints , "type": joint_type2(brace_list), "chords": [members["chords"]] * n_joints, "braces": brace_list, "gaps": gap_list, "angles": angle_list,
                "chd_ODs": [chd_ODs] * n_joints, "chd_THKs": [chd_THKs] * n_joints, "brc_ODs": brc_ODs, "brc_THKs": brc_THKs, "plan": plan_list, "elevations": elevation_list, "chd_IPBs": chd_IPBs, "brc_IPBs": braces_IPB,
                "brc_SYs": brc_SYs, "chd_SYs": [chd_SYs]* n_joints }
    else: # turn all SACS objects to Id strings
        data = { "joint": obj_to_Id(n_joints, [joint]), "type": joint_type2(brace_list),"chords": obj_to_Id3(n_joints, members["chords"]), "braces": obj_to_Id2(brace_list), "gaps": gap_list, "angles": angle_list,
                "chd_ODs": [chd_ODs] * n_joints, "chd_THKs": [chd_THKs] * n_joints, "brc_ODs": brc_ODs, "brc_THKs": brc_THKs, "plan": plan_list, "elevations": elevation_list, "chd_IPBs": chd_IPBs, "brc_IPBs": braces_IPB,
                "brc_SYs": brc_SYs, "chd_SYs": [chd_SYs]* n_joints }
        return data 


def calc_inline_SCF(memb1_OD, memb1_THK, memb2_OD, memb2_THK, EC = False, 𝛿m_actual = False):
    OD_max = max(memb1_OD, memb2_OD)
    T_max = max(memb1_THK, memb2_THK)
    T_min = min(memb1_THK, memb2_THK)
    𝛿m, slope = [6, 4]
    𝛿t= (T_max - T_min)/2
    

    𝛿m = min(0.1 * T_min,𝛿m) if 𝛿m_actual else 𝛿m

    L = 𝛿t * 4 * 2 if T_max != T_min else (T_min-3)*math.tan(30*math.pi/180)*2
    
    if EC == True:
        β = 1.5 
        𝛼 = 0
    else:     
        β = 1.5 - 1/(math.log10(OD_max / T_min)) + 3/((math.log10(OD_max / T_min))**2)
        𝛼 = 1.82 * L / (math.sqrt(OD_max * T_min) * (1 + (T_max / T_min)**β)) if T_max != T_min else 0.91 * L / math.sqrt(OD_max * T_min)
    
    Root_Cap_Ratio = 4 / (T_min * math.tan(45/2*math.pi/180)*2)
    if T_max != T_min:
        SCFtoe = 1 + (6 * (𝛿m+𝛿t)/T_min) * (1/(1 + (T_max / T_min)**β)) * math.e**(-𝛼)
    else:
        SCFtoe = 1 + 3*(𝛿m/T_min) * math.e**(-𝛼) 
    SCFroot = 1 + (SCFtoe - 1) * Root_Cap_Ratio
    return [SCFtoe, SCFroot]


def get_fls_inline_geometry(joint):  
    member1, member2 = joint.AttachedMembers
    memb1_OD = get_geo(member1,"OD")
    memb1_THK = get_geo(member1,"THK")
    memb2_OD = get_geo(member2,"OD")
    memb2_THK = get_geo(member2,"THK")
    SY_1 = get_geo(member1,"FY")
    SY_2 = get_geo(member2,"FY")

    elevation = joint.Z
    SCFtoe, SCFroot = calc_inline_SCF(memb1_OD, memb1_THK, memb2_OD, memb2_THK, EC = False)
    
    data = {"joint": [joint.Id], "memb1": [member1.Id], "memb2": [member2.Id], "memb1_OD": [memb1_OD], "memb1_THK": [memb1_THK], "memb2_OD": [memb2_OD], "memb2_THK": [memb2_THK], "memb1_group": [member1.group.Id], "memb2_group": [member2.group.Id], "memb1_SY": SY_1, "memb2_SY": SY_2, "Elevation": elevation, "SCFtoe": SCFtoe, "SCFroot": SCFroot}
    # print(data)
    return data


def calc_cone_SCF(side, cone_ODL, cone_ODS, cone_THK, cone_length, tub_OD, tub_THK):
    cone_OD = cone_ODL if side == "L" else cone_ODS
    tan_alfa = (cone_ODL - cone_ODS) / (2 * cone_length)
    SCFtubular = 1 + 0.6 * tub_THK * math.sqrt(tub_OD * (tub_THK + cone_THK)) / ((tub_THK ** 2) ) * tan_alfa
    SCFcone = 1 + 0.6 * tub_THK * math.sqrt(cone_OD * (tub_THK + cone_THK)) / ((cone_THK ** 2) ) * tan_alfa

    return [SCFcone, SCFtubular]

def get_fls_cone_geometry(joint):
    member1, member2 = joint.AttachedMembers
    cone = member1 if member1.is_cone else member2
    tub = member2 if member2.is_tube else member1
    cone_section = cone.group.section
    cone_ODL = cone_section.OD_L * 10
    cone_ODS =cone_section.OD_S * 10
    cone_THK = cone_section.THK * 10
    cone_SY = cone.group.FY
    cone_length = cone.length * 1000
    tub_OD = get_geo(tub,"OD")
    tub_THK = get_geo(tub,"THK")
    tub_SY = get_geo(tub,"FY")
    elevation = joint.Z
 
    side = "L" if joint.Id == cone.joint1.Id else "S"

    SCFcone, SCFtubular = calc_cone_SCF(side, cone_ODL, cone_ODS, cone_THK, cone_length, tub_OD, tub_THK)

    data = {"joint": [joint.Id], "cone": [cone.Id], "tubular": [tub.Id], "cone_L": [cone_ODL], "cone_S": [cone_ODS], "cone_THK": [cone_THK], "tub_OD": [tub_OD], "tub_THK": [tub_THK], "length": [cone_length], "cone_group": [cone.group.Id], "tub_group": [tub.group.Id], "cone_SY": cone_SY, "tub_SY": tub_SY, "Elevation": elevation ,  "SCFtubular": [SCFtubular], "SCFcone": [SCFcone] }
    return data

def expand_keys_fls(jointprop):    
    new_dict = jointprop
    group_keys = [["chords", "chd_ODs", "chd_THKs" ],["braces","brc_ODs", "brc_THKs", "brc_IPBs", "angles", "gaps", "elevations", "chd_SYs", "brc_SYs"]]
    #group_keys = [["chords", "chd_ODs", "chd_THKs"  ],["braces","brc_ODs", "brc_THKs", "angles", "gaps"]]
    r = 1
    for target_keys in group_keys:
        r += 1
        for target_key in target_keys:
            original_arr = jointprop[target_key]
            new_arr = []
            for i in range(r):
                new_arr = []
                new_key = f'{target_key[:-1]}' + f'{i+1}'
                for j in range(len(original_arr)):
                    if len(original_arr[j]) >= i+1:
                        new_arr.append(original_arr[j][i])
                    else:
                        new_arr.append("NA")
                new_dict[new_key] = new_arr
            new_dict.pop(target_key)
    return new_dict

def add_X_braces_FLS(df):
    # Step 1: Group by 'joint'
    for i in range(1, 4):
        df[f"x-brace{i}"] = ""
        df[f"x-angle{i}"] = ""
    
    grouped = df.groupby('joint')

    # Step 2: Process each joint group
    for joint, group in grouped:
        # Step 2.1: Group by 'type' and 'plan' to find pairs
        print(f'X-Brace Testing {joint}')
        type_plan_groups = group.groupby(['type', 'plan'])

        for (type, plan), type_plan_group in type_plan_groups:
            # print(type, plan, loadcase, len(type_plan_group))
            if len(type_plan_group) >= 2:
                # Step 3: Combine the two rows, making sure to copy them
                index1, index2 = type_plan_group.index[0], type_plan_group.index[1]
                row1, row2 = type_plan_group.iloc[0], type_plan_group.iloc[1]

                if row1["type"] == "T" and row2["type"] == "T":
                    df.at[index1, "type"] = "T-X"
                    df.at[index2, "type"] = "T-X"
                else:
                    df.at[index1, "type"] = f"{row1['type']}-X"
                    df.at[index2, "type"] = f"{row1['type']}-X"
                # Add Associated X-Brace Columns 
                for i in range(1, 4):
                    angle1 = row1[f"angle{i}"]
                    if angle1:
                        target_angle1 = 360 - (angle1 + 180) 
                        # Calculate the absolute differences from target angles
                        # Calculate the absolute differences from target angles
                        delta_angles = [abs(target_angle1 - float(row2[f"angle{j}"] or 1000)) for j in range(1, 4)]
                        # Find the index with the smallest difference
                        x_index = delta_angles.index(min(delta_angles)) + 1

                        print(df.at[index1, f"brace{i}"], df.at[index2, f"brace{x_index}"])

                        # Update the x-brace and x-angle fields
                        df.at[index1, f"x-brace{i}"] = df.at[index2, f"brace{x_index}"]
                        df.at[index2, f"x-brace{x_index}"] = df.at[index1, f"brace{i}"]
                        df.at[index1, f"x-angle{i}"] = df.at[index2, f"angle{x_index}"]
                        df.at[index2, f"x-angle{x_index}"] = df.at[index1, f"angle{i}"]
    return df

def effective_KT_joints(model, joint_df):
    # Initialize a list to hold the effective KT values
    joint_df['eff_KT'] = False

    # Iterate through each joint in the DataFrame
    for index, row in joint_df.iterrows():
        central_joint_id = row['joint']
        central_joint = model.joints[central_joint_id]
        chord1_id = row['chord1']
        chord2_id = row['chord2']
        brace1_id = row['brace1']
        brace1 = model.members[brace1_id]
        chord1 = model.members[chord1_id]
        chord2 = model.members[chord2_id]
        chord_OD = row['chd_OD1'] / 1000
        brace_OD = row['brc_OD1'] / 1000
        allowed_types = {"Y", "Y-X", "T", "T-X"}
        central_plan = row['plan']  # adjust column name if needed

        left_joint_id = other_item(chord1.Id, central_joint_id)
        right_joint_id = other_item(chord2.Id, central_joint_id)

        if central_joint_id == "GA79":
            a = 1
        # Check if left_joint_id and right_joint_id are present in joint_df['joint'].values:
        if left_joint_id in joint_df['joint'].values and right_joint_id in joint_df['joint'].values:
            left_rows = joint_df.loc[joint_df['joint'] == left_joint_id]
            right_rows = joint_df.loc[joint_df['joint'] == right_joint_id]
            # has_brace1 = any((nbr_idx != index) and (nbr['type'] in allowed_types) and (nbr['plan'] == central_plan and are_on_same_side2(model.joints[nbr["joint"]], chord1, brace1, model.members[nbr["brace1"]])) for nbr_idx, nbr in left_rows.iterrows())
            # has_brace2 = any((nbr_idx != index) and (nbr['type'] in allowed_types) and (nbr['plan'] == central_plan and are_on_same_side2(model.joints[nbr["joint"]], chord1, brace1, model.members[nbr["brace1"]])) for nbr_idx, nbr in right_rows.iterrows())
            has_brace1 = False
            has_brace2 = False
            for nbr_idx, nbr in left_rows.iterrows():
                not_self = nbr_idx != index
                type_ok = nbr['type'] in allowed_types
                plan_ok = nbr['plan'] == central_plan
                same_side_ok = are_on_same_side2(central_joint, model.joints[nbr["joint"]], chord1, brace1,model.members[nbr["brace1"]]                )

                if not_self and type_ok and plan_ok and same_side_ok:
                    has_brace1 = True

            for nbr_idx, nbr in right_rows.iterrows():
                not_self = nbr_idx != index
                type_ok = nbr['type'] in allowed_types
                plan_ok = nbr['plan'] == central_plan
                same_side_ok = are_on_same_side2(central_joint, model.joints[nbr["joint"]], chord2, brace1, model.members[nbr["brace1"]]               )

                if not_self and type_ok and plan_ok and same_side_ok:
                    has_brace2 = True

            is_KT_ = has_brace1 and has_brace2
            if is_KT_ and ((chord1.length / brace_OD) < 1.2)  and (chord1.length < chord_OD) and (chord2.length < chord_OD):
                joint_df.at[index, 'eff_KT'] = True

def are_on_same_side2(central_joint, joint, chord, brace1, brace2):
    chord_v = outer_vect(joint, chord)
    brace1_v = outer_vect(central_joint, brace1)
    brace2_v = outer_vect(joint, brace2)
    # Calculate cross product of chord with each brace
    crossProduct1 = np.cross(chord_v, brace1_v)
    crossProduct2 = np.cross(chord_v, brace2_v)
    # Check if the dot product of the cross products is positive
    internal_product = np.dot(crossProduct1, crossProduct2)
    are_on_same_side2 = (internal_product > 0)
    return are_on_same_side2

def get_member_between(model, joint_a_id, joint_b_id):
    # Tries both possible member name combinations
    
    matching_keys = [key for key in model.members if joint_a_id in key]
    # print(f"🔍 Members involving joint '{joint_a_id}':")
    # for key in matching_keys:
    #     print(f"{key}")
    #     a = 1
    for name in (joint_a_id + "-" + joint_b_id, joint_b_id + "-" + joint_a_id):
        if name in model.members:
            return model.members[name]
    return None

def get_OD_between(model, joint_a_id, joint_b_id):
    member = get_member_between(model, joint_a_id, joint_b_id)
    if member is None:
        return None  # Can't find the member
    if member.joint1.Id == joint_a_id:
        return get_geo(member, "OD1")
    else:
        return get_geo(member, "OD2")

def get_unconstrained_list(model, joint_ids, central_id, constraint_ids):
    idx = joint_ids.index(central_id)
    central_joint = model.joints[central_id]

    # --- LEFT SEARCH ---
    left_ = joint_ids[0]
    for i in reversed(range(0, idx)):
        j = joint_ids[i]
        if j in constraint_ids:
            chdbrcs = get_joint_members(model.joints[j], [])
            braces = chdbrcs['braces']
            chords = chdbrcs.get('chords', braces[:1] if braces else [])
            # Get member between j and the next one (towards central)
            if i + 1 < len(joint_ids):
                j_next = joint_ids[i + 1]
                member = get_member_between(model, j, j_next)
                OD = get_geo(member, "OD") / 1000 if member else None
                group = member.group.Id
                is_leg = (left(group, 1) == "L")
                leg_junction = bool(chords) and (left(chords[0].group.Id, 1) == "L" and not is_leg)
            else:
                OD = None
            dist = distance.euclidean(model.joints[j].coord, central_joint.coord)
            if OD is None or (leg_junction) or (dist > OD):
                left_ = j
                break

    # --- RIGHT SEARCH ---
    right_ = joint_ids[-1]
    for i in range(idx + 1, len(joint_ids)):
        j = joint_ids[i]
        if j in constraint_ids:
            chdbrcs = get_joint_members(model.joints[j], [])
            braces = chdbrcs['braces']
            chords = chdbrcs.get('chords', braces[:1] if braces else [])
            if i - 1 >= 0:
                j_prev = joint_ids[i - 1]
                member = get_member_between(model, j_prev, j)
                OD = get_geo(member, "OD") / 1000 if member else None
                group = member.group.Id
                is_leg = (left(group, 1) == "L")
                leg_junction = bool(chords) and (left(chords[0].group.Id, 1) == "L" and not is_leg)
            else:
                OD = None
            dist = distance.euclidean(model.joints[j].coord, central_joint.coord)
            if OD is None or (leg_junction) or (dist > OD):
                right_ = j
                break

    return left_, central_id, right_

def get_effective_BCs(model, central_member, joint_ids, constraint_ids):
    j1 = central_member.joint1.Id
    j2 = central_member.joint2.Id

    # ---------------------------------------------------------
    # 1. EARLY EXIT: both ends are constraints → no iteration
    # ---------------------------------------------------------
    if j1 in constraint_ids and j2 in constraint_ids:
        return j1, j2

    # ---------------------------------------------------------
    # 2. Determine initial BCs
    # ---------------------------------------------------------
    left_BC  = j1 if j1 in constraint_ids else None
    right_BC = j2 if j2 in constraint_ids else None

    idx1 = joint_ids.index(j1)
    idx2 = joint_ids.index(j2)

    central_joint1 = model.joints[j1]
    central_joint2 = model.joints[j2]

    # ---------------------------------------------------------
    # 3. LEFT SEARCH (only if j1 is NOT a constraint)
    # ---------------------------------------------------------
    if left_BC is None:
        left_BC = joint_ids[0]  # fallback

        for i in reversed(range(0, idx1)):
            j = joint_ids[i]

            if j in constraint_ids:
                chdbrcs = get_joint_members(model.joints[j], [])
                braces = chdbrcs['braces']
                chords = chdbrcs.get('chords', braces[:1] if braces else [])

                j_next = joint_ids[i + 1]
                member = get_member_between(model, j, j_next)

                if member:
                    OD = get_geo(member, "OD") / 1000
                    gid = member.group.Id
                    is_leg = gid.startswith("L")
                    leg_junction = bool(chords) and chords[0].group.Id.startswith("L") and not is_leg
                else:
                    OD = None
                    leg_junction = False

                dist = distance.euclidean(model.joints[j].coord, central_joint1.coord)

                if OD is None or leg_junction or (dist > OD):
                    left_BC = j
                    break

    # ---------------------------------------------------------
    # 4. RIGHT SEARCH (only if j2 is NOT a constraint)
    # ---------------------------------------------------------
    if right_BC is None:
        right_BC = joint_ids[-1]  # fallback

        for i in range(idx2 + 1, len(joint_ids)):
            j = joint_ids[i]

            if j in constraint_ids:
                chdbrcs = get_joint_members(model.joints[j], [])
                braces = chdbrcs['braces']
                chords = chdbrcs.get('chords', braces[:1] if braces else [])

                j_prev = joint_ids[i - 1]
                member = get_member_between(model, j_prev, j)

                if member:
                    OD = get_geo(member, "OD") / 1000
                    gid = member.group.Id
                    is_leg = gid.startswith("L")
                    leg_junction = bool(chords) and chords[0].group.Id.startswith("L") and not is_leg
                else:
                    OD = None
                    leg_junction = False

                dist = distance.euclidean(model.joints[j].coord, central_joint2.coord)

                if OD is None or leg_junction or (dist > OD):
                    right_BC = j
                    break

    # ---------------------------------------------------------
    # 5. Return ONLY the two boundary joints
    # ---------------------------------------------------------
    return left_BC, right_BC



def effective_chord_length(row, model, z_max):
    braces_to_avoid = avoid_groups("avoid_brace_groups.txt")
    member = model.members[row["chord1"]]
    is_right_direction  = True
    central_member = member
    central_joint = row["joint"]
    joint_type = row["type"]
    eff_KT = row["eff_KT"]
    current_joint = ""
    OD = get_geo(member, "OD")
    # Initialize lists to hold member and joint objects for clarity and explicit handling
    member_list = [central_member]  # Start with central member included
    joint_list = []
    angle_tolerance = 1
    # Determine the direction flags for iteration
    direction_flags = [True, False]  # True for right, False for left

    print(central_joint)
    # CYCLE THROUGH COLINEAR MEMBERS TO THE RIGHT AND THEN TO THE LEFT
    for is_right_direction in direction_flags:
        current_member = central_member
        current_joint_id = right(central_member.Id, 4) if is_right_direction else left(central_member.Id, 4)
        current_joint = model.joints[current_joint_id]
        joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)

        angle_ = 0  # Reset angle for each direction
        while angle_ < angle_tolerance:
            found_colinear = False
            skip_member = False
            n_members = len(current_joint.AttachedMembers) 
            if n_members > 1 and current_joint.Z <= z_max:
                for member_i in current_joint.AttachedMembers:
                    if member_i.Id != current_member.Id:
                    # if member_i.Id != current_member.Id and (member_i.group.Id not in braces_to_avoid):
                        if left(member_i.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                            angle_ = 10  # Reset angle to exit loop
                            break
                        angle_ = math.degrees(member_angle_min(current_member, member_i))
                        if angle_ < angle_tolerance:
                            found_colinear = True
                            next_member = member_i
                            next_joint_id = other_item(next_member.Id, current_joint.Id)
                            next_joint = model.joints[next_joint_id]
                            break
                if found_colinear and not skip_member:
                    current_member = next_member
                    current_joint = next_joint
                    member_list.append(current_member) if is_right_direction else member_list.insert(0, current_member)
                    joint_list.append(current_joint) if is_right_direction else joint_list.insert(0, current_joint)
                else:
                        # End the current direction iteration and break to switch direction or end loop
                    break
            else:
                    # If no valid joint or member found, force exit from the loop
                break

    chord_plan = get_plan(joint_list[0].coord, joint_list[-1].coord)
    # print(chord_plan)
    if left(member.group.Id, 1) == "L" and ("_X_" not in chord_plan) and ("_Y_" not in chord_plan):
        is_leg = True
        target_ratio = 0.35
    else:
        is_leg = False
        target_ratio = 0.4

    joint_ids = [joint.Id for joint in joint_list]
    if "GK28" in joint_ids:
         print(joint_ids) 
         a = 1   

    My_constraints = []
    Mz_constraints = []
    #' Find hard point constraints only mid joints
    for joint in joint_list[1:-1]: 
        # print(joint.Id)
        if joint.Id == "GH29":
            a = 1
        if len(joint.AttachedMembers) > 2:
            chdbrcs = get_joint_members(joint, braces_to_avoid)
            braces = chdbrcs['braces']
            chords = chdbrcs.get('chords', braces[:1] if braces else [])
            if chords and (joint.Id != central_joint):
                group, CHD_OD = chords[0].group.Id, get_geo(chords[0], "OD", joint.Id)
                n_xy_braces, is_plan_elevation = 0, False
                if is_leg:
                    for brace in braces:
                        brace_plan = get_plan(brace.coord1, brace.coord2)
                        BRC_OD = get_geo(brace, "OD", joint.Id)
                        BRC_to_CHD = BRC_OD / CHD_OD if is_numerical(BRC_OD) and is_numerical(CHD_OD) else 0

                        if ("XY" in brace_plan) and (BRC_to_CHD > target_ratio) and (left(brace.group.Id, 1) != "P") and is_leg:
                            n_xy_braces += 1
                    if n_xy_braces >= 2:
                        is_plan_elevation = True
                # elif (not is_leg) and left(group, 1) == "L":


                # for brace in braces + chords:
                for brace in (braces + chords):
                    brc_plan = get_plan(brace.coord1, brace.coord2)
                    # brc_chd_plan = common_plan(brc_plan, memb_plan)
                    BRC_OD = get_geo(brace, "OD", joint.Id)
                    BRC_to_CHD = BRC_OD / CHD_OD if is_numerical(BRC_OD) and is_numerical(CHD_OD) else 0
                    if (group not in braces_to_avoid):
                        # 'LEG CONNECTIONS
                        if (is_leg and is_plan_elevation or ((not is_leg) and left(group, 1) == "L")):
                        # if LEG in you encounter a plan elevation or if PLAN ELVATION and you enconter a leg
                            # print(joint.Id)
                            My_constraints.append(joint.Id)
                            Mz_constraints.append(joint.Id)
                        elif (BRC_to_CHD > target_ratio) and (not is_leg): 
                            IP_proj_angle = projection_angle(brace, member, get_planes(chord_plan, 0))
                            OP1_proj_angle = projection_angle(brace, member, get_planes(chord_plan, 1))
                            OP2_proj_angle = projection_angle(brace, member, get_planes(chord_plan, 2))
                            # NOTE: The bellow code relfects the default SACS local CSYS orientation! 
                            if right(chord_plan, 2) == "XY": 
                                if (IP_proj_angle > 30) and (joint.Id not in Mz_constraints):
                                    Mz_constraints.append(joint.Id)
                                if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in My_constraints):
                                    My_constraints.append(joint.Id)
                            else:
                                if (IP_proj_angle > 30) and (joint.Id not in My_constraints):
                                    My_constraints.append(joint.Id)
                                if (OP1_proj_angle > 30 or OP2_proj_angle > 30) and (joint.Id not in Mz_constraints):
                                    Mz_constraints.append(joint.Id)
        # ******************* CALCULATE ACTUAL LENGTHS BETWEEN CONSTRAINTS *************************

    My_constraints_ = get_unconstrained_list(model, joint_ids, central_joint, My_constraints)
    Mz_constraints_ = get_unconstrained_list(model, joint_ids, central_joint, Mz_constraints)
    central_joint_coords = model.joints[central_joint].coord
    # Left Side
    canti_f_y_L = cantilever_factor_joint(model, My_constraints_[0]) 
    canti_f_z_L = cantilever_factor_joint(model, Mz_constraints_[0]) 
    # Handle effective length in Z-direction
    left_joint_coord_y = model.joints[My_constraints_[0]].coord
    left_joint_coord_z = model.joints[Mz_constraints_[0]].coord
    
    Ly_L = distance.euclidean(left_joint_coord_y, central_joint_coords)  * canti_f_y_L
    Lz_L = distance.euclidean(left_joint_coord_z, central_joint_coords)  * canti_f_z_L
    # Right Side
    canti_f_y_R = cantilever_factor_joint(model, My_constraints_[-1]) 
    canti_f_z_R = cantilever_factor_joint(model, Mz_constraints_[-1]) 
    # Handle effective length in Z-direction
    right_joint_coord_y = model.joints[My_constraints_[-1]].coord
    right_joint_coord_z = model.joints[Mz_constraints_[-1]].coord
    Ly_R = distance.euclidean(right_joint_coord_y, central_joint_coords)  * canti_f_y_R
    Lz_R = distance.euclidean(right_joint_coord_z, central_joint_coords)  * canti_f_z_R
    
    Ly = Ly_L + Ly_R
    Lz = Lz_L + Lz_R

    L = max(Ly, Lz)

    if L == Ly:
        jointend1 = My_constraints_[0]
        jointend2 = My_constraints_[-1]
    else:
        jointend1 = My_constraints_[0]
        jointend2 = My_constraints_[-1]     

    
    if central_joint == "GK38":
         print(joint_ids)
         a = 1

    is_KT = ("KT" in joint_type or eff_KT)
    eff_chord = L / 8 if is_KT else L * 0.7
    
    # eff_chord = L / 8 if (is_leg or joint_type == "KT" or eff_KT) else L * 0.7
    eff_chord = max(OD * 2/1000, eff_chord)
    return pd.Series({"eff_chord": eff_chord,"jointend1": jointend1,"jointend2": jointend2, "is_KT": is_KT})



def group_force_files():
    angle_re = re.compile(r'-([0-9]+\.[0-9]+)(?:_[0-9]+)?-rpt\.lst$', re.IGNORECASE)
    groups = defaultdict(list)

    for f in glob.glob("**/*-rpt.lst", recursive=True):
        filename = os.path.basename(f)
        m = angle_re.search(filename)
        if m:
            angle = m.group(1)
            groups[angle].append(f)

    return groups




def parse_single_report_file(SACSReportFile):
    member_pattern = r'^[A-Z0-9]{4}-[A-Z0-9]{4}$'
    joint_pattern = r'^[A-Z0-9]{4}'
    load_pattern   = r'^([0-9A-Za-z]{4}|\d{1,4})$'

    paths, members, loads = [], [], []
    forcesA, forcesB = [], []
    member_load_keys_A, member_load_keys_B = [], []

    j = joint = ""
    m = member = ""
    l = load = ""

    with open(SACSReportFile, 'r') as file:
        for i, line in enumerate(file):
            if len(line) > 8:  m = line[0:9].strip()
            if len(line) > 15: j = line[11:15].strip()
            if len(line) > 27: l = line[23:27].strip()

            if re.fullmatch(member_pattern, m) and re.fullmatch(joint_pattern, j):
                member = m
                joint = j
                load_counter = 0

            if re.fullmatch(joint_pattern, j):
                joint = j
                load_counter = 0

            if re.fullmatch(load_pattern, l):
                load = l
                line_arr = line.split()

                if checkIfForceRow(line_arr):
                    load_counter += 1
                    key = f"{member}-{load}"

                    if start_end(joint, member):
                        forcesA.append([float(val) for val in line_arr[-6:]])
                        members.append(member)
                        loads.append(int(load_counter))
                        paths.append(SACSReportFile)
                        member_load_keys_A.append(key)
                    else:
                        forcesB.append([float(val) for val in line_arr[-6:]])
                        member_load_keys_B.append(key)

    # mismatch check
    if set(member_load_keys_A) != set(member_load_keys_B):
        raise ValueError(f"Mismatch in file {SACSReportFile}")

    df1 = pd.DataFrame({"member": members, "loads": loads, "paths": paths})
    df2 = pd.DataFrame(forcesA, columns=["FXA","FYA","FZA","MXA","MYA","MZA"])
    df3 = pd.DataFrame(forcesB, columns=["FXB","FYB","FZB","MXB","MYB","MZB"])

    return pd.concat([df1, df2, df3], axis=1)

def read_report_FLS_forces(suffix):

    files_by_direction = group_force_files()

    for direction, file_list in files_by_direction.items():
        print(f"PROCESSING DIRECTION {direction}° with {len(file_list)} files")

        force_file = f"forces_df_{direction}.pkl.gz"
        if os.path.exists(force_file):
            continue

        dfs = []
        last_LC = 0   # keeps track of the last assigned LC number

        for rpt in file_list:
            print(f"  → Reading {os.path.basename(rpt)}")

            # Parse forces
            df = parse_single_report_file(rpt)
            df["direction"] = direction

            # Extract index from filename
            filename = os.path.basename(rpt)
            m = re.search(r'-([0-9]+\.[0-9]+)(?:_([0-9]+))?-rpt\.lst$', filename)
            if m:
                direction = m.group(1)
                index = int(m.group(2)) if m.group(2) else 0
            else:
                raise ValueError(f"Filename does not match expected pattern: {filename}")


            df["index"] = index

            # Read wave properties for THIS run only
            df = read_wave_prop(df, direction, index)

            # Convert loads to int
            df["loads"] = df["loads"].astype(int)

            # Assign global LC numbers WITHOUT sorting
            df["loads"] = df["loads"] + last_LC

            # Update last_LC for next run
            last_LC = df["loads"].max()

            dfs.append(df)


        # Now combine all enriched runs
        forces_df = pd.concat(dfs, ignore_index=True)

        # Final ordering
        forces_df = forces_df.sort_values(["member", "loads"]).reset_index(drop=True)


        forces_df.to_pickle(force_file)

    return




def read_wave_prop(forces_df, direction, index):
    # Normalize direction formatting (e.g. "45.0")
    direction = f"{float(direction):.1f}"

    # Build expected wave filename pattern
    if index == 0:
        # Example: "*-45.0-wvr.lst"
        suffix = f"-{direction}-wvr.lst"
    else:
        # Example: "*-45.0_2-wvr.lst"
        suffix = f"-{direction}_{index}-wvr.lst"

    # Find matching wave file
    matching_files = glob.glob(f"**/*{suffix}", recursive=True)


    if len(matching_files) == 0:
        raise FileNotFoundError(f"No wave file found for direction={direction}, index={index}, suffix={suffix}")

    if len(matching_files) > 1:
        raise ValueError(f"Multiple wave files found for direction={direction}, index={index}: {matching_files}")

    wave_file = matching_files[0]
    print(f"READING WAVE FILE: {os.path.basename(wave_file)}  (dir={direction}, index={index})")

    # Read file
    with open(wave_file, "r") as f:
        contents = f.readlines()

    # Find n_steps
    n_steps = None
    for line in contents:
        if "NO. STEPS" in line:
            m = re.search(r'NO\. STEPS\s+\*+\s+([\d\.]+)\s', line)
            if m:
                n_steps = int(m.group(1))
                break

    if n_steps is None:
        raise ValueError(f"n_steps not found in wave file {wave_file}")

    # Ensure loads are strings
    forces_df["loads"] = forces_df["loads"].astype(str)

    processed_loadcases = set()

    # Loop through wave cases
    for i, line in enumerate(contents):
        if "* WAVE CASE" in line:
            m = re.search(r'\* WAVE CASE\s+(\d+)\s+\*', line)
            if not m:
                continue

            loadcase = m.group(1)

            if loadcase in processed_loadcases:
                continue

            # Next line contains wave height and period
            next_line = contents[i + 1]
            m2 = re.search(
                r'WAVE HEIGHT\s+=\s+([\d\.]+)\s+M\s+WAVE PERIOD\s+=\s+([\d\.]+)\s+SECS',
                next_line
            )
            if not m2:
                continue

            wave_height = float(m2.group(1))
            wave_period = float(m2.group(2))

            print(f'Found Wave Load Case: {loadcase}, wave height: {wave_height} M, wave period: {wave_period} SECS')

            # Determine load range for this wave case
            start_load = (int(loadcase) - 1) * n_steps + 1
            end_load   = int(loadcase) * n_steps

            # Zero-pad loads for lexicographic comparison
            pad = len(str(end_load))
            forces_df["loads"] = forces_df["loads"].str.zfill(pad)

            start_load = str(start_load).zfill(pad)
            end_load   = str(end_load).zfill(pad)

            # Assign wave properties
            mask = (forces_df["loads"] >= start_load) & (forces_df["loads"] <= end_load)
            forces_df.loc[mask, "wave_height"] = wave_height
            forces_df.loc[mask, "wave_period"] = wave_period

            # print(f"Direction: {direction}",forces_df)

            processed_loadcases.add(loadcase)

    return forces_df








def read_wave_prop_old(forces_df):
    direction = str(forces_df.iloc[0]["direction"])
    # Remove trailing zeros and the decimal point if the number is an integer
    # if '.' in direction:
        # direction = direction.rstrip('0').rstrip('.')
    
    direction = f"{float(direction):.1f}"
    
    # Create a new suffix that includes the direction
    suffix = f"-{direction}-wvr.lst"
    
    # Use glob to find the file with the new suffix in the current directory and subdirectories
    matching_files = glob.glob(f"**/*{suffix}", recursive=True)
    
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file found with suffix {suffix}")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files found with suffix {suffix}: {matching_files}")
    
    # There should be only one matching file
    file_path = matching_files[0]
    filename = os.path.basename(file_path)
    print(f'READING DIRECTION: {direction} degrees, FILE NAME: {filename}')
    
    # Read the file contents
    with open(file_path, 'r') as f:
        contents = f.readlines()
    
    # Initialize variable for n_steps
    n_steps = None
    
    # Find the n_steps value, which should be constant for all wave load cases
    for line in contents:
        if "NO. STEPS" in line:
            n_steps_match = re.search(r'NO\. STEPS\s+\*+\s+([\d\.]+)\s', line)
            if n_steps_match:
                n_steps = int(n_steps_match.group(1))
                print(f'Found n_steps: {n_steps}')
                break
    
    if n_steps is None:
        raise ValueError("n_steps not found in the file")
    
    # Initialize a set to keep track of processed load cases
    processed_loadcases = set()
    
    # Ensure the loads column contains only string values
    forces_df["loads"] = forces_df["loads"].astype(str)
    
    # Check for NaNs or empty cells in the loads column
    if forces_df["loads"].isnull().any() or forces_df["loads"].eq('').any():
        nan_rows = forces_df[forces_df["loads"].isnull() | forces_df["loads"].eq('')]
        raise ValueError(f"NaN or empty values found in 'loads' column at the following rows:\n{nan_rows}")
    
    # Search for load cases and corresponding wave heights and periods until the end of the file
    for i, line in enumerate(contents):
        if "* WAVE CASE" in line:
            loadcase_match = re.search(r'\* WAVE CASE\s+(\d+)\s+\*', line)
            if loadcase_match:
                loadcase = loadcase_match.group(1)
                
                # Skip this load case if it has already been processed
                if loadcase in processed_loadcases:
                    continue
                
                print(f'Found load case number: {loadcase}')
                
                # The next line should contain wave height and wave period
                next_line = contents[i + 1]
                wave_info_match = re.search(r'WAVE HEIGHT\s+=\s+([\d\.]+)\s+M\s+WAVE PERIOD\s+=\s+([\d\.]+)\s+SECS', next_line)
                if wave_info_match:
                    wave_height = wave_info_match.group(1)
                    wave_period = wave_info_match.group(2)
                    print(f'Found wave height: {wave_height} M, wave period: {wave_period} SECS')
                    
                    # Update the DataFrame for the corresponding range of load cases
                    start_loadcase = str((int(loadcase) - 1) * n_steps + 1)
                    end_loadcase = str(int(loadcase) * n_steps)
                    
                    # Ensure lexicographic comparison works correctly for zero-padded numbers
                    forces_df['loads'] = forces_df['loads'].apply(lambda x: x.zfill(len(end_loadcase)))
                    start_loadcase = start_loadcase.zfill(len(end_loadcase))
                    end_loadcase = end_loadcase.zfill(len(end_loadcase))
                    
                    forces_df.loc[(forces_df["loads"] >= start_loadcase) & (forces_df["loads"] <= end_loadcase), "wave_height"] = wave_height
                    forces_df.loc[(forces_df["loads"] >= start_loadcase) & (forces_df["loads"] <= end_loadcase), "wave_period"] = wave_period
                    print(f"Direction: {direction}",forces_df)
                    # Add this load case to the set of processed load cases
                    processed_loadcases.add(loadcase)
                    
    return forces_df

def read_wave_prop_SACS25(forces_df):
    direction = str(forces_df.iloc[0]["direction"])
    # Remove trailing zeros and the decimal point if the number is an integer
    # if '.' in direction:
        # direction = direction.rstrip('0').rstrip('.')
    
    direction = f"{float(direction):.1f}"
    
    # Create a new suffix that includes the direction
    suffix = f"-{direction}-wvr.lst"
    
    # Use glob to find the file with the new suffix in the current directory and subdirectories
    matching_files = glob.glob(f"**/*{suffix}", recursive=True)
    
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file found with suffix {suffix}")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files found with suffix {suffix}: {matching_files}")
    
    # There should be only one matching file
    file_path = matching_files[0]
    filename = os.path.basename(file_path)
    print(f'READING DIRECTION: {direction} degrees, FILE NAME: {filename}')
    
    # Read the file contents
    with open(file_path, 'r') as f:
        contents = f.readlines()
    
    # Initialize variable for n_steps
    n_steps = None
    
    # Find the n_steps value, which should be constant for all wave load cases
    for line in contents:
        if "NO. STEPS" in line:
            n_steps_match = re.search(r'NO\. STEPS\s+\*+\s+([\d\.]+)\s', line)
            if n_steps_match:
                n_steps = int(n_steps_match.group(1))
                print(f'Found n_steps: {n_steps}')
                break
    
    if n_steps is None:
        raise ValueError("n_steps not found in the file")
    
    # Initialize a set to keep track of processed load cases
    processed_loadcases = set()
    
    # Ensure the loads column contains only string values
    forces_df["loads"] = forces_df["loads"].astype(str)
    
    # Check for NaNs or empty cells in the loads column
    if forces_df["loads"].isnull().any() or forces_df["loads"].eq('').any():
        nan_rows = forces_df[forces_df["loads"].isnull() | forces_df["loads"].eq('')]
        raise ValueError(f"NaN or empty values found in 'loads' column at the following rows:\n{nan_rows}")
    
    # Search for load cases and corresponding wave heights and periods until the end of the file
    for i, line in enumerate(contents):
        if "* WAVE CASE" in line:
            loadcase_match = re.search(r'\* WAVE CASE\s+(\d+)\s+\*', line)
            if loadcase_match:
                loadcase = loadcase_match.group(1)
                
                # Skip this load case if it has already been processed
                if loadcase in processed_loadcases:
                    continue
                
                print(f'Found load case number: {loadcase}')
                
                # The next line should contain wave height and wave period
                next_line = contents[i + 1]
                wave_info_match = re.search(r'WAVE HEIGHT\s+=\s+([\d\.]+)\s+M\s+WAVE PERIOD\s+=\s+([\d\.]+)\s+SECS', next_line)
                if wave_info_match:
                    wave_height = wave_info_match.group(1)
                    wave_period = wave_info_match.group(2)
                    print(f'Found wave height: {wave_height} M, wave period: {wave_period} SECS')
                    
                    # Update the DataFrame for the corresponding range of load cases
                    start_loadcase = str((int(loadcase) - 1) * n_steps + 1)
                    end_loadcase = str(int(loadcase) * n_steps)
                    
                    # Ensure lexicographic comparison works correctly for zero-padded numbers
                    forces_df['loads'] = forces_df['loads'].apply(lambda x: x.zfill(len(end_loadcase)))
                    start_loadcase = start_loadcase.zfill(len(end_loadcase))
                    end_loadcase = end_loadcase.zfill(len(end_loadcase))
                    
                    forces_df.loc[(forces_df["loads"] >= start_loadcase) & (forces_df["loads"] <= end_loadcase), "wave_height"] = wave_height
                    forces_df.loc[(forces_df["loads"] >= start_loadcase) & (forces_df["loads"] <= end_loadcase), "wave_period"] = wave_period
                    print(f"Direction: {direction}",forces_df)
                    # Add this load case to the set of processed load cases
                    processed_loadcases.add(loadcase)
                    
    return forces_df

def validate_and_adjust(value_array, validity_range):
    lower_bound, upper_bound = validity_range
    # Check if values are within the validity range
    valid_values = np.clip(value_array, lower_bound, upper_bound)  # Clip values to stay within bounds
    return valid_values           

def interp_beta(R_lim, beta_actual):
    return R_lim + ((0.85 - R_lim) / (1 - 0.85)) * (beta_actual - 0.85)

def calculate_tubular_scf(joint_df):
    scf_df = joint_df.copy()
    # scf_df = joint_df[joint_df["joint"] == "GK38"].copy()
    # Filter for debugging
    # joint_filter = (scf_df["joint"] == "10D1") & (scf_df["brace1"] == "10D1-1181")
    # scf_df = scf_df.loc[joint_filter]
    # print(scf_df)

    cdata = pd.read_csv('constants.csv')
    C, n_crest, n_freq, method = [convert_to_float(value) for value in cdata.iloc[0]]
    C1 = 2 * (C - 0.5)
    C2 = C/2
    C3 = C/5
    # Chord Joint Parameters
    scf_df["alpha"] = α = 2 * scf_df["eff_chord"] * 1000 / scf_df["chd_OD1"]
    γ  =  scf_df["chd_OD1"] / (2 *scf_df["chd_THK1"])
    param_options = ["actual", "limit"]
    

	# cycle through braces for parameter calcualtions
    for brc in range(1, 4):
        # Brace Joint Parameters
        scf_df[f"β{brc}"] = scf_df[f"brc_OD{brc}"] / scf_df["chd_OD1"]
        scf_df[f"τ{brc}"] = scf_df[f"brc_THK{brc}"] / scf_df["chd_THK1"]
	
    # cycle through braces for brace SCF
    for brc in range(1, 4):
        # Initialize SCF Columns
        scf_df[f"T_SCF{brc}_CHD_SDL_FX"] = 0
        scf_df[f"T_SCF{brc}_CHD_CRN_FX"] = 0
        scf_df[f"T_SCF{brc}_BRC_SDL_FX"] = 0
        scf_df[f"T_SCF{brc}_BRC_CRN_FX"] = 0
        scf_df[f"T_SCF{brc}_CHD_CRN_IPB"] = 0
        scf_df[f"T_SCF{brc}_BRC_CRN_IPB"] = 0
        scf_df[f"T_SCF{brc}_CHD_SDL_OPB"] = 0
        scf_df[f"T_SCF{brc}_BRC_SDL_OPB"] = 0
        scf_df[f"X_SCF{brc}_CHD_SDL_FX_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_CHD_CRN_FX_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_BRC_SDL_FX_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_BRC_CRN_FX_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_CHD_CRN_IPB_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_BRC_CRN_IPB_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_CHD_SDL_OPB_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_BRC_SDL_OPB_BALANCED"] = 0
        scf_df[f"X_SCF{brc}_CHD_SDL_FX_ONE"] = 0
        scf_df[f"X_SCF{brc}_CHD_CRN_FX_ONE"] = 0
        scf_df[f"X_SCF{brc}_BRC_SDL_FX_ONE"] = 0
        scf_df[f"X_SCF{brc}_BRC_CRN_FX_ONE"] = 0
        scf_df[f"X_SCF{brc}_CHD_SDL_OPB_ONE"] = 0
        scf_df[f"X_SCF{brc}_BRC_SDL_OPB_ONE"] = 0
        # Initialize R root reduction Factors
        scf_df[f"T_R{brc}_FX"] = 0
        scf_df[f"T_R{brc}_IPB"] = 0
        scf_df[f"T_R{brc}_OPB"] = 0
        scf_df[f"X_R{brc}_FX"] = 0
        scf_df[f"X_R{brc}_IPB"] = 0
        scf_df[f"X_R{brc}_OPB"] = 0

        for param_option in param_options:
            β = scf_df[f"β{brc}"]
            β_actual = scf_df[f"β{brc}"].values
            τ = scf_df[f"τ{brc}"]
            θ = scf_df[f"angle{brc}"]

            # PARAMETERS USED FOR TOE SCF CALCULATIONS
            β_validity = [0.2, 1]
            γ_validity = [8, 32]
            τ_validity = [0.2, 1]
            θ_validity = [20, 90]
            ζ_validity = [(-0.6 * β) / (np.sin(np.radians(θ))), 1]

            # T-JOINT PARAMETERS USED FOR R-REDUCTION CALCULATIONS
            R_β_validity = [0.4, 0.85]
            R_γ_validity = [10, 30]
            R_τ_validity = [0.35, 0.85]
            R_θ_validity = [30, 90]

            R_β = β
            R_γ = γ
            R_τ = τ
            R_θ = θ

            if param_option == "limit":
                β = validate_and_adjust(β, β_validity)
                γ = validate_and_adjust(γ, γ_validity)
                τ = validate_and_adjust(τ, τ_validity)
                θ = validate_and_adjust(θ, θ_validity)
                R_β = validate_and_adjust(β, R_β_validity)
                R_γ = validate_and_adjust(γ, R_γ_validity)
                R_τ = validate_and_adjust(τ, R_τ_validity)
                R_θ = validate_and_adjust(θ, R_θ_validity)
            
            K_F1 = X_F1 = T_F1 = 1 - (0.83*β -(0.56*β**2) - 0.02) * (γ**0.23) * np.exp(-0.21 * (γ**-1.16) * (α**2.5)  )
            X_F2 = T_F2 = 1 - (1.43*β -(0.97*β**2) - 0.03) * (γ**0.04) * np.exp(-0.71 * (γ**-1.38) * (α**2.5)  )
            K_F3 = X_F3 = T_F3 = 1 - 0.55 * (β**1.8) * (γ**0.16) * np.exp(-0.49 * (γ**-0.89) * (α**1.8  ) )
            K_F4 = 1 - 1.07 * (β**1.88) * np.exp(-0.16 * (γ**-1.06) * α**2.4  )

            # ***********************************************************
            # ******************        T-JOINTS       ******************
            # ***********************************************************
            # Axial load-chord ends fixed
            EQ1 = (γ * (τ**1.1) * (1.11 - 3 * (β - 0.52)**2) * np.sin(np.radians(θ)) ** 1.6)
            T_SCF_CHD_SDL_FX_fixed = np.where(α < 12, T_F1 * EQ1, EQ1)
            EQ2 = T_SCF_CHD_CRN_FX_fixed = (γ**0.2) * (τ) * (2.65 + 5 * (β - 0.65)**2) + τ * β * (0.25 * α - 3) * np.sin(np.radians(θ))
            EQ3 = 1.3 + γ * (τ**0.52) * (α**0.1) * (0.187 - 1.25 * (β**1.1) * (β - 0.96)) * np.sin(np.radians(θ))**(2.7-0.01 * α)
            T_SCF_BRC_SDL_FX_fixed = np.where(α < 12, T_F1 * EQ3, EQ3)
            EQ4 = T_SCF_BRC_CRN_FX_fixed = 3 + (γ**1.2) * (0.12 * np.exp(-4*β) + 0.011*(β**2) -0.045) + β*τ * (0.1*α - 1.2) 
            # Axial load general fixity conditions
            EQ5 =  (EQ1 + C1 * (0.8*α - 6) * τ * (β**2) * ((1 -(β**2))**0.5) * np.sin(np.radians(2*θ)) ** 2)
            T_SCF_CHD_SDL_FX_general = np.where(α < 12, T_F2 * EQ5, EQ5)
            EQ6 = T_SCF_CHD_CRN_FX_general = (γ**0.2) * τ * (2.65 + 5*(β - 0.65)**2) + τ * β * (C2 * α - 3) * np.sin(np.radians(θ))
            T_SCF_BRC_SDL_FX_general = np.where(α < 12, T_F2 * EQ3, EQ3)
            EQ7 = T_SCF_BRC_CRN_FX_general = 3 + (γ**1.2) * (0.12*np.exp(-4*β) + 0.011*(β**2) -0.045) + β*τ * (C3*α - 1.2)    

            # T-Joint Root Reduction Factors R
            R_T_FX = (2.35*(0.318+1.557*R_β-1.802*(R_β**2))) * ((0.5+0.007*R_γ)) * ((0.85-0.556*R_τ)) * ((0.54+0.679*np.radians(R_θ)-0.246*(np.radians(R_θ)**2))) 
            R_T_IPB = (2.55*(0.419+0.334*R_β)) * ((0.578-0.002*R_γ+0.0002*(R_γ**2))) * ((0.611+0.252*R_τ-0.648*(R_τ**2))) * ((3.985-5.536*np.radians(R_θ)+2.314*(np.radians(R_θ)**2)))
            R_T_OPB = (2.4*(0.469+0.856*R_β-1.051*(R_β**2))) * ((0.456+0.014*R_γ-0.0002*(R_γ**2))) * ((0.856-0.6*R_τ)) * ((1.426-0.454*np.radians(R_θ)+0.117*(np.radians(R_θ)**2))) 

            if param_option == "limit":
                R_T_FX = np.where(β_actual > 0.85, interp_beta(R_T_FX, β_actual), R_T_FX )
                R_T_IPB = np.where(β_actual > 0.85, interp_beta(R_T_IPB, β_actual), R_T_IPB )
                R_T_OPB = np.where(β_actual > 0.85,interp_beta(R_T_OPB, β_actual),R_T_OPB   )

            # Applying Fixity Coefficient
            # T_SCF_CHD_SDL_FX = T_SCF_CHD_SDL_FX_fixed * C + T_SCF_CHD_SDL_FX_general * (1-C)
            # T_SCF_CHD_CRN_FX = T_SCF_CHD_CRN_FX_fixed * C + T_SCF_CHD_CRN_FX_general * (1-C)
            # T_SCF_BRC_SDL_FX = T_SCF_BRC_SDL_FX_fixed * C + T_SCF_BRC_SDL_FX_general * (1-C)
            # T_SCF_BRC_CRN_FX = T_SCF_BRC_CRN_FX_fixed * C + T_SCF_BRC_CRN_FX_general * (1-C)
            T_SCF_CHD_SDL_FX = T_SCF_CHD_SDL_FX_general
            T_SCF_CHD_CRN_FX = T_SCF_CHD_CRN_FX_general
            T_SCF_BRC_SDL_FX = T_SCF_BRC_SDL_FX_general
            T_SCF_BRC_CRN_FX = T_SCF_BRC_CRN_FX_general
            
            scf_df[f"T_SCF{brc}_CHD_SDL_FX"] = np.maximum(scf_df[f"T_SCF{brc}_CHD_SDL_FX"], T_SCF_CHD_SDL_FX)
            scf_df[f"T_SCF{brc}_CHD_CRN_FX"] = np.maximum(scf_df[f"T_SCF{brc}_CHD_CRN_FX"] , T_SCF_CHD_CRN_FX)
            scf_df[f"T_SCF{brc}_BRC_SDL_FX"] = np.maximum(scf_df[f"T_SCF{brc}_BRC_SDL_FX"], T_SCF_BRC_SDL_FX)
            scf_df[f"T_SCF{brc}_BRC_CRN_FX"] = np.maximum(scf_df[f"T_SCF{brc}_BRC_CRN_FX"], T_SCF_BRC_CRN_FX)

            scf_df[f"T_R{brc}_FX"] = np.maximum( scf_df[f"T_R{brc}_FX"], R_T_FX)
            scf_df[f"T_R{brc}_IPB"] = np.maximum(scf_df[f"T_R{brc}_IPB"], R_T_IPB)
            scf_df[f"T_R{brc}_OPB"] = np.maximum(scf_df[f"T_R{brc}_OPB"], R_T_OPB)

            # IPB bending
            EQ8 = T_SCF_CHD_CRN_IPB = 1.45 * β * (τ**0.85) * (γ**(1-0.68*β)) * np.sin(np.radians(θ))**0.7
            EQ9 = T_SCF_BRC_CRN_IPB = 1 + 0.65 * β * (τ**0.4) * (γ**(1.09-0.77*β)) * np.sin(np.radians(θ))**(0.06*γ-1.16)
            
            # OPB bending
            EQ10 =  (γ * τ * β *  (1.7 - 1.05 * (β**3)) * np.sin(np.radians(θ))**(1.6))
            T_SCF_CHD_SDL_OPB  = np.where(α < 12, T_F3 * EQ10, EQ10) 
            EQ11 = ((τ**-0.54) * (γ**-0.05) * (0.99 - 0.47*β + 0.08*(β**4)) * EQ10)
            T_SCF_BRC_SDL_OPB  = np.where(α < 12, T_F3 * EQ11, EQ11)

            scf_df[f"T_SCF{brc}_CHD_CRN_IPB"] = np.maximum(scf_df[f"T_SCF{brc}_CHD_CRN_IPB"], T_SCF_CHD_CRN_IPB)
            scf_df[f"T_SCF{brc}_BRC_CRN_IPB"] = np.maximum(scf_df[f"T_SCF{brc}_BRC_CRN_IPB"], T_SCF_BRC_CRN_IPB)
            scf_df[f"T_SCF{brc}_CHD_SDL_OPB"] = np.maximum(scf_df[f"T_SCF{brc}_CHD_SDL_OPB"], T_SCF_CHD_SDL_OPB)
            scf_df[f"T_SCF{brc}_BRC_SDL_OPB"] = np.maximum(scf_df[f"T_SCF{brc}_BRC_SDL_OPB"], T_SCF_BRC_SDL_OPB)

            # ***********************************************************
            # ******************        X-JOINTS        ******************
            # ***********************************************************
            # Balanced Loads
            # Axial load (balanced)
            # Create a mask where "X" is a substring in the "type" column
            # X-JOINT PARAMETERS USED FOR R-REDUCTION CALCULATIONS
            R_β_validity = [0.4, 0.85]
            R_γ_validity = [10, 30]
            R_τ_validity = [0.35, 0.85]

            if param_option == "limit":
                R_β = validate_and_adjust(β, R_β_validity)
                R_γ = validate_and_adjust(γ, R_γ_validity)
                R_τ = validate_and_adjust(τ, R_τ_validity)

            X_mask = scf_df["type"].str.contains("X")
            X_F1_BALANCED = X_F1
            X_F2_BALANCED = X_F2
            X_F_BALANCED = X_F1_BALANCED * C + X_F2_BALANCED * (1-C)
            
            EQ12 =  3.87 * γ * τ * β * (1.10 - (β**1.8)) * np.sin(np.radians(θ))**(1.7)
            X_SCF_CHD_SDL_FX_BALANCED = np.where(α < 12, X_F_BALANCED * EQ12, EQ12) 
            EQ13 = X_SCF_CHD_CRN_FX_BALANCED = (γ **0.2) * τ * (2.65 + 5 * ((β - 0.65)**2) ) - 3 * τ * β * np.sin(np.radians(θ))
            EQ14 = 1 + 1.9 * γ * (τ**0.5) * (β**0.9) * (1.09 - (β**1.7)) * np.sin(np.radians(θ))**2.5 
            X_SCF_BRC_SDL_FX_BALANCED = np.where(α < 12, X_F_BALANCED * EQ14, EQ14) 
            EQ15 = X_SCF_BRC_CRN_FX_BALANCED = 3  + (γ**1.2) * (0.12 * np.exp(-4*β) + 0.011*(β**2) - 0.045 )

            # X-Joint Root Reduction Factors R
            R_X_FX = (2.25*(0.326+1.565*R_β-1.734*(R_β**2))) * ((0.53+0.0065*R_γ)) * ((0.297+2.496*R_τ-5.117*(R_τ**2)+2.687*(R_τ**3)))
            R_X_IPB = (2.5*(0.548+0.25*R_β)) * ((0.472+0.008*R_γ)) * ((4.136*R_τ-0.073-7.478*(R_τ**2)+3.772*(R_τ**3)))
            R_X_OPB = (2.4*(0.453+0.981*R_β-1.188*(R_β**2))) * ((0.46+0.014*R_γ-0.0002*(R_γ**2))) * ((0.194+3.101*R_τ-6.33*(R_τ**2)+3.414*(R_τ**3)))

            if param_option == "limit":
                R_X_FX = np.where(β_actual > 0.85, interp_beta(R_X_FX, β_actual), R_X_FX )
                R_X_IPB = np.where(β_actual > 0.85, interp_beta(R_X_IPB, β_actual), R_X_IPB )
                R_X_OPB = np.where(β_actual > 0.85,interp_beta(R_X_OPB, β_actual),R_X_OPB   )

            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_FX_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_FX_BALANCED"], X_SCF_CHD_SDL_FX_BALANCED[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_CRN_FX_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_CRN_FX_BALANCED"], X_SCF_CHD_CRN_FX_BALANCED[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_FX_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_FX_BALANCED"], X_SCF_BRC_SDL_FX_BALANCED[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_CRN_FX_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_CRN_FX_BALANCED"], X_SCF_BRC_CRN_FX_BALANCED[X_mask])
            
            # IPB load (balanced)
            X_SCF_CHD_CRN_IPB_BALANCED = EQ8
            X_SCF_BRC_CRN_IPB_BALANCED = EQ9
            # OPB load (balanced)
            X_F3_BALANCED = X_F3
            EQ16 =  γ * τ * β * (1.56 - 1.3 * (β**4)) * np.sin(np.radians(θ))**1.6
            X_SCF_CHD_SDL_OPB_BALANCED = np.where(α < 12, X_F3_BALANCED * EQ16, EQ16) 
            EQ17 =  (τ**-0.54) * (γ**-0.05) * (0.99 - 0.47*β + 0.08*(β**4)) * EQ16
            X_SCF_BRC_SDL_OPB_BALANCED = np.where(α < 12, X_F3_BALANCED * EQ17, EQ17) 

            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_CRN_IPB_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_CRN_IPB_BALANCED"], X_SCF_CHD_CRN_IPB_BALANCED[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_CRN_IPB_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_CRN_IPB_BALANCED"], X_SCF_BRC_CRN_IPB_BALANCED[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_OPB_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_OPB_BALANCED"], X_SCF_CHD_SDL_OPB_BALANCED[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_OPB_BALANCED"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_OPB_BALANCED"], X_SCF_BRC_SDL_OPB_BALANCED[X_mask])

            scf_df.loc[X_mask, f"X_R{brc}_FX"] = np.maximum(scf_df.loc[X_mask,f"X_R{brc}_FX"], R_X_FX[X_mask])
            scf_df.loc[X_mask, f"X_R{brc}_IPB"] = np.maximum(scf_df.loc[X_mask,f"X_R{brc}_IPB"], R_X_IPB[X_mask])
            scf_df.loc[X_mask, f"X_R{brc}_OPB"] = np.maximum(scf_df.loc[X_mask,f"X_R{brc}_OPB"], R_X_OPB[X_mask])

            # Axial Load in 1x Brace Only
            X_F1_ONE = X_F1
            X_F2_ONE = X_F2
            X_F_ONE = X_F1_ONE * C + X_F2_ONE * (1-C)

            EQ18 = (1 - 0.26 * (β**3)) * EQ5
            X_SCF_CHD_SDL_FX_ONE = np.where(α < 12, X_F_ONE * EQ18, EQ18) 
            X_SCF_CHD_CRN_FX_ONE = EQ6
            EQ19 = (1 - 0.26 * (β**3)) * EQ3
            X_SCF_BRC_SDL_FX_ONE = np.where(α < 12, X_F_ONE * EQ19, EQ19) 
            X_SCF_BRC_CRN_FX_ONE = EQ7

            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_FX_ONE"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_FX_ONE"], X_SCF_CHD_SDL_FX_ONE[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_CRN_FX_ONE"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_CRN_FX_ONE"], X_SCF_CHD_CRN_FX_ONE[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_FX_ONE"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_FX_ONE"], X_SCF_BRC_SDL_FX_ONE[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_CRN_FX_ONE"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_CRN_FX_ONE"], X_SCF_BRC_CRN_FX_ONE[X_mask])

            # OPB bending on one brace only
            X_F3_ONE = X_F3
            X_SCF_CHD_SDL_OPB_ONE = np.where(α < 12, X_F3_ONE * EQ10, EQ10) 
            X_SCF_BRC_SDL_OPB_ONE = np.where(α < 12, X_F3_ONE * EQ11, EQ11) 

            scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_OPB_ONE"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_CHD_SDL_OPB_ONE"], X_SCF_CHD_SDL_OPB_ONE[X_mask])
            scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_OPB_ONE"] = np.maximum(scf_df.loc[X_mask, f"X_SCF{brc}_BRC_SDL_OPB_ONE"], X_SCF_BRC_SDL_OPB_ONE[X_mask])

            # ***********************************************************
            # ******************        K-JOINTS         ****************
            # ***********************************************************
            # Get unique combinations of braces 1, 2, 3 for K-Joint Checks
            K_mask = scf_df["type"].str.contains("K")

            # K-JOINT PARAMETERS USED FOR R-REDUCTION CALCULATIONS

            R_β_validity = [0.2, 0.9]
            R_γ_validity = [10, 30]
            R_τ_validity = [0.2, 1]
            R_θ_validity = [30, 60]
            R_ζ_validity = [0.042, 0.175]

            if param_option == "limit":
                R_β = validate_and_adjust(β, R_β_validity)
                R_γ = validate_and_adjust(γ, R_γ_validity)
                R_τ = validate_and_adjust(τ, R_τ_validity)
                R_θ = validate_and_adjust(θ, R_θ_validity)
            

            for brc2 in range(1, 4):
                if brc2 != brc:
                    # Initialize K SCF columns
                    scf_df[f"K_SCF{brc}{brc2}_CHD_SDL_FX_BALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_CRN_FX_BALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_SDL_FX_BALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_CRN_FX_BALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_CRN_IPB_UNBALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_CRN_IPB_UNBALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_SDL_OPB_UNBALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_SDL_OPB_UNBALANCED"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_SDL_FX_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_CRN_FX_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_SDL_FX_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_CRN_FX_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_SDL_OPB_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_SDL_OPB_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_CHD_CRN_IPB_ONE"] = 0
                    scf_df[f"K_SCF{brc}{brc2}_BRC_CRN_IPB_ONE"] = 0
                    # Initialize R Reduction Factors
                    scf_df[f"K_R{brc}{brc2}_FX"] = 0
                    scf_df[f"K_R{brc}{brc2}_IPB"] = 0
                    scf_df[f"K_R{brc}{brc2}_OPB"] = 0

                    braces_exists = ~scf_df[f"brace{brc}"].isna() & (scf_df[f"brace{brc}"] != '') & ~scf_df[f"brace{brc2}"].isna() & (scf_df[f"brace{brc2}"] != '')
                    gap = np.where((brc == 1) & (brc2 == 2) | (brc == 2) & (brc2 == 1), scf_df["gap1"], np.where((brc == 2) & (brc2 == 3) | (brc == 3) & (brc2 == 2), scf_df["gap2"], scf_df["gap3"]))
                    ζ = gap / scf_df["chd_OD1"]
                    R_ζ = ζ
                    
                    if param_option == "limit":
                        ζ = validate_and_adjust(ζ, ζ_validity)
                        R_ζ = validate_and_adjust(ζ, R_ζ_validity)

                    C1 = np.where(gap > 0, 0, 0.5)

                    βmax = scf_df[[f"β{brc}", f"β{brc2}"]].max(axis=1)
                    θmax = scf_df[[f"angle{brc}", f"angle{brc2}"]].max(axis=1)
                    βmin = scf_df[[f"β{brc}", f"β{brc2}"]].min(axis=1)
                    θmin = scf_df[[f"angle{brc}", f"angle{brc2}"]].min(axis=1)

                    if param_option == "limit":
                        βmax = validate_and_adjust(βmax, β_validity)
                        θmax = validate_and_adjust(θmax, θ_validity)
                        βmin = validate_and_adjust(βmin, β_validity)
                        θmin = validate_and_adjust(θmin, θ_validity)

                    # K-Joint Root Reduction Factors R
                    
                    R_K_FX = 2.6*(0.203+1.66*R_β-1.3*(R_β**2)) * ((0.47+0.024*R_γ-0.00054*(R_γ**2))) * ((((0.187*R_τ)/(((R_τ**2)+(0.52**2))**2))+0.39)) * ((0.808+1.053*np.radians(R_θ)-1.029*(np.radians(R_θ)**2))) * ((1.64-(0.005**(0.0012/R_ζ))))
                    R_K_IPB = (3.04*(((0.31*R_β)/(((R_β**2)+(0.77**2))**2))+0.37)) * ((((0.44*R_τ)/(((R_τ**2)+(0.67**2))**2))+0.13)) * ((6.22-10.6*np.radians(R_θ)+5.034*(np.radians(R_θ)**2))) * ((1.97-(R_ζ**(-0.13))))
                    R_K_OPB = (4.82*(1.04-1.17*R_β+0.7*(R_β**2))) * ((R_γ**(-0.175))) * ((((0.28*R_τ)/(((R_τ**2)+(0.57**2))**2))+0.17)) * ((1.56-0.663*np.radians(R_θ))) * (((R_ζ**(-0.017))-0.45))

                    if param_option == "limit":
                        R_K_FX = np.where(β_actual > 0.85, interp_beta(R_K_FX, β_actual), R_K_FX )
                        R_K_IPB = np.where(β_actual > 0.85, interp_beta(R_K_IPB, β_actual), R_K_IPB )
                        R_K_OPB = np.where(β_actual > 0.85,interp_beta(R_K_OPB, β_actual),R_K_OPB   )

                    # K BALANCED
                    # Axial Balanced K Brace
                    EQ20 = ((τ**0.9) * (γ**0.5) * (0.67 - (β**2) +1.16*β) * np.sin(np.radians(θ)) * (np.sin(np.radians(θmax)) 
                            / np.sin(np.radians(θmin)))**0.30 * (βmax/βmin)**0.30 * (1.64 + 0.29*(β**(-0.38)) * np.arctan(8 * ζ)))
                    K_SCF_CHD_SDL_FX_BALANCED = K_SCF_CHD_CRN_FX_BALANCED = EQ20
                    EQ21 = 1 + (1.97 - 1.57*(β**0.25)) * (τ**(-0.14)) * np.sin(np.radians(θ))**0.7 * EQ20 +  (np.sin(np.radians(θmax-θmin))**1.8) * (0.131 - 0.084 * np.arctan(14*ζ + 4.2*β )) * C1 * (β**1.5) * (γ**0.5) * (τ**-1.22)
                    K_SCF_BRC_SDL_FX_BALANCED = K_SCF_BRC_CRN_FX_BALANCED = EQ21
                    
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_FX_BALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_FX_BALANCED"] , K_SCF_CHD_SDL_FX_BALANCED[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_FX_BALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_FX_BALANCED"] , K_SCF_CHD_CRN_FX_BALANCED[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_FX_BALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_FX_BALANCED"] , K_SCF_BRC_SDL_FX_BALANCED[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_FX_BALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_FX_BALANCED"] , K_SCF_BRC_CRN_FX_BALANCED[K_mask & braces_exists])

                
                    scf_df.loc[K_mask & braces_exists, f"K_R{brc}{brc2}_FX"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_R{brc}{brc2}_FX"] , R_K_FX[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_R{brc}{brc2}_IPB"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_R{brc}{brc2}_IPB"], R_K_IPB[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_R{brc}{brc2}_OPB"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_R{brc}{brc2}_OPB"], R_K_OPB[K_mask & braces_exists])

                    # K UNBALANCED
                    # IPB Unbalanced K Brace
                    K_SCF_CHD_CRN_IPB_UNBALANCED = EQ8
                    K_SCF_BRC_CRN_IPB_UNBALANCED = np.where(gap > 0, EQ9, EQ9*(0.9 + 0.4*β))
                    
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_IPB_UNBALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_IPB_UNBALANCED"], K_SCF_CHD_CRN_IPB_UNBALANCED[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_IPB_UNBALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_IPB_UNBALANCED"], K_SCF_BRC_CRN_IPB_UNBALANCED[K_mask & braces_exists])
                    
                    # OPB Unbalanced K Brace
                    x = 1 + (ζ * np.sin(np.radians(θ))) / β
                    θ_b, τ_b, β_b = scf_df[f"angle{brc2}"], scf_df[f"τ{brc2}"], scf_df[f"β{brc2}"]

                    if param_option == "limit":                    
                        θ_b = validate_and_adjust(θ_b, θ_validity)
                        τ_b = validate_and_adjust(τ_b, τ_validity)
                        β_b = validate_and_adjust(β_b, β_validity)

                    EQ10_B =  (γ * τ_b * β_b *  (1.7-1.05 * (β_b**3)) * np.sin(np.radians(θ_b))**(1.6))
                    EQ23 = EQ10 * (1 - 0.08 * ((β_b * γ )**0.5) * np.exp(-0.8 * x)) + EQ10_B * (1 - 0.08 * ((β*γ)**0.5) * np.exp(-0.8 * x)) * (2.05 * (βmax**0.5) * np.exp(-1.3 * x))
        
                    K_SCF_CHD_SDL_OPB_UNBALANCED = np.where(α < 12, K_F4 * EQ23, EQ23)
                    EQ24 = K_SCF_BRC_SDL_OPB_UNBALANCED = (τ**(-0.54)) * (γ**(-0.05)) * (0.99 - 0.47*β + 0.08*(β**4)) * EQ23
                    K_SCF_BRC_SDL_OPB_UNBALANCED = np.where(α < 12, K_F4 * EQ24, EQ24)
                    
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_OPB_UNBALANCED"], K_SCF_CHD_SDL_OPB_UNBALANCED[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_OPB_UNBALANCED"], K_SCF_BRC_SDL_OPB_UNBALANCED[K_mask & braces_exists])
                    
                    # K ONE BRACE
                    # Axial load on one brace only
                    K_SCF_CHD_SDL_FX_ONE = np.where(α < 12, K_F1 * EQ5, EQ5)
                    K_SCF_CHD_CRN_FX_ONE = np.where(α < 12, K_F1 * EQ6, EQ6)
                    K_SCF_BRC_SDL_FX_ONE = np.where(α < 12, K_F1 * EQ3, EQ3)
                    K_SCF_BRC_CRN_FX_ONE = np.where(α < 12, K_F1 * EQ7, EQ7)
                    
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_FX_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_FX_ONE"], K_SCF_CHD_SDL_FX_ONE[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_FX_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_FX_ONE"], K_SCF_CHD_CRN_FX_ONE[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_FX_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_FX_ONE"], K_SCF_BRC_SDL_FX_ONE[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_FX_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_FX_ONE"], K_SCF_BRC_CRN_FX_ONE[K_mask & braces_exists])	

                    # IPB bending on one brace only
                    K_SCF_CHD_CRN_IPB_ONE = EQ8
                    K_SCF_BRC_CRN_IPB_ONE = EQ9
                    
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_IPB_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_CRN_IPB_ONE"], K_SCF_CHD_CRN_IPB_ONE[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_IPB_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_CRN_IPB_ONE"], K_SCF_BRC_CRN_IPB_ONE[K_mask & braces_exists])
                    
                    # OPB bending on one brace only
                    EQ25 = EQ10 * (1 - 0.08 * ((β_b*γ)**0.5) * np.exp(-0.8 * x))
                    EQ26 = (τ**(-0.54)) * (γ**(-0.05)) * (0.99 - 0.47*β + 0.08*(β**4)) * EQ25
                    K_SCF_CHD_SDL_OPB_ONE = np.where(α < 12, K_F3 * EQ25, EQ25)
                    K_SCF_BRC_SDL_OPB_ONE = np.where(α < 12, K_F3 * EQ26, EQ26)

                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_OPB_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_CHD_SDL_OPB_ONE"], K_SCF_CHD_SDL_OPB_ONE[K_mask & braces_exists])
                    scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_OPB_ONE"] = np.maximum(scf_df.loc[K_mask & braces_exists, f"K_SCF{brc}{brc2}_BRC_SDL_OPB_ONE"], K_SCF_BRC_SDL_OPB_ONE[K_mask & braces_exists])	

    # ***********************************************************
	# ******************        KT-JOINTS         ****************
	# ***********************************************************
	# Get unique combinations of braces 1, 2, 3 for KT-Joint Checks
	# cycle through braces for brace SCF
    # Pre-Calculate all parameters
    EQ10, x, β, τ, θ = {}, {}, {}, {}, {}
    ζab = scf_df["gap1"] / scf_df["chd_OD1"]
    ζbc = scf_df["gap2"] / scf_df["chd_OD1"]
    if param_option == "limit":
        ζab = validate_and_adjust(ζab, ζ_validity)
        ζbc = validate_and_adjust(ζbc, ζ_validity)

    ζac = ζ1 = ζ3 = ζab + ζbc + scf_df[f"β2"]
    ζ2 = np.maximum(ζab, ζbc)
    ζ = {"1": ζ1, "2": ζ2, "3": ζ3}
    KT_mask = scf_df["type"].str.contains("KT")

    for brc in range(1, 4):
         # Brace Joint Parameters
        β[f"{brc}"] = scf_df[f"β{brc}"]
        τ[f"{brc}"] = scf_df[f"τ{brc}"]
        θ[f"{brc}"] = scf_df[f"angle{brc}"]
        
        θmax = pd.DataFrame(θ).max(axis=1)
        θmin = pd.DataFrame(θ).min(axis=1)  
        βmax = pd.DataFrame(β).max(axis=1)
        βmin = pd.DataFrame(β).min(axis=1)

        if param_option == "limit":
            β[f"{brc}"] = validate_and_adjust(scf_df[f"β{brc}"], β_validity)
            τ[f"{brc}"] = validate_and_adjust(scf_df[f"τ{brc}"], τ_validity)
            θ[f"{brc}"] = validate_and_adjust(scf_df[f"angle{brc}"], θ_validity)     
            θmax = validate_and_adjust(θmax, θ_validity)
            θmin = validate_and_adjust(θmin, θ_validity)
            βmax = validate_and_adjust(βmax, β_validity)
            βmin = validate_and_adjust(βmin, β_validity)  
        
        x[f"{brc}"] = 1 + (ζ[f"{brc}"] * np.sin(np.radians(θ[f"{brc}"]))) / β[f"{brc}"]
        EQ10[f"{brc}"] =  γ * τ[f"{brc}"] * β[f"{brc}"] *  (1.7 - 1.05*(β[f"{brc}"])**3) * np.sin(np.radians(θ[f"{brc}"]))**(1.6)
        


    for brc in range(1, 4):

        # Initialize SCF Columns 
        scf_df[f"KT_SCF{brc}_CHD_SDL_FX_BALANCED"] = 0
        scf_df[f"KT_SCF{brc}_CHD_CRN_FX_BALANCED"] = 0
        scf_df[f"KT_SCF{brc}_BRC_SDL_FX_BALANCED"] = 0
        scf_df[f"KT_SCF{brc}_BRC_CRN_FX_BALANCED"] = 0
        scf_df[f"KT_SCF{brc}_CHD_CRN_IPB_BALANCED"] = 0
        scf_df[f"KT_SCF{brc}_BRC_CRN_IPB_BALANCED"] = 0
        scf_df[f"KT_SCF{brc}_CHD_SDL_FX_ONE"] = 0
        scf_df[f"KT_SCF{brc}_CHD_CRN_FX_ONE"] = 0
        scf_df[f"KT_SCF{brc}_BRC_SDL_FX_ONE"] = 0
        scf_df[f"KT_SCF{brc}_BRC_CRN_FX_ONE"] = 0
        scf_df[f"KT_SCF{brc}_CHD_SDL_OPB_UNBALANCED"] = 0
        scf_df[f"KT_SCF{brc}_BRC_SDL_OPB_UNBALANCED"] = 0
        scf_df[f"KT_SCF{brc}_CHD_SDL_OPB_ONE"] = 0
        scf_df[f"KT_SCF{brc}_BRC_SDL_OPB_ONE"] = 0

        gap = scf_df[f"gap{brc}"]
        C1 = np.where(gap > 0, 0, 0.5)

        

        EQ20 = ((τ[f"{brc}"]**0.9) * (γ**0.5) * (0.67 - (β[f"{brc}"]**2) +1.16*β[f"{brc}"]) * np.sin(np.radians(θ[f"{brc}"])) * ((np.sin(np.radians(θmax)) 
            / np.sin(np.radians(θmin)))**0.30) * ((βmax/βmin)**0.30) * (1.64 + 0.29*(β[f"{brc}"]**(-0.38)) * np.arctan(8 * ζ[f"{brc}"])))
        # Axial Load Balanced
        EQ21 = (1 + (1.97 - 1.57*(β[f"{brc}"]**0.25)) * (τ[f"{brc}"]**(-0.14)) * (np.sin(np.radians(θ[f"{brc}"]))**0.7) * EQ20  
            + (np.sin(np.radians(θmax-θmin))**1.8) * (0.131 - 0.084 * np.arctan(14*ζ[f"{brc}"] + 4.2*β[f"{brc}"]  )) * C1 * (β[f"{brc}"] **1.5) * (γ**0.5) * (τ[f"{brc}"]**-1.22))
        
        KT_SCF_CHD_SDL_FX_BALANCED = EQ20
        KT_SCF_CHD_CRN_FX_BALANCED = EQ20
        KT_SCF_BRC_SDL_FX_BALANCED = EQ21
        KT_SCF_BRC_CRN_FX_BALANCED = EQ21

        scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_SDL_FX_BALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_SDL_FX_BALANCED"], KT_SCF_CHD_SDL_FX_BALANCED[KT_mask])
        scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_CRN_FX_BALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_CRN_FX_BALANCED"], KT_SCF_CHD_CRN_FX_BALANCED[KT_mask])				
        scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_SDL_FX_BALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_SDL_FX_BALANCED"], KT_SCF_BRC_SDL_FX_BALANCED[KT_mask])
        scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_CRN_FX_BALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_CRN_FX_BALANCED"], KT_SCF_BRC_CRN_FX_BALANCED[KT_mask])

        # IPB Balanced
        EQ8 = T_SCF_CHD_CRN_IPB = 1.45 * β[f"{brc}"] * (τ[f"{brc}"]**0.85) * (γ**(1-0.68*β[f"{brc}"])) * np.sin(np.radians(θ[f"{brc}"]))**0.7
        EQ9 = T_SCF_BRC_CRN_IPB = 1 + 0.65 * β[f"{brc}"] * (τ[f"{brc}"]**0.4) * (γ**(1.09-0.77*β[f"{brc}"])) * np.sin(np.radians(θ[f"{brc}"]))**(0.06*γ-1.16)
        KT_SCF_CHD_CRN_IPB_BALANCED = EQ8
        KT_SCF_BRC_CRN_IPB_BALANCED = EQ9

        scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_CRN_IPB_BALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_CRN_IPB_BALANCED"], KT_SCF_CHD_CRN_IPB_BALANCED[KT_mask])
        scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_CRN_IPB_BALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_CRN_IPB_BALANCED"], KT_SCF_BRC_CRN_IPB_BALANCED[KT_mask])
		
		# AXIAL ONE brace [SAME AS SIMPLE T JOINT SCF]
        scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_SDL_FX_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_SDL_FX_ONE"], scf_df.loc[KT_mask, f"T_SCF{brc}_CHD_SDL_FX"])
        scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_CRN_FX_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_CHD_CRN_FX_ONE"], scf_df.loc[KT_mask, f"T_SCF{brc}_CHD_CRN_FX"])
        scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_SDL_FX_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_SDL_FX_ONE"], scf_df.loc[KT_mask, f"T_SCF{brc}_BRC_SDL_FX"])
        scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_CRN_FX_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF{brc}_BRC_CRN_FX_ONE"], scf_df.loc[KT_mask, f"T_SCF{brc}_BRC_CRN_FX"])
		
    # OPB Unbalanced
	# Brace A
    x_ab = 1 + (ζab * np.sin(np.radians(θ["1"]))) / β["1"]
    x_ac = 1 + (ζac * np.sin(np.radians(θ["1"]))) / β["1"]
	
    EQ27 = (EQ10["1"] * (1 - 0.08 * ((β["2"]*γ)**0.5) * np.exp(-0.8 * x_ab)) * (1 - 0.08 * ((β["3"]*γ)**0.5) * np.exp(-0.8 * x_ac)) + 
            EQ10["2"] * (1 - 0.08 * ((β["1"]*γ)**0.5) * np.exp(-0.8 * x_ab)) * 2.05 * (βmax**0.5) * np.exp(-1.3 * x_ab) + 
            EQ10["3"] * (1 - 0.08 * ((β["1"]*γ)**0.5) * np.exp(-0.8 * x_ac)) * 2.05 * (βmax**0.5) * np.exp(-1.3 * x_ac) )
    EQ29 = ((τ["1"]**-0.54) * (γ**-0.05) * (0.99 - 0.47*β["1"] + 0.08*(β["1"]**4))) * EQ27

    KT_SCF1_CHD_SDL_OPB_UNBALANCED = EQ27
    KT_SCF1_BRC_SDL_OPB_UNBALANCED = EQ29

    scf_df.loc[KT_mask, f"KT_SCF1_CHD_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF1_CHD_SDL_OPB_UNBALANCED"], KT_SCF1_CHD_SDL_OPB_UNBALANCED[KT_mask]) 
    scf_df.loc[KT_mask, f"KT_SCF1_BRC_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF1_BRC_SDL_OPB_UNBALANCED"], KT_SCF1_BRC_SDL_OPB_UNBALANCED[KT_mask]) 	

	# OPB on ONE Brace Only
	# Brace A
    EQ30 = EQ10["1"] * (1 - 0.08*((β["2"]*γ)**0.5) * np.exp(-0.8 * x_ab)) * (1 - 0.08*(β["3"]*γ)**0.5) * np.exp(-0.8 * x_ac)
    EQ32 = ((τ["1"]**-0.54) * (γ**-0.05) * (0.99 - 0.47*β["1"] + 0.08*(β["1"]**4))) * EQ30
    KT_SCF1_CHD_SDL_OPB_ONE = EQ30
    KT_SCF1_BRC_SDL_OPB_ONE = EQ32
    scf_df.loc[KT_mask, f"KT_SCF1_CHD_SDL_OPB_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF1_CHD_SDL_OPB_ONE"], KT_SCF1_CHD_SDL_OPB_ONE[KT_mask]) 
    scf_df.loc[KT_mask, f"KT_SCF1_BRC_SDL_OPB_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF1_BRC_SDL_OPB_ONE"], KT_SCF1_BRC_SDL_OPB_ONE[KT_mask]) 
    
	# OPB Unbalanced
	# Brace B
    x_ab = 1 + (ζab * np.sin(np.radians(θ["2"]))) / β["2"]
    x_bc = 1 + (ζbc * np.sin(np.radians(θ["2"]))) / β["2"]
    P1 = (β["1"] / β["2"])**2
    P2 = (β["3"] / β["2"])**2
    EQ28 = (EQ10["2"] * ((1 - 0.08 * ((β["1"]*γ)**0.5) * np.exp(-0.8 * x_ab))**P1) * (1 - 0.08 * ((β["3"]*γ)**0.5) * np.exp(-0.8 * x_bc))**P2 + 
            EQ10["1"] * (1 - 0.08 * ((β["2"]*γ)**0.5) * np.exp(-0.8 * x_ab)) * 2.05 * (βmax**0.5) * np.exp(-1.3 * x_ab) + 
            EQ10["3"] * (1 - 0.08 * ((β["2"]*γ)**0.5) * np.exp(-0.8 * x_bc)) * 2.05 * (βmax**0.5) * np.exp(-1.3 * x_bc) )
    EQ29b = ((τ["2"]**-0.54) * (γ**-0.05) * (0.99 - 0.47*β["2"] + 0.08*(β["2"]**4))) * EQ28
    KT_SCF2_CHD_SDL_OPB_UNBALANCED = EQ28 
    KT_SCF2_BRC_SDL_OPB_UNBALANCED = EQ29b
    scf_df.loc[KT_mask, f"KT_SCF2_CHD_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF2_CHD_SDL_OPB_UNBALANCED"], KT_SCF2_CHD_SDL_OPB_UNBALANCED[KT_mask])
    scf_df.loc[KT_mask, f"KT_SCF2_BRC_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF2_BRC_SDL_OPB_UNBALANCED"], KT_SCF2_BRC_SDL_OPB_UNBALANCED[KT_mask])
    
	# OPB on ONE Brace Only
	# Brace B
    EQ31 = EQ10["2"] * ((1 - 0.08 * ((β["1"]*γ)**0.5) * np.exp(-0.8 * x_ab))**P1) * (1 - 0.08 * ((β["3"]*γ)**0.5) * np.exp(-0.8 * x_bc))**P2
    EQ32b = ((τ["2"]**-0.54) * (γ**-0.05) * (0.99 - 0.47*β["2"] + 0.08*(β["2"]**4))) * EQ31
    KT_SCF2_CHD_SDL_OPB_ONE = EQ31
    KT_SCF2_BRC_SDL_OPB_ONE = EQ32b
    scf_df.loc[KT_mask, f"KT_SCF2_CHD_SDL_OPB_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF2_CHD_SDL_OPB_ONE"], KT_SCF2_CHD_SDL_OPB_ONE[KT_mask])
    scf_df.loc[KT_mask, f"KT_SCF2_BRC_SDL_OPB_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF2_BRC_SDL_OPB_ONE"], KT_SCF2_BRC_SDL_OPB_ONE[KT_mask])
	
	# OPB Unbalanced
	# Brace C
    x_bc = 1 + (ζbc * np.sin(np.radians(θ["3"]))) / β["3"]
    x_ac = 1 + (ζac * np.sin(np.radians(θ["3"]))) / β["3"]
    EQ27c = (EQ10["3"] * (1 - 0.08 * ((β["2"]*γ)**0.5) * np.exp(-0.8 * x_bc)) * (1 - 0.08 * (β["1"]*γ)**0.5) * np.exp(-0.8 * x_ac) + 
            EQ10["2"] * (1 - 0.08 * ((β["3"]*γ)**0.5) * np.exp(-0.8 * x_bc)) * 2.05 * (βmax**0.5) * np.exp(-1.3 * x_bc) + 
            EQ10["1"] * (1 - 0.08 * ((β["3"]*γ)**0.5) * np.exp(-0.8 * x_ac)) * 2.05 * (βmax**0.5) * np.exp(-1.3 * x_ac) )
    EQ29c = ((τ["3"]**-0.54) * (γ**-0.05) * (0.99 - 0.47*β["3"] + 0.08*(β["3"]**4))) * EQ27c
    KT_SCF3_CHD_SDL_OPB_UNBALANCED = EQ27c
    KT_SCF3_BRC_SDL_OPB_UNBALANCED = EQ29c
    scf_df.loc[KT_mask, f"KT_SCF3_CHD_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF3_CHD_SDL_OPB_UNBALANCED"], KT_SCF3_CHD_SDL_OPB_UNBALANCED[KT_mask] )
    scf_df.loc[KT_mask, f"KT_SCF3_BRC_SDL_OPB_UNBALANCED"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF3_BRC_SDL_OPB_UNBALANCED"], KT_SCF3_BRC_SDL_OPB_UNBALANCED[KT_mask] )
	
    # OPB on ONE Brace Only
	# Brace C
    EQ30c = EQ10["3"] * (1 - 0.08 * ((β["2"]*γ)**0.5) * np.exp(-0.8 * x_bc)) * (1 - 0.08 * (β["3"]*γ)**0.5) * np.exp(-0.8 * x_ac)
    EQ32c = ((τ["3"]**-0.54) * (γ**-0.05) * (0.99 - 0.47*β["3"] + 0.08*(β["3"]**4))) * EQ30c
    KT_SCF3_CHD_SDL_OPB_ONE = EQ30c
    KT_SCF3_BRC_SDL_OPB_ONE = EQ32c
    scf_df.loc[KT_mask, f"KT_SCF3_CHD_SDL_OPB_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF3_CHD_SDL_OPB_ONE"], KT_SCF3_CHD_SDL_OPB_ONE[KT_mask])
    scf_df.loc[KT_mask, f"KT_SCF3_BRC_SDL_OPB_ONE"] = np.maximum(scf_df.loc[KT_mask, f"KT_SCF3_BRC_SDL_OPB_ONE"], KT_SCF3_BRC_SDL_OPB_ONE[KT_mask])
    print(scf_df)
    # scf_df.replace(0, "", inplace=True)
    return scf_df

def process_FLS_load_new(joint_data, forces_data, target_columns):

    member_keys = ['brace1', 'brace2', 'brace3', 'x-brace1', 'x-brace2', 'x-brace3']
    joint_data_output_columns = {member_key: target_columns[i*3:(i+1)*3] for i, member_key in enumerate(member_keys)}
    members_ipbs_columns = {member_key: ('chd_IPBs' if 'chord' in member_key else 'brc_IPB' + member_key[-1]) for member_key in member_keys}
    force_columns = ['FXA', 'MYA', 'MZA', 'FXB', 'MYB', 'MZB', 'direction', 'wave_height', 'wave_period']

    for member_key in member_keys:
        if member_key not in joint_data.columns:
            continue  # Skip if column missing

        member_ids = joint_data[member_key].astype(str)
        temp_df = joint_data[[member_key]].copy()
        temp_df = temp_df.merge(
            forces_data[['member'] + force_columns],
            left_on=member_key,
            right_on='member',
            how='left'
        ).drop(columns=['member'])

        temp_df_a = temp_df[force_columns[:3]].values
        temp_df_b = temp_df[force_columns[3:6]].values
        wave_data = temp_df[['direction', 'wave_height', 'wave_period']].values

        try:
            mask_a = joint_data['joint'].astype(str)  == member_ids.str[:4]
            mask_b = joint_data['joint'].astype(str)  == member_ids.str[-4:]
        except:
            mask_a = mask_b = pd.Series([False] * len(joint_data))

        mask_c = joint_data[members_ipbs_columns[member_key]] == "Z"

        # Write force components
        joint_data.loc[mask_a, joint_data_output_columns[member_key]] = temp_df_a[mask_a]
        joint_data.loc[mask_b, joint_data_output_columns[member_key]] = temp_df_b[mask_b]

        # Swap IPB/OPB if Z axis is active
        if mask_c.any():
            col2, col3 = joint_data_output_columns[member_key][1:3]
            joint_data.loc[mask_c, [col2, col3]] = joint_data.loc[mask_c, [col3, col2]].values

        # Populate wave info wherever a or b matched
        wave_mask = mask_a | mask_b
        joint_data.loc[wave_mask, ['direction', 'wave_height', 'wave_period']] = wave_data[wave_mask]

    return joint_data


def process_FLS_load(joint_data, forces_data, target_columns):
    member_keys = ['brace1', 'brace2', 'brace3','x-brace1','x-brace2','x-brace3']

    # Create output columns mapping for joint_data
    joint_data_output_columns = {member_key: target_columns[i*3:(i+1)*3] for i, member_key in enumerate(member_keys)}
    # print(joint_data_output_columns)
    # Create mapping for IPB columns
    members_ipbs_columns = {member_key: ('chd_IPBs' if 'chord' in member_key else 'brc_IPB' + member_key[-1])  for member_key in member_keys} 
    # print(members_ipbs_columns)
    # Force columns to retrieve from forces_data
    force_columns = ['FXA', 'MYA', 'MZA', 'FXB', 'MYB', 'MZB', 'direction', 'wave_height', 'wave_period']

    for member_key in member_keys:
        # Merge joint_data with forces_data to get relevant force columns including wave height and period
        temp_df = joint_data[[member_key]].merge(
            forces_data[['member'] + force_columns], # Retain force and wave columns
            left_on=member_key,
            right_on='member',
            how='left'
        ).drop(columns=['member'])
        
        # Split temp_df into parts: 'A' forces, 'B' forces, and wave data
        temp_df_a = temp_df[force_columns[:3]].values
        temp_df_b = temp_df[force_columns[3:6]].values
        wave_data = temp_df[['direction', 'wave_height', 'wave_period']].values

        # Determine which rows correspond to 'A' forces and which to 'B' forces
        mask_a = joint_data['joint'] == joint_data[member_key].str[:4]
        mask_b = joint_data['joint'] == joint_data[member_key].str[-4:]
        # Mask for changing Y/Z local bending axis
        mask_c = joint_data[members_ipbs_columns[member_key]] == "Z"

        # Update joint_data with the new columns
        joint_data.loc[mask_a, joint_data_output_columns[member_key]] = temp_df_a[mask_a]
        joint_data.loc[mask_b, joint_data_output_columns[member_key]] = temp_df_b[mask_b]

        # Update the second and third columns based on mask_c
        if mask_c.any():
            # Get the column indices for the second and third columns
            col_2_index = joint_data_output_columns[member_key][1]
            col_3_index = joint_data_output_columns[member_key][2]

            # Swap values between the second and third columns where mask_c is True
            joint_data.loc[mask_c, [col_2_index, col_3_index]] = joint_data.loc[mask_c, [col_3_index, col_2_index]].values

        # Add wave_height and wave_period to joint_data
        joint_data.loc[mask_a | mask_b, ['direction', 'wave_height', 'wave_period']] = wave_data[mask_a | mask_b]
        # joint_data[['wave_height', 'wave_period']] = wave_data
    return joint_data

def process_FLS_inline_load(inline_data, forces_data, target_columns, inline_type):
    if inline_type == "inline":
        member_keys = ['memb1', 'memb2']
    else: 
        member_keys = ['cone', 'tubular']

    # Create output columns mapping for inline_data
    inline_data_output_columns = {member_key: target_columns[i*3:(i+1)*3] for i, member_key in enumerate(member_keys)}
    # Force columns to retrieve from forces_data
    force_columns = ['FXA', 'MYA', 'MZA', 'FXB', 'MYB', 'MZB', 'direction', 'wave_height', 'wave_period']

    for member_key in member_keys:
        # Merge inline_data with forces_data to get relevant force columns including wave height and period
        temp_df = inline_data[[member_key]].merge(
            forces_data[['member'] + force_columns], # Retain force and wave columns
            left_on=member_key,
            right_on='member',
            how='left'
        ).drop(columns=['member'])
        
        # Split temp_df into parts: 'A' forces, 'B' forces, and wave data
        temp_df_a = temp_df[force_columns[:3]].values
        temp_df_b = temp_df[force_columns[3:6]].values
        wave_data = temp_df[['direction', 'wave_height', 'wave_period']].values

        # Determine which rows correspond to 'A' forces and which to 'B' forces
        mask_a = inline_data['joint'] == inline_data[member_key].str[:4]
        mask_b = inline_data['joint'] == inline_data[member_key].str[-4:]

        # Update inline_data with the new columns
        inline_data.loc[mask_a, inline_data_output_columns[member_key]] = temp_df_a[mask_a]
        inline_data.loc[mask_b, inline_data_output_columns[member_key]] = temp_df_b[mask_b]


        # Add wave_height and wave_period to inline_data
        inline_data.loc[mask_a | mask_b, ['direction', 'wave_height', 'wave_period']] = wave_data[mask_a | mask_b]
        # inline_data[['wave_height', 'wave_period']] = wave_data

        # mask_a.to_csv(f'testa.csv', index=False)
        # mask_b.to_csv(f'testb.csv', index=False)
    return inline_data

def axial_balanced(expanded_DF):

    # Initialize the new classification columns for each brace with zeros
    for i in range(1, 4):
        other_braces = [j for j in range(1, 4) if j != i]
        for classification in ["T", "X", "K", "KT"]:
            if classification == "K":
                expanded_DF[f"brc{i}{other_braces[0]}_{classification}_FX_balanced"] = 0  # Fill with zeros initially
                expanded_DF[f"brc{i}{other_braces[1]}_{classification}_FX_balanced"] = 0  # Fill with zeros initially
            else:
                expanded_DF[f"brc{i}_{classification}_FX_balanced"] = 0  # Fill with zeros initially  

    K_mask = expanded_DF["type"].str.contains("K")
    KT_mask = expanded_DF["type"].str.contains("KT")
    X_mask = expanded_DF["type"].str.contains("X")

    for i in range(1, 4):
        other_braces = [j for j in range(1, 4) if j != i]
        brace_exists = ~expanded_DF[f"brace{i}"].isna() & (expanded_DF[f"brace{i}"] != '')
        # Forces in braces for K-Joint evaluation
        alpha1 = expanded_DF[f"angle{i}"]
        alpha2 = expanded_DF[f"angle{other_braces[0]}"]
        alpha3 = expanded_DF[f"angle{other_braces[1]}"]
        # Safely convert FX values to floats

        FX1 = expanded_DF[f"brc{i}_FX"] * np.sin(np.radians(alpha1))
        FX2 = expanded_DF[f"brc{other_braces[0]}_FX"] * np.sin(np.radians(alpha2))
        FX3 = expanded_DF[f"brc{other_braces[1]}_FX"] * np.sin(np.radians(alpha3))

        # Calculate the tolerances (10% and 50% of the absolute value of the force in the current brace)
        tolerance_10 = 0.1 * FX1.abs() 
        tolerance_50 = 0.5 * FX1.abs()

        # Check if the brace is fully balanced (100%) by the other braces with 10% tolerance
        KT_balanced = ((FX1 + FX2 + FX3).abs() <= tolerance_10) 
        K_balanced2 = ((FX1 + FX2).abs() <= tolerance_10) 
        K_balanced3 = ((FX1 + FX3).abs() <= tolerance_10) 
        # Check if the brace is partially balanced (50%) by the other braces with 50% tolerance
        KT_half_balanced = ((FX1 + FX2 + FX3).abs() <= tolerance_50) & ~KT_balanced
        K_half_balanced2 = ((FX1 + FX2).abs() <= tolerance_50) & ~K_balanced2
        K_half_balanced3 = ((FX1 + FX3).abs() <= tolerance_50) & ~K_balanced3
        
        
        # Assign percentages for KT classifications
        expanded_DF.loc[KT_balanced & KT_mask & brace_exists, f"brc{i}_KT_FX_balanced"] = 100
        expanded_DF.loc[KT_half_balanced & KT_mask & brace_exists, f"brc{i}_KT_FX_balanced"] = 50
        # Assign percentages for K classifications
        expanded_DF.loc[K_balanced2 & K_mask & brace_exists, f"brc{i}{other_braces[0]}_K_FX_balanced"] = 100
        expanded_DF.loc[K_half_balanced2 & K_mask & brace_exists, f"brc{i}{other_braces[0]}_K_FX_balanced"] = 50
        expanded_DF.loc[K_balanced3 & K_mask & brace_exists, f"brc{i}{other_braces[1]}_K_FX_balanced"] = 100  
        expanded_DF.loc[K_half_balanced3 & K_mask & brace_exists, f"brc{i}{other_braces[1]}_K_FX_balanced"] = 50        
        
        # Now check if FX4, FX5, or FX6 are within 10% or 50% of FX1, FX2, or FX3 or their combinations
        # Mirrored Braces for x-brace evaluation
        xalpha1 = expanded_DF[f"x-angle{i}"]
        xalpha2 = expanded_DF[f"x-angle{other_braces[0]}"]
        xalpha3 = expanded_DF[f"x-angle{other_braces[1]}"]
        FX4 = expanded_DF[f"xbr{i}_FX"]  * np.sin(np.radians(xalpha1))
        FX5 = expanded_DF[f"xbr{other_braces[0]}_FX"]  * np.sin(np.radians(xalpha2))
        FX6 = expanded_DF[f"xbr{other_braces[1]}_FX"]  * np.sin(np.radians(xalpha3))

        X_tolerance_10 = 0.1 * FX4.abs()
        X_tolerance_50 = 0.5 * FX4.abs()
        
        x_fully_balanced = (
            ((FX4 - FX1).abs() <= X_tolerance_10) |
            ((FX4 - FX2).abs() <= X_tolerance_10) |
            ((FX4 - FX3).abs() <= X_tolerance_10) |
            ((FX4 - (FX1 + FX2)).abs() <= X_tolerance_10) |
            ((FX4 - (FX1 + FX3)).abs() <= X_tolerance_10) |
            ((FX4 - (FX2 + FX3)).abs() <= X_tolerance_10) |
            ((FX4 - (FX1 + FX2 + FX3)).abs() <= X_tolerance_10)
        )

        x_half_balanced = (
            ((FX4 - FX1).abs() <= X_tolerance_50) |
            ((FX4 - FX2).abs() <= X_tolerance_50) |
            ((FX4 - FX3).abs() <= X_tolerance_50) |
            ((FX4 - (FX1 + FX2)).abs() <= X_tolerance_50) |
            ((FX4 - (FX1 + FX3)).abs() <= X_tolerance_50) |
            ((FX4 - (FX2 + FX3)).abs() <= X_tolerance_50) |
            ((FX4 - (FX1 + FX2 + FX3)).abs() <= X_tolerance_50)
        )

        # Assign percentages for X classifications
        expanded_DF.loc[x_fully_balanced & X_mask & brace_exists, f"brc{i}_X_FX_balanced"] = 100
        expanded_DF.loc[~x_fully_balanced & x_half_balanced & X_mask & brace_exists, f"brc{i}_X_FX_balanced"] = 50

    # Handle T-joint classifications & Over 100 percentages
        # Calculate the remaining percentage for T-classification
        total_percentage_k = expanded_DF[[f"brc{i}_KT_FX_balanced", f"brc{i}{other_braces[0]}_K_FX_balanced", f"brc{i}{other_braces[1]}_K_FX_balanced"]].sum(axis=1)
        T_percentage = (np.where(total_percentage_k < 100, 100 - total_percentage_k, 0))
        total_percentage = total_percentage_k + T_percentage
        mask_non_zero = total_percentage_k != 0
        # Ensure non-negative and assign to T-classification where there are non-zero values in the brace
        mask_valid_remaining = T_percentage > 0
        mask_normalize = (total_percentage != 100) & brace_exists & mask_non_zero
        # Place remaining percentage on T joints
        expanded_DF.loc[mask_valid_remaining & brace_exists, f"brc{i}_T_FX_balanced"] = T_percentage[mask_valid_remaining & brace_exists]
        # Normalize if different from 100
        expanded_DF.loc[mask_normalize, f"brc{i}_T_FX_balanced"] = expanded_DF.loc[mask_normalize, f"brc{i}_T_FX_balanced"] * 100 / total_percentage
        expanded_DF.loc[mask_normalize, f"brc{i}_KT_FX_balanced"] = expanded_DF.loc[mask_normalize, f"brc{i}_KT_FX_balanced"] * 100 / total_percentage
        expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[0]}_K_FX_balanced"] = expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[0]}_K_FX_balanced"] * 100 / total_percentage
        expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[1]}_K_FX_balanced"] = expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[1]}_K_FX_balanced"] * 100 / total_percentage
        
    classification_columns = []
    for i in range(1, 4):
        other_braces = [j for j in range(1, 4) if j != i]
        # Replace zeros with empty strings
        for classification in ["T", "X", "K", "KT"]:
            if classification == "K":
                classification_columns.append(f"brc{i}{other_braces[0]}_{classification}_FX_balanced")
                classification_columns.append(f"brc{i}{other_braces[1]}_{classification}_FX_balanced")
            else:
                classification_columns.append(f"brc{i}_{classification}_FX_balanced")
        
    # expanded_DF[classification_columns] = expanded_DF[classification_columns].replace(0, "")


    return expanded_DF



def bending_unbalanced(expanded_DF):
 # Initialize the new classification columns for each brace with zeros
    for i in range(1, 4):
        other_braces = [j for j in range(1, 4) if j != i]
        for prefix in ["OPB", "IPB"]: 
            expanded_DF[f"brc{i}_T_{prefix}_unbalanced"] = 0
            expanded_DF[f"brc{i}_KT_{prefix}_unbalanced"] = 0
            expanded_DF[f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"] = 0  # Fill with zeros initially
            expanded_DF[f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"] = 0  # Fill with zeros initially
            expanded_DF[f"brc{i}_X_{prefix}_balanced"] = 0

    KT_mask = expanded_DF["type"].str.contains("KT")
    K_mask = expanded_DF["type"].str.contains("K")
    X_mask  = expanded_DF["type"].str.contains("X")

    for i in range(1, 4):
        for prefix in ["OPB", "IPB"]:
            other_braces = [j for j in range(1, 4) if j != i]
            brace_exists = ~expanded_DF[f"brace{i}"].isna() & (expanded_DF[f"brace{i}"] != '')
           
            # Calculate MO values for each brace
            alpha1 = expanded_DF[f"angle{i}"]
            alpha2 = expanded_DF[f"angle{other_braces[0]}"]
            alpha3 = expanded_DF[f"angle{other_braces[1]}"]

            MO1 = expanded_DF[f"brc{i}_{prefix}"] * np.sin(np.radians(alpha1))
            MO2 = expanded_DF[f"brc{other_braces[0]}_{prefix}"] * np.sin(np.radians(alpha2))
            MO3 = expanded_DF[f"brc{other_braces[1]}_{prefix}"] * np.sin(np.radians(alpha3))

            # Calculate the tolerances (10% and 50% of the absolute value of the force in the current brace)
            MO_tolerance_10 = 0.1 * MO1.abs() 
            MO_tolerance_50 = 0.5 * MO1.abs()
            
            # Check if the KT Joint is unbalanced (100/50)
            MO_KT_unbalanced = ((MO1 + MO2 + MO3).abs() >= MO_tolerance_10) & ((MO1 + MO2 + MO3).abs() <= MO_tolerance_50)
            MO_KT_half_unbalanced = ((MO1 + MO2 + MO3).abs() >= MO_tolerance_50) 

            # Check if the KT Joint is unbalanced (100/50)
            MO_K_half_unbalanced2 = ((MO1 + MO2).abs() > MO_tolerance_10) & ((MO1 + MO2).abs() <= MO_tolerance_50) 
            MO_K_half_unbalanced3 = ((MO1 + MO3).abs() > MO_tolerance_10) & ((MO1 + MO3).abs() <= MO_tolerance_50) 
            MO_K_unbalanced2 = ((MO1 + MO2).abs() > MO_tolerance_50) 
            MO_K_unbalanced3 = ((MO1 + MO3).abs() > MO_tolerance_50) 
    
            # Assign percentages for KT classifications
            expanded_DF.loc[MO_KT_unbalanced & KT_mask & brace_exists, f"brc{i}_KT_{prefix}_unbalanced"] = 100
            expanded_DF.loc[MO_KT_half_unbalanced & KT_mask & brace_exists, f"brc{i}_KT_{prefix}_unbalanced"] = 50
            # Assign percentages for K classifications
            expanded_DF.loc[MO_K_unbalanced2 & ~MO_KT_unbalanced & K_mask & brace_exists, f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"] = 100
            expanded_DF.loc[MO_K_unbalanced3 & ~MO_KT_unbalanced & K_mask & brace_exists, f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"] = 100
            expanded_DF.loc[MO_K_half_unbalanced2 & ~MO_KT_unbalanced & K_mask & brace_exists, f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"] = 50
            expanded_DF.loc[MO_K_half_unbalanced3 & ~MO_KT_unbalanced & K_mask & brace_exists, f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"] = 50

         # Now check if MO4, MO5, or MO6 are within 10% of MO1, MO2, or MO3 or their combinations   
			# Mirrored Braces for x-brace evaluation
            alpha1 = expanded_DF[f"x-angle{i}"]
            MO4 = expanded_DF[f"xbr{i}_{prefix}"] * np.sin(np.radians(alpha1))

            X_tolerance_10 = 0.1 * MO4.abs()

            # Determine if absolute values should be used based on the prefix
            if prefix == "IPB":
                MO1 = MO1.abs()
                MO2 = MO2.abs()
                MO3 = MO3.abs()
                MO4 = MO4.abs()

            # Calculate the tolerance
            X_tolerance_10 = 0.1 * MO4.abs()
            X_tolerance_50 = 0.5 * MO4.abs()

            # Check if MO4 is within 10% of MO1, MO2, or MO3 or their combinations
            x_fully_balanced = (
                ((MO4 - MO1).abs() <= X_tolerance_10) |
                ((MO4 - MO2).abs() <= X_tolerance_10) |
                ((MO4 - MO3).abs() <= X_tolerance_10) |
                ((MO4 - (MO1 + MO2)).abs() <= X_tolerance_10) |
                ((MO4 - (MO1 + MO3)).abs() <= X_tolerance_10) |
                ((MO4 - (MO2 + MO3)).abs() <= X_tolerance_10) |
                ((MO4 - (MO1 + MO2 + MO3)).abs() <= X_tolerance_10)
            )

            x_half_balanced = (
                ((MO4 - MO1).abs() <= X_tolerance_50) |
                ((MO4 - MO2).abs() <= X_tolerance_50) |
                ((MO4 - MO3).abs() <= X_tolerance_50) |
                ((MO4 - (MO1 + MO2)).abs() <= X_tolerance_50) |
                ((MO4 - (MO1 + MO3)).abs() <= X_tolerance_50) |
                ((MO4 - (MO2 + MO3)).abs() <= X_tolerance_50) |
                ((MO4 - (MO1 + MO2 + MO3)).abs() <= X_tolerance_50)
            )
            
            expanded_DF.loc[x_fully_balanced & X_mask & brace_exists, f"brc{i}_X_{prefix}_balanced"] = 100
            expanded_DF.loc[x_half_balanced & X_mask & brace_exists, f"brc{i}_X_{prefix}_balanced"] = 50

# Handle T-joint classifications & Over 100 percentages
        for prefix in ["OPB", "IPB"]: 
            # Handle T-joint classifications & Over 100 percentages
            # Calculate the remaining percentage for T-classification
            total_percentage_k = expanded_DF[[f"brc{i}_KT_{prefix}_unbalanced", f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced", f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"]].sum(axis=1)
            T_percentage = np.where(total_percentage_k < 100, 100 - total_percentage_k, 0)
            total_percentage = total_percentage_k + T_percentage
            mask_non_zero = total_percentage_k != 0
            # Ensure non-negative and assign to T-classification where there are non-zero values in the brace
            mask_valid_remaining = T_percentage > 0
            mask_normalize = (total_percentage != 100) & brace_exists & mask_non_zero
            # Place remaining percentage on T joints
            expanded_DF.loc[mask_valid_remaining & brace_exists, f"brc{i}_T_{prefix}_unbalanced"] = T_percentage[mask_valid_remaining & brace_exists]
            # Normalize if different from 100
            expanded_DF.loc[mask_normalize, f"brc{i}_T_{prefix}_unbalanced"] = expanded_DF.loc[mask_normalize, f"brc{i}_T_{prefix}_unbalanced"] * 100 / total_percentage
            expanded_DF.loc[mask_normalize, f"brc{i}_KT_{prefix}_unbalanced"] = expanded_DF.loc[mask_normalize, f"brc{i}_KT_{prefix}_unbalanced"] * 100 / total_percentage
            expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"] = expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"] * 100 / total_percentage
            expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"] = expanded_DF.loc[mask_normalize, f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"] * 100 / total_percentage

    # Replace zeros with empty strings
    for i in range(1, 4):
        for prefix in ["OPB", "IPB"]: 
            other_braces = [j for j in range(1, 4) if j != i]
            # expanded_DF[f"brc{i}_T_{prefix}_unbalanced"] = expanded_DF[f"brc{i}_T_{prefix}_unbalanced"].replace(0, "")
            # expanded_DF[f"brc{i}_KT_{prefix}_unbalanced"] = expanded_DF[f"brc{i}_KT_{prefix}_unbalanced"].replace(0, "")
            # expanded_DF[f"brc{i}_X_{prefix}_balanced"] = expanded_DF[f"brc{i}_X_{prefix}_balanced"].replace(0, "")
            # expanded_DF[f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"] = expanded_DF[f"brc{i}{other_braces[0]}_K_{prefix}_unbalanced"].replace(0, "")
            # expanded_DF[f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"] = expanded_DF[f"brc{i}{other_braces[1]}_K_{prefix}_unbalanced"].replace(0, "")
    
    return expanded_DF

def one_force_only(expanded_DF):
 # Initialize the new classification columns for each brace with zeros
    for i in range(1, 4):
        expanded_DF[f"brc{i}_FX_one"] = 0  # Fill with zeros initially
        expanded_DF[f"brc{i}_OPB_one"] = 0  # Fill with zeros initially
        expanded_DF[f"brc{i}_IPB_one"] = 0  # Fill with zeros initially

    K_mask = expanded_DF["type"].str.contains("K")

    for i in range(1, 4):
        brace_exists = ~expanded_DF[f"brace{i}"].isna() & (expanded_DF[f"brace{i}"] != '')
        other_braces = [j for j in range(1, 4) if j != i]
        alpha1 = expanded_DF[f"angle{i}"]
        alpha2 = expanded_DF[f"angle{other_braces[0]}"]
        alpha3 = expanded_DF[f"angle{other_braces[1]}"]

        # Forces in braces for K-Joint evaluation
        other_braces = [j for j in range(1, 4) if j != i]
        FX1 = expanded_DF[f"brc{i}_FX"] * np.sin(np.radians(alpha1))
        FX2 = expanded_DF[f"brc{other_braces[0]}_FX"] * np.sin(np.radians(alpha2))
        FX3 = expanded_DF[f"brc{other_braces[1]}_FX"] * np.sin(np.radians(alpha3))

        OPB1 = expanded_DF[f"brc{i}_OPB"] * np.sin(np.radians(alpha1))
        OPB2 = expanded_DF[f"brc{other_braces[0]}_OPB"] * np.sin(np.radians(alpha2))
        OPB3 = expanded_DF[f"brc{other_braces[1]}_OPB"] * np.sin(np.radians(alpha3))

        # Calculate IPB values for each brace
        IPB1 = expanded_DF[f"brc{i}_IPB"] * np.sin(np.radians(alpha1))
        IPB2 = expanded_DF[f"brc{other_braces[0]}_IPB"] * np.sin(np.radians(alpha2))
        IPB3 = expanded_DF[f"brc{other_braces[1]}_IPB"] * np.sin(np.radians(alpha3))

        # Calculate the tolerances
        FX_tolerance_10 = 0.1 * FX1.abs() 
        OPB_tolerance_10 = 0.1 * OPB1.abs() 
        IPB_tolerance_10 = 0.1 * IPB1.abs() 
        
        FX_ONE = (FX2.abs() + FX3.abs() <= FX_tolerance_10)
        OPB_ONE = (OPB2.abs() + OPB3.abs() <= OPB_tolerance_10)
        IPB_ONE = (IPB2.abs() + IPB3.abs() <= IPB_tolerance_10)

        
        # Assign percentages for K and KT classifications
        expanded_DF.loc[FX_ONE & K_mask & brace_exists, f"brc{i}_FX_one"] = 100
        expanded_DF.loc[OPB_ONE & K_mask & brace_exists, f"brc{i}_OPB_one"] = 100
        expanded_DF.loc[IPB_ONE & K_mask & brace_exists, f"brc{i}_IPB_one"] = 100

    for i in range(1, 4):
    # Replace zeros with empty strings
        for prefix in ["FX", "OPB", "IPB"]:
            classification_columns = [f"brc{i}_{prefix}_one" for i in range(1, 4)]
            # expanded_DF[classification_columns] = expanded_DF[classification_columns].replace(0, "")
    
    return expanded_DF



def select_SCF(expanded_DF):
    # Initialize new columns only once
    KT_mask = expanded_DF["type"].str.contains("KT")
    K_mask = expanded_DF["type"].str.contains("K") & ~KT_mask
    T_mask = expanded_DF["type"] == "T"
    X_mask = expanded_DF["type"].str.contains("X")

    cols = [
        f"SCF{i}_{prefix}_{suffix}"
        for i in range(1, 4)
        for prefix in ("CHD", "BRC")
        for suffix in ("SDL_FX", "CRN_FX", "CRN_IPB", "SDL_OPB")
    ]

    # one operation, no extra DF, minimal overhead
    expanded_DF[cols] = np.float32(0.0)
    


    for i in range(1, 4):
        brace1_exists = ~expanded_DF[f"brace{i}"].isna() & (expanded_DF[f"brace{i}"] != '')
        FX_balanced_columns = {
            classification: f"brc{i}_{classification}_FX_balanced" for classification in ["X", "KT"]
        }

        # Initialize R-Reduction Columns
        expanded_DF[f"R{i}_FX"] = 0.0
        expanded_DF[f"R{i}_IPB"] = 0.0
        expanded_DF[f"R{i}_OPB"] = 0.0

        # Define masks
        mask_T_FX = ~expanded_DF[f"brc{i}_T_FX_balanced"].isna() & (expanded_DF[f"brc{i}_T_FX_balanced"] != '') & (expanded_DF[f"brc{i}_T_FX_balanced"] != 0) &  brace1_exists
        mask_X_FX_balanced = ~expanded_DF[FX_balanced_columns["X"]].isna() & (expanded_DF[FX_balanced_columns["X"]] != '') & (expanded_DF[FX_balanced_columns["X"]] != 0) &  brace1_exists
        mask_KT_FX_balanced = ~expanded_DF[FX_balanced_columns["KT"]].isna() & (expanded_DF[FX_balanced_columns["KT"]] != '') & (expanded_DF[FX_balanced_columns["KT"]] != 0) &  brace1_exists

        mask_T_IPB = ~expanded_DF[f"brc{i}_T_IPB_unbalanced"].isna() & (expanded_DF[f"brc{i}_T_IPB_unbalanced"] != '') & (expanded_DF[f"brc{i}_T_IPB_unbalanced"] != 0) &  brace1_exists
        mask_T_OPB = ~expanded_DF[f"brc{i}_T_OPB_unbalanced"].isna() & (expanded_DF[f"brc{i}_T_OPB_unbalanced"] != '') & (expanded_DF[f"brc{i}_T_OPB_unbalanced"] != 0) &  brace1_exists
        mask_X_IPB_balanced = ~expanded_DF[f"brc{i}_X_IPB_balanced"].isna() & (expanded_DF[f"brc{i}_X_IPB_balanced"] != '') & (expanded_DF[f"brc{i}_X_IPB_balanced"] != 0) &  brace1_exists
        mask_X_OPB_balanced = ~expanded_DF[f"brc{i}_X_OPB_balanced"].isna() & (expanded_DF[f"brc{i}_X_OPB_balanced"] != '') & (expanded_DF[f"brc{i}_X_OPB_balanced"] != 0) &  brace1_exists
        
        mask_KT_IPB_unbalanced = ~expanded_DF[f"brc{i}_KT_IPB_unbalanced"].isna() & (expanded_DF[f"brc{i}_KT_IPB_unbalanced"] != '') & (expanded_DF[f"brc{i}_KT_IPB_unbalanced"] != 0) &  brace1_exists
        mask_KT_OPB_unbalanced = ~expanded_DF[f"brc{i}_KT_OPB_unbalanced"].isna() & (expanded_DF[f"brc{i}_KT_OPB_unbalanced"] != '') & (expanded_DF[f"brc{i}_KT_OPB_unbalanced"] != 0) &  brace1_exists

        mask_FX_ONE = ~expanded_DF[f"brc{i}_FX_one"].isna() & (expanded_DF[f"brc{i}_FX_one"] != '') & (expanded_DF[f"brc{i}_FX_one"] != 0) &  brace1_exists
        mask_IPB_ONE = ~expanded_DF[f"brc{i}_IPB_one"].isna() & (expanded_DF[f"brc{i}_IPB_one"] != '') & (expanded_DF[f"brc{i}_IPB_one"] != 0) &  brace1_exists
        mask_OPB_ONE = ~expanded_DF[f"brc{i}_OPB_one"].isna() & (expanded_DF[f"brc{i}_OPB_one"] != '') & (expanded_DF[f"brc{i}_OPB_one"] != 0) &  brace1_exists

        # KT FX BALANCED
        mask_KT_FX_balanced = mask_KT_FX_balanced & ~mask_FX_ONE
        if mask_KT_FX_balanced.any():
            multiplier = expanded_DF.loc[mask_KT_FX_balanced, FX_balanced_columns["KT"]] / 100
            expanded_DF.loc[mask_KT_FX_balanced, f"SCF{i}_CHD_SDL_FX"] += expanded_DF.loc[mask_KT_FX_balanced, f"KT_SCF{i}_CHD_SDL_FX_BALANCED"] * multiplier
            expanded_DF.loc[mask_KT_FX_balanced, f"SCF{i}_CHD_CRN_FX"] += expanded_DF.loc[mask_KT_FX_balanced, f"KT_SCF{i}_CHD_CRN_FX_BALANCED"] * multiplier
            expanded_DF.loc[mask_KT_FX_balanced, f"SCF{i}_BRC_SDL_FX"] += expanded_DF.loc[mask_KT_FX_balanced, f"KT_SCF{i}_BRC_SDL_FX_BALANCED"] * multiplier
            expanded_DF.loc[mask_KT_FX_balanced, f"SCF{i}_BRC_CRN_FX"] += expanded_DF.loc[mask_KT_FX_balanced, f"KT_SCF{i}_BRC_CRN_FX_BALANCED"] * multiplier
            
            max_values = np.zeros_like(expanded_DF.loc[mask_KT_FX_balanced, f"K_R{1}{2}_FX"])
            for j in [1, 2, 3]:
                if i != j:
                    max_values = np.maximum( max_values, expanded_DF.loc[mask_KT_FX_balanced, f"K_R{i}{j}_FX"] * multiplier)
            expanded_DF.loc[mask_KT_FX_balanced, f"R{i}_FX"] += max_values

        # KT IPB UNBALANCED  <<<< This must be unbalanced as all KT / K are unbalnced SCFs
        mask_KT_IPB_unbalanced = mask_KT_IPB_unbalanced & ~mask_IPB_ONE
        if (mask_KT_IPB_unbalanced).any():    
            multiplier = (expanded_DF.loc[mask_KT_IPB_unbalanced, f"brc{i}_KT_IPB_unbalanced"]) / 100
            expanded_DF.loc[mask_KT_IPB_unbalanced, f"SCF{i}_BRC_CRN_IPB"] += expanded_DF.loc[mask_KT_IPB_unbalanced, f"KT_SCF{i}_BRC_CRN_IPB_BALANCED"] * multiplier
            expanded_DF.loc[mask_KT_IPB_unbalanced, f"SCF{i}_CHD_CRN_IPB"] += expanded_DF.loc[mask_KT_IPB_unbalanced, f"KT_SCF{i}_CHD_CRN_IPB_BALANCED"] * multiplier

            max_values = np.zeros_like(expanded_DF.loc[mask_KT_IPB_unbalanced, f"K_R{1}{2}_IPB"])
            for j in [1, 2, 3]:
                if i != j:
                    max_values = np.maximum( max_values, expanded_DF.loc[mask_KT_IPB_unbalanced, f"K_R{i}{j}_IPB"] * multiplier)
            expanded_DF.loc[mask_KT_IPB_unbalanced, f"R{i}_IPB"] += max_values

        # KT OPB UNBALANCED
        mask_KT_OPB_unbalanced = mask_KT_OPB_unbalanced & ~mask_OPB_ONE
        if mask_KT_OPB_unbalanced.any():
            multiplier = (expanded_DF.loc[mask_KT_OPB_unbalanced, f"brc{i}_KT_OPB_unbalanced"]) / 100
            expanded_DF.loc[mask_KT_OPB_unbalanced, f"SCF{i}_BRC_SDL_OPB"] += expanded_DF.loc[mask_KT_OPB_unbalanced, f"KT_SCF{i}_BRC_SDL_OPB_UNBALANCED"] * multiplier  
            expanded_DF.loc[mask_KT_OPB_unbalanced, f"SCF{i}_CHD_SDL_OPB"] += expanded_DF.loc[mask_KT_OPB_unbalanced, f"KT_SCF{i}_CHD_SDL_OPB_UNBALANCED"] * multiplier

            max_values = np.zeros_like(expanded_DF.loc[mask_KT_OPB_unbalanced, f"K_R{1}{2}_OPB"])
            for j in [1, 2, 3]:
                if i != j:
                    max_values = np.maximum( max_values, expanded_DF.loc[mask_KT_OPB_unbalanced, f"K_R{i}{j}_OPB"] * multiplier)
            expanded_DF.loc[mask_KT_OPB_unbalanced, f"R{i}_OPB"] += max_values

        # KT FX ONE BRACE ONLY
        mask_FX_ONE_KT = mask_FX_ONE & KT_mask & ~mask_KT_FX_balanced
        if mask_FX_ONE_KT.any():
            expanded_DF.loc[mask_FX_ONE_KT, f"SCF{i}_CHD_SDL_FX"] = expanded_DF.loc[mask_FX_ONE_KT, f"KT_SCF{i}_CHD_SDL_FX_ONE"]
            expanded_DF.loc[mask_FX_ONE_KT, f"SCF{i}_CHD_CRN_FX"] = expanded_DF.loc[mask_FX_ONE_KT, f"KT_SCF{i}_CHD_CRN_FX_ONE"]
            expanded_DF.loc[mask_FX_ONE_KT, f"SCF{i}_BRC_SDL_FX"] = expanded_DF.loc[mask_FX_ONE_KT, f"KT_SCF{i}_BRC_SDL_FX_ONE"]
            expanded_DF.loc[mask_FX_ONE_KT, f"SCF{i}_BRC_CRN_FX"] = expanded_DF.loc[mask_FX_ONE_KT, f"KT_SCF{i}_BRC_CRN_FX_ONE"]

            for j in [1, 2, 3]:
                if i != j:
                    max_values = np.maximum( expanded_DF.loc[mask_FX_ONE_KT, f"R{i}_FX"], expanded_DF.loc[mask_FX_ONE_KT, f"K_R{i}{j}_FX"])
            expanded_DF.loc[mask_FX_ONE_KT, f"R{i}_FX"] = max_values

        # KT OPB ONE BRACE ONLY
        mask_OPB_ONE_KT = mask_OPB_ONE & KT_mask & ~mask_KT_OPB_unbalanced
        if mask_OPB_ONE_KT.any():
            expanded_DF.loc[mask_OPB_ONE_KT, f"SCF{i}_BRC_SDL_OPB"] = expanded_DF.loc[mask_OPB_ONE_KT, f"KT_SCF{i}_BRC_SDL_OPB_ONE"]
            expanded_DF.loc[mask_OPB_ONE_KT, f"SCF{i}_CHD_SDL_OPB"] = expanded_DF.loc[mask_OPB_ONE_KT, f"KT_SCF{i}_CHD_SDL_OPB_ONE"]

            for j in [1, 2, 3]:
                if i != j:
                    max_values = np.maximum( expanded_DF.loc[mask_OPB_ONE_KT, f"R{i}_OPB"], expanded_DF.loc[mask_OPB_ONE_KT, f"K_R{i}{j}_OPB"])
            expanded_DF.loc[mask_OPB_ONE_KT, f"R{i}_OPB"] = max_values

        #  <<<<<<<<<<<<<<       K JOINTS        >>>>>>>>>>>>>>>>>>>>>>>>>
        for j in range(1, 4):
            if i != j:
                brace2_exists = ~expanded_DF[f"brace{j}"].isna() & (expanded_DF[f"brace{j}"] != '')
                # K FX BALANCED
                mask_K_FX_balanced = ~expanded_DF[f"brc{i}{j}_K_FX_balanced"].isna() & (expanded_DF[f"brc{i}{j}_K_FX_balanced"] != '') & brace1_exists & brace2_exists & ~mask_FX_ONE
                if mask_K_FX_balanced.any():
                    multiplier = expanded_DF.loc[mask_K_FX_balanced, f"brc{i}{j}_K_FX_balanced"] / 100
                    expanded_DF.loc[mask_K_FX_balanced, f"SCF{i}_CHD_SDL_FX"] += expanded_DF.loc[mask_K_FX_balanced, f"K_SCF{i}{j}_CHD_SDL_FX_BALANCED"] * multiplier
                    expanded_DF.loc[mask_K_FX_balanced, f"SCF{i}_CHD_CRN_FX"] += expanded_DF.loc[mask_K_FX_balanced, f"K_SCF{i}{j}_CHD_CRN_FX_BALANCED"] * multiplier
                    expanded_DF.loc[mask_K_FX_balanced, f"SCF{i}_BRC_SDL_FX"] += expanded_DF.loc[mask_K_FX_balanced, f"K_SCF{i}{j}_BRC_SDL_FX_BALANCED"] * multiplier
                    expanded_DF.loc[mask_K_FX_balanced, f"SCF{i}_BRC_CRN_FX"] += expanded_DF.loc[mask_K_FX_balanced, f"K_SCF{i}{j}_BRC_CRN_FX_BALANCED"] * multiplier

                    expanded_DF.loc[mask_K_FX_balanced, f"R{i}_FX"] += expanded_DF.loc[mask_K_FX_balanced, f"K_R{i}{j}_FX"] * multiplier
                # K IPB UNBALANCED
                mask_K_IPB_unbalanced = ~expanded_DF[f"brc{i}{j}_K_IPB_unbalanced"].isna() & (expanded_DF[f"brc{i}{j}_K_IPB_unbalanced"] != '')  & (expanded_DF[f"brc{i}{j}_K_IPB_unbalanced"] != 0) & brace1_exists & brace2_exists  & ~mask_IPB_ONE
                if mask_K_IPB_unbalanced.any():
                    multiplier = (expanded_DF.loc[mask_K_IPB_unbalanced, f"brc{i}{j}_K_IPB_unbalanced"]) / 100
                    expanded_DF.loc[mask_K_IPB_unbalanced, f"SCF{i}_BRC_CRN_IPB"] += expanded_DF.loc[mask_K_IPB_unbalanced, f"K_SCF{i}{j}_BRC_CRN_IPB_UNBALANCED"] * multiplier
                    expanded_DF.loc[mask_K_IPB_unbalanced, f"SCF{i}_CHD_CRN_IPB"] += expanded_DF.loc[mask_K_IPB_unbalanced, f"K_SCF{i}{j}_CHD_CRN_IPB_UNBALANCED"] * multiplier

                    expanded_DF.loc[mask_K_IPB_unbalanced, f"R{i}_IPB"] += expanded_DF.loc[mask_K_IPB_unbalanced, f"K_R{i}{j}_IPB"] * multiplier
                # K OPB UNBALANCED
                mask_K_OPB_unbalanced = ~expanded_DF[f"brc{i}{j}_K_OPB_unbalanced"].isna() & (expanded_DF[f"brc{i}{j}_K_OPB_unbalanced"] != '') & (expanded_DF[f"brc{i}{j}_K_OPB_unbalanced"] != 0) & brace1_exists & brace2_exists & ~mask_OPB_ONE
                if mask_K_OPB_unbalanced.any():
                    multiplier = (expanded_DF.loc[mask_K_OPB_unbalanced & ~mask_OPB_ONE, f"brc{i}{j}_K_OPB_unbalanced"]) / 100
                    expanded_DF.loc[mask_K_OPB_unbalanced, f"SCF{i}_BRC_SDL_OPB"] += expanded_DF.loc[mask_K_OPB_unbalanced, f"K_SCF{i}{j}_BRC_SDL_OPB_UNBALANCED"] * multiplier
                    expanded_DF.loc[mask_K_OPB_unbalanced, f"SCF{i}_CHD_SDL_OPB"] += expanded_DF.loc[mask_K_OPB_unbalanced, f"K_SCF{i}{j}_CHD_SDL_OPB_UNBALANCED"] * multiplier

                    expanded_DF.loc[mask_K_OPB_unbalanced, f"R{i}_OPB"] += expanded_DF.loc[mask_K_OPB_unbalanced, f"K_R{i}{j}_OPB"] * multiplier
                # K FX ONE BRACE ONLY
                mask_FX_ONE_K = mask_FX_ONE & ~mask_K_FX_balanced & ~mask_KT_FX_balanced  & brace1_exists & brace2_exists
                if mask_FX_ONE_K.any(): 
                    expanded_DF.loc[mask_FX_ONE_K, f"SCF{i}_CHD_SDL_FX"] = expanded_DF.loc[mask_FX_ONE_K, f"K_SCF{i}{j}_CHD_SDL_FX_ONE"]
                    expanded_DF.loc[mask_FX_ONE_K, f"SCF{i}_CHD_CRN_FX"] = expanded_DF.loc[mask_FX_ONE_K, f"K_SCF{i}{j}_CHD_CRN_FX_ONE"]
                    expanded_DF.loc[mask_FX_ONE_K, f"SCF{i}_BRC_SDL_FX"] = expanded_DF.loc[mask_FX_ONE_K, f"K_SCF{i}{j}_BRC_SDL_FX_ONE"]
                    expanded_DF.loc[mask_FX_ONE_K, f"SCF{i}_BRC_CRN_FX"] = expanded_DF.loc[mask_FX_ONE_K, f"K_SCF{i}{j}_BRC_CRN_FX_ONE"]

                    expanded_DF.loc[mask_FX_ONE_K, f"R{i}_FX"] = expanded_DF.loc[mask_FX_ONE_K, f"K_R{i}{j}_FX"]

                # K OPB ONE BRACE ONLY
                # mask_empty = expanded_DF[f"SCF{i}_BRC_SDL_OPB"].isna() | (expanded_DF[f"SCF{i}_BRC_SDL_OPB"] == '') | (expanded_DF[f"SCF{i}_BRC_SDL_OPB"] == 0)
                mask_OPB_ONE_K = mask_OPB_ONE & ~mask_K_OPB_unbalanced & ~mask_KT_OPB_unbalanced & brace1_exists & brace2_exists  
                if mask_OPB_ONE_K.any():
                    expanded_DF.loc[mask_OPB_ONE_K, f"SCF{i}_BRC_SDL_OPB"] = expanded_DF.loc[mask_OPB_ONE_K, f"K_SCF{i}{j}_BRC_SDL_OPB_ONE"]
                    expanded_DF.loc[mask_OPB_ONE_K, f"SCF{i}_CHD_SDL_OPB"] = expanded_DF.loc[mask_OPB_ONE_K, f"K_SCF{i}{j}_CHD_SDL_OPB_ONE"]

                    expanded_DF.loc[mask_OPB_ONE_K, f"R{i}_OPB"] = expanded_DF.loc[mask_OPB_ONE_K, f"K_R{i}{j}_OPB"]
                
                # K IPB ONE BRACE ONLY
                # mask_empty = expanded_DF[f"SCF{i}_BRC_CRN_IPB"].isna() | (expanded_DF[f"SCF{i}_BRC_CRN_IPB"] == '') | (expanded_DF[f"SCF{i}_BRC_CRN_IPB"] == 0)
                mask_IPB_ONE_K = mask_IPB_ONE & ~mask_K_IPB_unbalanced & ~mask_KT_IPB_unbalanced & brace1_exists & brace2_exists   
                if mask_IPB_ONE_K.any():
                    expanded_DF.loc[mask_IPB_ONE_K, f"SCF{i}_BRC_CRN_IPB"] = expanded_DF.loc[mask_IPB_ONE_K, f"K_SCF{i}{j}_BRC_CRN_IPB_ONE"]
                    expanded_DF.loc[mask_IPB_ONE_K, f"SCF{i}_CHD_CRN_IPB"] = expanded_DF.loc[mask_IPB_ONE_K, f"K_SCF{i}{j}_CHD_CRN_IPB_ONE"]     

                    expanded_DF.loc[mask_IPB_ONE_K, f"R{i}_IPB"] = expanded_DF.loc[mask_IPB_ONE_K, f"K_R{i}{j}_IPB"]   
        
        #  <<<<<<<<<<<<<<       T JOINTS        >>>>>>>>>>>>>>>>>>>>>>>>>
        # T Axial
        mask_T_FX = mask_T_FX & ~mask_FX_ONE
        if mask_T_FX.any(): 
            multiplier = expanded_DF.loc[mask_T_FX & ~mask_FX_ONE, f"brc{i}_T_FX_balanced"] / 100
            expanded_DF.loc[mask_T_FX, f"SCF{i}_CHD_SDL_FX"] += expanded_DF.loc[mask_T_FX, f"T_SCF{i}_CHD_SDL_FX"] * multiplier
            expanded_DF.loc[mask_T_FX, f"SCF{i}_CHD_CRN_FX"] += expanded_DF.loc[mask_T_FX, f"T_SCF{i}_CHD_CRN_FX"] * multiplier
            expanded_DF.loc[mask_T_FX, f"SCF{i}_BRC_SDL_FX"] += expanded_DF.loc[mask_T_FX, f"T_SCF{i}_BRC_SDL_FX"] * multiplier
            expanded_DF.loc[mask_T_FX, f"SCF{i}_BRC_CRN_FX"] += expanded_DF.loc[mask_T_FX, f"T_SCF{i}_BRC_CRN_FX"] * multiplier

            expanded_DF.loc[mask_T_FX, f"R{i}_FX"] += expanded_DF.loc[mask_T_FX, f"T_R{i}_FX"] * multiplier
        # T IPB 
        mask_T_IPB = mask_T_IPB & ~mask_IPB_ONE
        if mask_T_IPB.any():
            multiplier = expanded_DF.loc[mask_T_IPB, f"brc{i}_T_IPB_unbalanced"] / 100
            expanded_DF.loc[mask_T_IPB, f"SCF{i}_BRC_CRN_IPB"] += expanded_DF.loc[mask_T_IPB, f"T_SCF{i}_BRC_CRN_IPB"] * multiplier 
            expanded_DF.loc[mask_T_IPB, f"SCF{i}_CHD_CRN_IPB"] += expanded_DF.loc[mask_T_IPB, f"T_SCF{i}_CHD_CRN_IPB"] * multiplier 

            expanded_DF.loc[mask_T_IPB, f"R{i}_IPB"] += expanded_DF.loc[mask_T_IPB, f"T_R{i}_IPB"] * multiplier

        # T OPB
        mask_T_OPB = mask_T_OPB & ~mask_OPB_ONE
        if mask_T_OPB.any():
            multiplier = (expanded_DF.loc[mask_T_OPB & ~mask_OPB_ONE, f"brc{i}_T_OPB_unbalanced"]) / 100
            expanded_DF.loc[mask_T_OPB, f"SCF{i}_BRC_SDL_OPB"] += expanded_DF.loc[mask_T_OPB, f"T_SCF{i}_BRC_SDL_OPB"] * multiplier 
            expanded_DF.loc[mask_T_OPB, f"SCF{i}_CHD_SDL_OPB"] += expanded_DF.loc[mask_T_OPB, f"T_SCF{i}_CHD_SDL_OPB"] * multiplier 

            expanded_DF.loc[mask_T_OPB, f"R{i}_OPB"] += expanded_DF.loc[mask_T_OPB, f"T_R{i}_OPB"] * multiplier
        
        #  <<<<<<<<<<<<<<       X JOINTS        >>>>>>>>>>>>>>>>>>>>>>>>>
        # X Axial
        if mask_X_FX_balanced.any():
            prefixes = ["CHD_SDL", "CHD_CRN", "BRC_SDL", "BRC_CRN"]
            for prefix in prefixes:
                xSCF_multiplier = (expanded_DF.loc[mask_X_FX_balanced, f"brc{i}_X_FX_balanced"]) / 100
                SCF_multiplier = 1 - xSCF_multiplier
                expanded_DF.loc[mask_X_FX_balanced, f"SCF{i}_{prefix}_FX"] = expanded_DF.loc[mask_X_FX_balanced, f"X_SCF{i}_{prefix}_FX_BALANCED"] * xSCF_multiplier + expanded_DF.loc[mask_X_FX_balanced, f"SCF{i}_{prefix}_FX"] * SCF_multiplier

            expanded_DF.loc[mask_X_FX_balanced, f"R{i}_FX"] = expanded_DF.loc[mask_X_FX_balanced, f"X_R{i}_FX"] * xSCF_multiplier + expanded_DF.loc[mask_X_FX_balanced, f"R{i}_FX"] * SCF_multiplier

        # X IPB Balanced
        if mask_X_IPB_balanced.any():
            prefixes = ["CHD_CRN", "BRC_CRN"]
            for prefix in prefixes:
                xSCF_multiplier = (expanded_DF.loc[mask_X_IPB_balanced, f"brc{i}_X_IPB_balanced"]) / 100
                SCF_multiplier = 1 - xSCF_multiplier
                expanded_DF.loc[mask_X_IPB_balanced, f"SCF{i}_{prefix}_IPB"] = expanded_DF.loc[mask_X_IPB_balanced, f"X_SCF{i}_{prefix}_IPB_BALANCED"] * xSCF_multiplier + expanded_DF.loc[mask_X_IPB_balanced, f"SCF{i}_{prefix}_IPB"] * SCF_multiplier 

            expanded_DF.loc[mask_X_IPB_balanced, f"R{i}_IPB"] = expanded_DF.loc[mask_X_IPB_balanced, f"X_R{i}_IPB"] * xSCF_multiplier + expanded_DF.loc[mask_X_IPB_balanced, f"R{i}_IPB"] * SCF_multiplier 

        # X OPB Balanced
        if mask_X_OPB_balanced.any():
            prefixes = ["CHD_SDL", "BRC_SDL"]
            for prefix in prefixes:
                xSCF_multiplier = (expanded_DF.loc[mask_X_OPB_balanced, f"brc{i}_X_OPB_balanced"]) / 100
                SCF_multiplier = 1 - xSCF_multiplier
                expanded_DF.loc[mask_X_OPB_balanced, f"SCF{i}_{prefix}_OPB"] = expanded_DF.loc[mask_X_OPB_balanced, f"X_SCF{i}_{prefix}_OPB_BALANCED"] * xSCF_multiplier + expanded_DF.loc[mask_X_OPB_balanced, f"SCF{i}_{prefix}_OPB"] * SCF_multiplier

            expanded_DF.loc[mask_X_OPB_balanced, f"R{i}_OPB"] = expanded_DF.loc[mask_X_OPB_balanced, f"X_R{i}_OPB"] * xSCF_multiplier + expanded_DF.loc[mask_X_OPB_balanced, f"R{i}_OPB"] * SCF_multiplier

    columns_to_print = ["joint", "type", "chord1", "brace1", "brace2", "brace3", "loads"]



    # for i in range(1, 4):
    #     for prefix in ["CHD", "BRC"]:
    #         for suffix in ["SDL_FX", "CRN_FX", "CRN_IPB", "SDL_OPB"]:
    #             columns_to_print.append(f"SCF{i}_{prefix}_{suffix}")
    # expanded_DF[columns_to_print] = expanded_DF[columns_to_print].replace(0, "")
    # expanded_DF[columns_to_print].to_csv(f'test.csv', index=False)
    return expanded_DF

def calc_hotspot_stresses_old(df):
    direction = df['direction'].iloc[0]
    # Chord Properties
    angles = [i * ((2*np.pi) / 8) for i in range(8)]
    chd_OD, chd_THK = (df[f"chd_OD1"], df[f"chd_THK1"])
    chd_A = (np.pi/4)*((chd_OD**2) - ((chd_OD - 2*chd_THK)**2))
    chd_I = (np.pi/64)*((chd_OD**4) - ((chd_OD - 2*chd_THK)**4))
    chd_W = 2*chd_I/chd_OD   
    # columns_to_print = ["joint", "type", "chord1", "brace1", "brace2", "brace3", "loads", "wave_height", "wave_period"]

    # Cycle through each brace
    for i in range(1,4):
        brace_exists = ~df[f"brace{i}"].isna() & (df[f"brace{i}"] != '')
        # Brace Properties
        brc_OD, brc_THK, brc_FX, brc_MIPB, brc_MOPB = (df[f"brc_OD{i}"], df[f"brc_THK{i}"], df[f"brc{i}_FX"], df[f"brc{i}_IPB"], df[f"brc{i}_OPB"])
        brc_A = (np.pi/4)*((brc_OD**2) - ((brc_OD - 2*brc_THK)**2))
        brc_I = (np.pi/64)*((brc_OD**4) - ((brc_OD - 2*brc_THK)**4))
        brc_W = 2*brc_I/brc_OD
        
        for prefix in ["BRC", "CHD"]:   
            for h in range(8):
                SCF_AXIAL = df[f"SCF{i}_{prefix}_CRN_FX"] * np.cos(angles[h]) + df[f"SCF{i}_{prefix}_SDL_FX"] * np.sin(angles[h])
                SCF_IPB = df[f"SCF{i}_{prefix}_CRN_IPB"] * np.cos(angles[h])
                SCF_OPB = df[f"SCF{i}_{prefix}_SDL_OPB"] * np.sin(angles[h])    

                # columns_to_print.append(f"{prefix}{i}_HS{h+1}") 
                df[f"{prefix}{i}_HS{h+1}"] = (SCF_AXIAL * brc_FX*1000/brc_A) + (SCF_IPB * brc_MIPB*1000000/brc_W) + (SCF_OPB * brc_MOPB*1000000/brc_W) 
            
    df.to_csv(f'Stresses_{direction}.csv', index=False)
    return df 

def calc_hotspot_stresses(df, side, IRS_calc = False):
    df = df.copy(deep=True)
    direction = df['direction'].iloc[0]
    # Chord Properties
    angles = [i * ((2*np.pi) / 8) for i in range(8)]
    chd_OD, chd_THK = (df[f"chd_OD1"], df[f"chd_THK1"])
    chd_A = (np.pi/4)*((chd_OD**2) - ((chd_OD - 2*chd_THK)**2))
    chd_I = (np.pi/64)*((chd_OD**4) - ((chd_OD - 2*chd_THK)**4))
    chd_W = 2*chd_I/chd_OD   
    # columns_to_print = ["joint", "type", "chord1", "brace1", "brace2", "brace3", "loads", "wave_height", "wave_period"]

    # Cycle through each brace
    for i in range(1,4):
        brace_exists = ~df[f"brace{i}"].isna() & (df[f"brace{i}"] != '')
        # Brace Properties
        brc_OD, brc_THK, brc_FX, brc_MIPB, brc_MOPB = (df[f"brc_OD{i}"], df[f"brc_THK{i}"], df[f"brc{i}_FX"], df[f"brc{i}_IPB"], df[f"brc{i}_OPB"])
        brc_A = (np.pi/4)*((brc_OD**2) - ((brc_OD - 2*brc_THK)**2))
        brc_I = (np.pi/64)*((brc_OD**4) - ((brc_OD - 2*brc_THK)**4))
        brc_W = 2*brc_I/brc_OD

        min_SCF = {}
        if IRS_calc == False: 
            min_SCF["min_BRC_CRN_FX_SCF"] = 1.5
            min_SCF["min_BRC_SDL_FX_SCF"] = 1.5
            min_SCF["min_BRC_CRN_IPB_SCF"] = 1.5
            min_SCF["min_BRC_SDL_OPB_SCF"] = 1.5
            min_SCF["min_CHD_CRN_FX_SCF"] = 1.5
            min_SCF["min_CHD_SDL_FX_SCF"] = 1.5
            min_SCF["min_CHD_CRN_IPB_SCF"] = 1.5
            min_SCF["min_CHD_SDL_OPB_SCF"] = 1.5
        else: # IRS
            min_SCF["min_BRC_CRN_FX_SCF"] = 2.5
            min_SCF["min_BRC_SDL_FX_SCF"] = 2.5
            min_SCF["min_BRC_CRN_IPB_SCF"] = 1.5
            min_SCF["min_BRC_SDL_OPB_SCF"] = 2.5
            min_SCF["min_CHD_CRN_FX_SCF"] = 2.5
            min_SCF["min_CHD_SDL_FX_SCF"] = 1.5
            min_SCF["min_CHD_CRN_IPB_SCF"] = 1.5
            min_SCF["min_CHD_SDL_OPB_SCF"] = 1.5

        for prefix in ["BRC", "CHD"]:   
            for h in range(8):
                # toe SCFs
                SCF_FX = df[f"SCF{i}_{prefix}_CRN_FX"] * np.cos(angles[h]) + df[f"SCF{i}_{prefix}_SDL_FX"] * np.sin(angles[h])
                SCF_IPB = df[f"SCF{i}_{prefix}_CRN_IPB"] * np.cos(angles[h])
                SCF_OPB = df[f"SCF{i}_{prefix}_SDL_OPB"] * np.sin(angles[h])  

                min_SCF_FX = min_SCF[f"min_{prefix}_CRN_FX_SCF"] * np.cos(angles[h]) + min_SCF[f"min_{prefix}_SDL_FX_SCF"]* np.sin(angles[h])
                min_SCF_IPB = min_SCF[f"min_{prefix}_CRN_IPB_SCF"] * np.cos(angles[h])
                min_SCF_OPB = min_SCF[f"min_{prefix}_SDL_OPB_SCF"] * np.sin(angles[h])  

                SCF_FX = np.maximum(SCF_FX, min_SCF_FX)
                SCF_IPB = np.maximum(SCF_IPB, min_SCF_IPB)
                SCF_OPB = np.maximum(SCF_OPB, min_SCF_OPB)

                if side == "root":
                    SCF_FX = df[f"R{i}_FX"] * SCF_FX
                    SCF_IPB = df[f"R{i}_IPB"] * SCF_IPB
                    SCF_OPB = df[f"R{i}_OPB"] * SCF_OPB
                    SCF_FX = np.maximum(SCF_FX, 1.0)
                    SCF_IPB = np.maximum(SCF_IPB, 1.0)
                    SCF_OPB = np.maximum(SCF_OPB, 1.0)

                # Create a mask for valid brace geometry
                valid_geometry = (brc_A > 0) & (brc_W > 0) & brace_exists

                # Compute stresses safely
                stress_result = np.full(len(df), np.nan)  # default to NaN
                stress_result[valid_geometry] = (
                    SCF_FX[valid_geometry] * brc_FX[valid_geometry] * 1000 / brc_A[valid_geometry]
                    + SCF_IPB[valid_geometry] * brc_MIPB[valid_geometry] * 1e6 / brc_W[valid_geometry]
                    + SCF_OPB[valid_geometry] * brc_MOPB[valid_geometry] * 1e6 / brc_W[valid_geometry]
                )

                # Store result
                df[f"{prefix}{i}_HS{h+1}"] = stress_result
                # columns_to_print.append(f"{prefix}{i}_HS{h+1}") 
                 
    # df.to_csv(f'Stresses_{direction}.csv', index=False)
    return df

def calc_hotspot_stresses_old2(df, side, IRS_calc = False):
    df = df.copy(deep=True)
    direction = df['direction'].iloc[0]
    # Chord Properties
    angles = [i * ((2*np.pi) / 8) for i in range(8)]
    chd_OD, chd_THK = (df[f"chd_OD1"], df[f"chd_THK1"])
    chd_A = (np.pi/4)*((chd_OD**2) - ((chd_OD - 2*chd_THK)**2))
    chd_I = (np.pi/64)*((chd_OD**4) - ((chd_OD - 2*chd_THK)**4))
    chd_W = 2*chd_I/chd_OD   
    # columns_to_print = ["joint", "type", "chord1", "brace1", "brace2", "brace3", "loads", "wave_height", "wave_period"]

    # Cycle through each brace
    for i in range(1,4):
        brace_exists = ~df[f"brace{i}"].isna() & (df[f"brace{i}"] != '')
        # Brace Properties
        brc_OD, brc_THK, brc_FX, brc_MIPB, brc_MOPB = (df[f"brc_OD{i}"], df[f"brc_THK{i}"], df[f"brc{i}_FX"], df[f"brc{i}_IPB"], df[f"brc{i}_OPB"])
        brc_A = (np.pi/4)*((brc_OD**2) - ((brc_OD - 2*brc_THK)**2))
        brc_I = (np.pi/64)*((brc_OD**4) - ((brc_OD - 2*brc_THK)**4))
        brc_W = 2*brc_I/brc_OD

        min_SCF = {}
        if IRS_calc == False: 
            min_SCF["min_BRC_CRN_FX_SCF"] = 1.5
            min_SCF["min_BRC_SDL_FX_SCF"] = 1.5
            min_SCF["min_BRC_CRN_IPB_SCF"] = 1.5
            min_SCF["min_BRC_SDL_OPB_SCF"] = 1.5
            min_SCF["min_CHD_CRN_FX_SCF"] = 1.5
            min_SCF["min_CHD_SDL_FX_SCF"] = 1.5
            min_SCF["min_CHD_CRN_IPB_SCF"] = 1.5
            min_SCF["min_CHD_SDL_OPB_SCF"] = 1.5
        else: # IRS
            min_SCF["min_BRC_CRN_FX_SCF"] = 2.5
            min_SCF["min_BRC_SDL_FX_SCF"] = 2.5
            min_SCF["min_BRC_CRN_IPB_SCF"] = 1.5
            min_SCF["min_BRC_SDL_OPB_SCF"] = 2.5
            min_SCF["min_CHD_CRN_FX_SCF"] = 2.5
            min_SCF["min_CHD_SDL_FX_SCF"] = 1.5
            min_SCF["min_CHD_CRN_IPB_SCF"] = 1.5
            min_SCF["min_CHD_SDL_OPB_SCF"] = 1.5

        for prefix in ["BRC", "CHD"]:   
            for h in range(8):
                # toe SCFs
                SCF_FX = df[f"SCF{i}_{prefix}_CRN_FX"] * np.cos(angles[h]) + df[f"SCF{i}_{prefix}_SDL_FX"] * np.sin(angles[h])
                SCF_IPB = df[f"SCF{i}_{prefix}_CRN_IPB"] * np.cos(angles[h])
                SCF_OPB = df[f"SCF{i}_{prefix}_SDL_OPB"] * np.sin(angles[h])  

                min_SCF_FX = min_SCF[f"min_{prefix}_CRN_FX_SCF"] * np.cos(angles[h]) + min_SCF[f"min_{prefix}_SDL_FX_SCF"]* np.sin(angles[h])
                min_SCF_IPB = min_SCF[f"min_{prefix}_CRN_IPB_SCF"] * np.cos(angles[h])
                min_SCF_OPB = min_SCF[f"min_{prefix}_SDL_OPB_SCF"] * np.sin(angles[h])  

                SCF_FX = np.maximum(SCF_FX, min_SCF_FX)
                SCF_IPB = np.maximum(SCF_IPB, min_SCF_IPB)
                SCF_OPB = np.maximum(SCF_OPB, min_SCF_OPB)

                if side == "root":
                    SCF_FX = df[f"R{i}_FX"] * SCF_FX
                    SCF_IPB = df[f"R{i}_IPB"] * SCF_IPB
                    SCF_OPB = df[f"R{i}_OPB"] * SCF_OPB
                    SCF_FX = np.maximum(SCF_FX, 1.0)
                    SCF_IPB = np.maximum(SCF_IPB, 1.0)
                    SCF_OPB = np.maximum(SCF_OPB, 1.0)

                # columns_to_print.append(f"{prefix}{i}_HS{h+1}") 
                df[f"{prefix}{i}_HS{h+1}"] = (SCF_FX * brc_FX*1000/brc_A) + (SCF_IPB * brc_MIPB*1000000/brc_W) + (SCF_OPB * brc_MOPB*1000000/brc_W) 
    
    df.to_csv(f'Stresses_{direction}.csv', index=False)
    return df

def calc_inline_hotspot_stresses(df, inline_type):
    angles = [i * ((2*np.pi) / 8) for i in range(8)]
    if inline_type == "inline":
        target_members = ["memb1", "memb2"]
    else:
        target_members = ["cone", "tub"]
    # Cycle through each member
    for member in target_members:
        # Member Properties
        if inline_type == "inline":
            mem_OD = df[f"{member}_OD"]
        else:
            if ("cone_L" in df.columns) & (member == "cone"):
                mem_OD = np.where(df['joint'] == df['cone'].str.slice(0, 4), df['cone_L'], df['cone_S'])
                df["cone_L"] = mem_OD
                df.rename(columns={"cone_L": "cone_OD"}, inplace=True)
                df.drop(columns=["cone_S"], inplace=True)
            else:
                mem_OD = df[f"{member}_OD"]


        mem_OD, mem_THK, mem_FX, mem_MIPB, mem_MOPB = (mem_OD, df[f"{member}_THK"], df[f"{member}_FX"], df[f"{member}_IPB"], df[f"{member}_OPB"])
        mem_A = (np.pi/4)*((mem_OD**2) - ((mem_OD - 2*mem_THK)**2))
        mem_I = (np.pi/64)*((mem_OD**4) - ((mem_OD - 2*mem_THK)**4))
        mem_W = 2*mem_I/mem_OD
        sigma_FX = mem_FX*1000/mem_A
        sigma_IPB = mem_MIPB*1000000/mem_W
        sigma_OPB = mem_MOPB*1000000/mem_W 
        for h in range(8):
            df[f"{member}_HS{h+1}"] = sigma_FX + (np.cos(angles[h]) * sigma_IPB) + (np.sin(angles[h])  * sigma_OPB) 

    return df


def transfer_function_simpler(df):
    # Define the target columns for grouping
    joint_columns = ["joint", "type", "chord1", "brace1", "brace2", "brace3"]
    wave_columns = ["wave_height", "wave_period"]
    
    # Get the direction (assuming it is consistent within the DataFrame)
    direction = df['direction'].iloc[0]

    # Create the list of columns in a compact way
    hs_columns = [f"{prefix}{i}_HS{j}" for prefix in ["BRC", "CHD"] for i in range(1, 4) for j in range(1, 9)]

    # Group by both joint and wave columns together to minimize the number of groupings
    grouped = df.groupby(joint_columns + wave_columns, dropna=False)

    results = []

    for (joint, joint_type, chord1, brace1, brace2, brace3, wave_height, wave_period), group_data in grouped:
        group_data.to_csv(f'Stresses_{direction}.csv', index=False)
        
        # Calculate the maximum and minimum values along each column for the subgroup
        max_values = group_data[hs_columns].max(axis=0)
        min_values = group_data[hs_columns].min(axis=0)

        # Calculate the stress ranges and normalize by wave_height
        ranges = (max_values - min_values) / wave_height

        # Prepare additional metadata for this row as a Series
        metadata = pd.Series({
            'joint': joint,
            'type': joint_type,
            'chord1': chord1,
            'brace1': brace1,
            'brace2': brace2,
            'brace3': brace3,
            'wave_height': wave_height,
            'wave_period': wave_period
        })

        # Concatenate metadata with ranges
        result_entry = pd.concat([metadata, ranges])

        # Append result_entry to results
        results.append(result_entry)
        

    # Convert the results (list of Series) to a DataFrame
    final_result_df = pd.DataFrame(results)

    # Save the final DataFrame to CSV
    final_result_df.to_csv(f'TF_{direction}.csv', index=False)

    return final_result_df

def SR_generation(df):
    # Define the target columns for sorting
    sort_columns = ["joint", "type", "chord1", "brace1", "brace2", "brace3"]
    wave_columns = ["direction", "wave_height", "wave_period"]
    OD_columns = ["chd_OD1", "brc_OD1", "brc_OD2", "brc_OD3"]
    thickness_columns = ["chd_THK1", "brc_THK1", "brc_THK2", "brc_THK3"]
    elevation_columns = ["elevation1", "elevation2", "elevation3"]
    direction = df['direction'].iloc[0]

    n_crest = df.groupby(sort_columns + wave_columns, dropna=False).size().iloc[0]
    # print(n_crest)
    hs_columns = [f"{prefix}{i}_HS{j}" for prefix in ["BRC", "CHD"] for i in range(1, 4) for j in range(1, 9)]
    df[hs_columns] = df[hs_columns].replace(["", "nan"], 0.0).fillna(0.0)
    # Sort the DataFrame by the specified columns to ensure correct grouping
    df_sorted = df.sort_values(by=sort_columns, ignore_index=True)

    # df_sorted.iloc[:36].to_csv("test.csv", index=False)

    # Check the length of the DataFrame is a multiple of n_crest
    if len(df_sorted) % n_crest != 0:
        raise ValueError("The number of rows in the DataFrame is not a multiple of n_crest.")

    # Reshape the DataFrame into (N, n_crest, M) where N is the number of groups, n_crest is the number of rows per group,
    # and M is the number of columns [see "transfer_function_generation_PD_version" for a clearer understanding of the function]
    n_chunks = len(df_sorted) // n_crest

    # Calculate the max and min over every n_crest rows for the hs_columns
    # Reshape is a "pandas groupby" equivalent in numpy. It splits the original 2D array in n_chuncks creating thus a 3D of size n_chuncks. Each sub-array has a size of n_crest
    # The above is much much quicker than pandas groupby
    # Reshape the hs_columns for max and min calculations
    # max_values and min_values should result in (n_chunks, n_columns)
    # here axis = 1 refers to the second dimension in the 3D dimension meaning max column of each chunch

    max_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).max(axis=1) 
    min_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).min(axis=1)

    # # Extract the correct wave_height values for each group (one per chunk)
    # # This extracts the first wave_height from each chunk, ensuring it's the same shape as max_values/min_values
    # wave_heights = df_sorted["wave_height"].values[::n_crest]

    # Ensure the wave_heights and the range calculations are compatible
    # wave_heights should be reshaped to (n_chunks, 1) to allow broadcasting with max_values and min_values
    ranges = (max_values - min_values) 
    print(max_values[:5], min_values[:5], ranges[:5])
    # This Creates a dataframe skeleton with every n_crest'th row. In this manner it will have the same size as ranges
    result_df = df_sorted.iloc[::n_crest][sort_columns + wave_columns + OD_columns + thickness_columns + elevation_columns].copy().reset_index(drop=True)
    
    # Add the calculated ranges to the result DataFrame
    result_ranges_df = pd.DataFrame(ranges, columns=hs_columns)
    final_result_df = pd.concat([result_df, result_ranges_df], axis=1)

    # final_result_df.to_csv(f'Ranges{direction}.csv', index=False)
    return final_result_df


def SR_inline_generation(df, inline_type):
    # Define the target columns for sorting
    if inline_type == "inline":
        sort_columns = ["joint", "memb1", "memb2"]
        geo_columns = ["memb1_OD", "memb1_THK", "memb2_OD", "memb2_THK"]
        hs_columns = [f"{prefix}_HS{j}" for prefix in ["memb1", "memb2"] for j in range(1, 9)]
        scf_columns = ["SCFtoe", "SCFroot"]
    else:
        sort_columns = ["joint", "cone", "tubular"]       
        geo_columns = ["cone_OD", "cone_THK", "tub_OD", "tub_THK"]
        hs_columns = [f"{prefix}_HS{j}" for prefix in ["cone", "tub"]  for j in range(1, 9)]
        scf_columns = ["SCFcone", "SCFtubular"]

    wave_columns = ["direction", "wave_height", "wave_period"]
    elevation_columns = ["Elevation"]
    n_crest = df.groupby(sort_columns + wave_columns, dropna=False).size().iloc[0] 

    # Sort the DataFrame by the specified columns to ensure correct grouping
    df_sorted = df.sort_values(by=sort_columns, ignore_index=True)

    # Check the length of the DataFrame is a multiple of n_crest
    if len(df_sorted) % n_crest != 0:
        raise ValueError("The number of rows in the DataFrame is not a multiple of n_crest.")

    # Reshape the DataFrame into (N, n_crest, M) where N is the number of groups, n_crest is the number of rows per group,
    # and M is the number of columns [see "transfer_function_generation_PD_version" for a clearer understanding of the function]
    n_chunks = len(df_sorted) // n_crest

    # Calculate the max and min over every n_crest rows for the hs_columns
    # Reshape is a "pandas groupby" equivalent in numpy. It splits the original 2D array in n_chuncks creating thus a 3D of size n_chuncks. Each sub-array has a size of n_crest
    # The above is much much quicker than pandas groupby
    # Reshape the hs_columns for max and min calculations
    # max_values and min_values should result in (n_chunks, n_columns)
    # here axis = 1 refers to the second dimension in the 3D dimension meaning max column of each chunch
    max_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).max(axis=1) 
    min_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).min(axis=1)

    # # Extract the correct wave_height values for each group (one per chunk)
    # # This extracts the first wave_height from each chunk, ensuring it's the same shape as max_values/min_values
    # wave_heights = df_sorted["wave_height"].values[::n_crest]


    # Ensure the wave_heights and the range calculations are compatible
    # wave_heights should be reshaped to (n_chunks, 1) to allow broadcasting with max_values and min_values
    if inline_type == "inline":
        SCFtoes = df_sorted[scf_columns[0]].values[::n_crest]
        SCFroots = df_sorted[scf_columns[1]].values[::n_crest]
        toe_ranges = SCFtoes.reshape(-1, 1) * (max_values - min_values) 
        root_ranges = SCFroots.reshape(-1, 1) * (max_values - min_values) 
        TFs = [toe_ranges, root_ranges]
    else: 
        SCFcones = df_sorted[scf_columns[0]].values[::n_crest]
        SCFtubs = df_sorted[scf_columns[1]].values[::n_crest]
        # Reshape SCFcones and SCFtubs for calculations
        cone_ranges = SCFcones.reshape(-1, 1) * (max_values[:, :8] - min_values[:, :8]) 
        tub_ranges = SCFtubs.reshape(-1, 1) * (max_values[:, 8:] - min_values[:, 8:]) 
        combined_ranges = np.concatenate([cone_ranges, tub_ranges], axis=1) 

    # This Creates a dataframe skeleton with every n_crest'th row. In this manner it will have the same size as ranges
    result_df = df_sorted.iloc[::n_crest][sort_columns + wave_columns + geo_columns + elevation_columns].copy().reset_index(drop=True)
    
    # Add the calculated ranges to the result DataFrame
    results = []
    for i in [0,1]:
        if inline_type == "inline":
            result_ranges_df = pd.DataFrame(TFs[i], columns=hs_columns)  
            result_ranges_df[scf_columns[0]] = SCFtoes
            result_ranges_df[scf_columns[1]] = SCFroots   
        else:
            result_ranges_df = pd.DataFrame(combined_ranges, columns=hs_columns)
            result_ranges_df[scf_columns[0]] = SCFcones
            result_ranges_df[scf_columns[1]] = SCFtubs   

        result_ranges_df = pd.concat([result_df, result_ranges_df], axis=1)
        results.append(result_ranges_df.copy())
    
    return results

def TF_generation(df):
    # Define the target columns for sorting
    sort_columns = ["joint", "type", "chord1", "brace1", "brace2", "brace3"]
    wave_columns = ["direction", "wave_height", "wave_period"]
    OD_columns = ["chd_OD1", "brc_OD1", "brc_OD2", "brc_OD3"]
    thickness_columns = ["chd_THK1", "brc_THK1", "brc_THK2", "brc_THK3"]
    elevation_columns = ["elevation1", "elevation2", "elevation3"]
    direction = df['direction'].iloc[0]
    n_crest = df.groupby(sort_columns + wave_columns, dropna=False).size().iloc[0]

    hs_columns = [f"{prefix}{i}_HS{j}" for prefix in ["BRC", "CHD"] for i in range(1, 4) for j in range(1, 9)]
    

    # Sort the DataFrame by the specified columns to ensure correct grouping
    df_sorted = df.sort_values(by=sort_columns, ignore_index=True)
    df_sorted["wave_height"] = pd.to_numeric(df_sorted["wave_height"], errors="coerce")

    # Check the length of the DataFrame is a multiple of n_crest
    if len(df_sorted) % n_crest != 0:
        raise ValueError("The number of rows in the DataFrame is not a multiple of n_crest.")

    # Reshape the DataFrame into (N, n_crest, M) where N is the number of groups, n_crest is the number of rows per group,
    # and M is the number of columns [see "transfer_function_generation_PD_version" for a clearer understanding of the function]
    n_chunks = len(df_sorted) // n_crest

    # Calculate the max and min over every n_crest rows for the hs_columns
    # Reshape is a "pandas groupby" equivalent in numpy. It splits the original 2D array in n_chuncks creating thus a 3D of size n_chuncks. Each sub-array has a size of n_crest
    # The above is much much quicker than pandas groupby
    # Reshape the hs_columns for max and min calculations
    # max_values and min_values should result in (n_chunks, n_columns)
    # here axis = 1 refers to the second dimension in the 3D dimension meaning max column of each chunch
    max_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).max(axis=1) 
    min_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).min(axis=1)

    # Extract the correct wave_height values for each group (one per chunk)
    # This extracts the first wave_height from each chunk, ensuring it's the same shape as max_values/min_values
    wave_heights = df_sorted["wave_height"].values[::n_crest]

    # Ensure the wave_heights and the range calculations are compatible
    # wave_heights should be reshaped to (n_chunks, 1) to allow broadcasting with max_values and min_values
    ranges = (max_values - min_values) / wave_heights.reshape(-1, 1)

    # This Creates a dataframe skeleton with every n_crest'th row. In this manner it will have the same size as ranges
    result_df = df_sorted.iloc[::n_crest][sort_columns + wave_columns + OD_columns + thickness_columns + elevation_columns].copy().reset_index(drop=True)
    
    # Add the calculated ranges to the result DataFrame
    result_ranges_df = pd.DataFrame(ranges, columns=hs_columns)
    final_result_df = pd.concat([result_df, result_ranges_df], axis=1)


    return final_result_df

def TF_inline_generation_older(df, inline_type):
    # Define the target columns for sorting
    if inline_type == "inline":
        sort_columns = ["joint", "memb1", "memb2"]
        geo_columns = ["memb1_OD", "memb1_THK", "memb2_OD", "memb2_THK"]
        hs_columns = [f"{prefix}_HS{j}" for prefix in ["memb1", "memb2"] for j in range(1, 9)]
    else:
        sort_columns = ["joint", "cone", "tubular"]       
        geo_columns = ["cone_OD", "cone_THK", "tub_OD", "tub_THK"]
        hs_columns = [f"{prefix}_HS{j}" for prefix in ["cone", "tub"]  for j in range(1, 9)]

    wave_columns = ["direction", "wave_height", "wave_period"]
    elevation_columns = ["Elevation"]
    n_crest = df.groupby(sort_columns + wave_columns, dropna=False).size().iloc[0] 

    # Sort the DataFrame by the specified columns to ensure correct grouping
    df_sorted = df.sort_values(by=sort_columns, ignore_index=True)

    # Check the length of the DataFrame is a multiple of n_crest
    if len(df_sorted) % n_crest != 0:
        raise ValueError("The number of rows in the DataFrame is not a multiple of n_crest.")

    # Reshape the DataFrame into (N, n_crest, M) where N is the number of groups, n_crest is the number of rows per group,
    # and M is the number of columns [see "transfer_function_generation_PD_version" for a clearer understanding of the function]
    n_chunks = len(df_sorted) // n_crest

    # Calculate the max and min over every n_crest rows for the hs_columns
    # Reshape is a "pandas groupby" equivalent in numpy. It splits the original 2D array in n_chuncks creating thus a 3D of size n_chuncks. Each sub-array has a size of n_crest
    # The above is much much quicker than pandas groupby
    # Reshape the hs_columns for max and min calculations
    # max_values and min_values should result in (n_chunks, n_columns)
    # here axis = 1 refers to the second dimension in the 3D dimension meaning max column of each chunch
    max_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).max(axis=1) 
    min_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).min(axis=1)

    # Extract the correct wave_height values for each group (one per chunk)
    # This extracts the first wave_height from each chunk, ensuring it's the same shape as max_values/min_values
    wave_heights = df_sorted["wave_height"].values[::n_crest]


    # Ensure the wave_heights and the range calculations are compatible
    # wave_heights should be reshaped to (n_chunks, 1) to allow broadcasting with max_values and min_values
    if inline_type == "inline":
        SCFtoes = df_sorted["SCFtoe"].values[::n_crest]
        SCFroots = df_sorted["SCFroot"].values[::n_crest]
        toe_ranges = SCFtoes.reshape(-1, 1) * (max_values - min_values) / wave_heights.reshape(-1, 1)
        root_ranges = SCFroots.reshape(-1, 1) * (max_values - min_values) / wave_heights.reshape(-1, 1)
        TFs = [toe_ranges, root_ranges]
    else: 
        SCFcones = df_sorted["SCFcone"].values[::n_crest]
        SCFtubs = df_sorted["SCFtubular"].values[::n_crest]
        # Reshape SCFcones and SCFtubs for calculations
        cone_ranges = SCFcones.reshape(-1, 1) * (max_values[:, :8] - min_values[:, :8]) / wave_heights.reshape(-1, 1)
        tub_ranges = SCFtubs.reshape(-1, 1) * (max_values[:, 8:] - min_values[:, 8:]) / wave_heights.reshape(-1, 1)
        combined_ranges = np.concatenate([cone_ranges, tub_ranges], axis=1) 

    # This Creates a dataframe skeleton with every n_crest'th row. In this manner it will have the same size as ranges
    result_df = df_sorted.iloc[::n_crest][sort_columns + wave_columns + geo_columns + elevation_columns].copy().reset_index(drop=True)
    
    # Add the calculated ranges to the result DataFrame
    results = []
    for i in [0,1]:
        if inline_type == "inline":
            result_ranges_df = pd.DataFrame(TFs[i], columns=hs_columns)      
        else:
            result_ranges_df = pd.DataFrame(combined_ranges, columns=hs_columns)

        result_ranges_df = pd.concat([result_df, result_ranges_df], axis=1)
        results.append(result_ranges_df.copy())
    
    return results

def TF_inline_generation(df, inline_type):
    # Define the target columns for sorting
    if inline_type == "inline":
        sort_columns = ["joint", "memb1", "memb2"]
        geo_columns = ["memb1_OD", "memb1_THK", "memb2_OD", "memb2_THK"]
        hs_columns = [f"{prefix}_HS{j}" for prefix in ["memb1", "memb2"] for j in range(1, 9)]
        scf_columns = ["SCFtoe", "SCFroot"]
    else:
        sort_columns = ["joint", "cone", "tubular"]       
        geo_columns = ["cone_OD", "cone_THK", "tub_OD", "tub_THK"]
        hs_columns = [f"{prefix}_HS{j}" for prefix in ["cone", "tub"]  for j in range(1, 9)]
        scf_columns = ["SCFcone", "SCFtubular"]

    wave_columns = ["direction", "wave_height", "wave_period"]
    elevation_columns = ["Elevation"]
    n_crest = df.groupby(sort_columns + wave_columns, dropna=False).size().iloc[0] 

    # Sort the DataFrame by the specified columns to ensure correct grouping
    df_sorted = df.sort_values(by=sort_columns, ignore_index=True)
    df_sorted["wave_height"] = pd.to_numeric(df_sorted["wave_height"], errors="coerce")

    # Check the length of the DataFrame is a multiple of n_crest
    if len(df_sorted) % n_crest != 0:
        raise ValueError("The number of rows in the DataFrame is not a multiple of n_crest.")

    # Reshape the DataFrame into (N, n_crest, M) where N is the number of groups, n_crest is the number of rows per group,
    # and M is the number of columns [see "transfer_function_generation_PD_version" for a clearer understanding of the function]
    n_chunks = len(df_sorted) // n_crest

    # Calculate the max and min over every n_crest rows for the hs_columns
    # Reshape is a "pandas groupby" equivalent in numpy. It splits the original 2D array in n_chuncks creating thus a 3D of size n_chuncks. Each sub-array has a size of n_crest
    # The above is much much quicker than pandas groupby
    # Reshape the hs_columns for max and min calculations
    # max_values and min_values should result in (n_chunks, n_columns)
    # here axis = 1 refers to the second dimension in the 3D dimension meaning max column of each chunch
    max_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).max(axis=1) 
    min_values = df_sorted[hs_columns].values.reshape(n_chunks, n_crest, -1).min(axis=1)

    # Extract the correct wave_height values for each group (one per chunk)
    # This extracts the first wave_height from each chunk, ensuring it's the same shape as max_values/min_values
    wave_heights = df_sorted["wave_height"].values[::n_crest]


    # Ensure the wave_heights and the range calculations are compatible
    # wave_heights should be reshaped to (n_chunks, 1) to allow broadcasting with max_values and min_values
    if inline_type == "inline":
        SCFtoes = df_sorted[scf_columns[0]].values[::n_crest]
        SCFroots = df_sorted[scf_columns[1]].values[::n_crest]
        toe_ranges = SCFtoes.reshape(-1, 1) * (max_values - min_values) / wave_heights.reshape(-1, 1)
        root_ranges = SCFroots.reshape(-1, 1) * (max_values - min_values) / wave_heights.reshape(-1, 1)
        TFs = [toe_ranges, root_ranges]
    else: 
        SCFcones = df_sorted[scf_columns[0]].values[::n_crest]
        SCFtubs = df_sorted[scf_columns[1]].values[::n_crest]
        # Reshape SCFcones and SCFtubs for calculations
        cone_ranges = SCFcones.reshape(-1, 1) * (max_values[:, :8] - min_values[:, :8]) / wave_heights.reshape(-1, 1)
        tub_ranges = SCFtubs.reshape(-1, 1) * (max_values[:, 8:] - min_values[:, 8:]) / wave_heights.reshape(-1, 1)
        combined_ranges = np.concatenate([cone_ranges, tub_ranges], axis=1) 

    # This Creates a dataframe skeleton with every n_crest'th row. In this manner it will have the same size as ranges
    result_df = df_sorted.iloc[::n_crest][sort_columns + wave_columns + geo_columns + elevation_columns].copy().reset_index(drop=True)
    
    # Add the calculated ranges to the result DataFrame
    results = []
    for i in [0,1]:
        if inline_type == "inline":
            result_ranges_df = pd.DataFrame(TFs[i], columns=hs_columns)  
            result_ranges_df[scf_columns[0]] = SCFtoes
            result_ranges_df[scf_columns[1]] = SCFroots   
        else:
            result_ranges_df = pd.DataFrame(combined_ranges, columns=hs_columns)
            result_ranges_df[scf_columns[0]] = SCFcones
            result_ranges_df[scf_columns[1]] = SCFtubs   

        result_ranges_df = pd.concat([result_df, result_ranges_df], axis=1)
        results.append(result_ranges_df.copy())
    
    return results

# def sigma_width_old(wp, w):
#     # Calculates the sigma width based on DNVGL-RP-C205.
#     if w <= wp:
#         return 0.07
#     else:
#         return 0.09
    
def sigma_width(wp, w):
    # Calculates the sigma width based on DNVGL-RP-C205.
    return np.where(w <= wp, 0.07, 0.09)

def spectrum_function(hs, wp, w, gamma):
    # Calculates the JONSWAP spectrum based on the Pierson-Moskowitz spectrum and JONSWAP model.
    # Pierson-Moskowitz Spectrum (S_PM)
    S_PM = (5 / 16) * (hs ** 2) * (wp ** 4) * (w ** -5) * np.exp(-(5 / 4) * (w / wp) ** -4)
    C = 0.2 / (0.065 * (gamma**0.803) + 0.135)
    # JONSWAP spectrum (S_J)
    sigma = sigma_width(wp, w)
    S_J = C * S_PM * gamma ** (np.exp(-0.5 * ((w - wp) / (sigma * wp)) ** 2))
    
    return S_J

def spectrum_function_vect(hs, wp, w, gamma):
    # Reshape the arrays for broadcasting
    hs = hs.reshape(-1, 1, 1)  # Shape (50, 1, 1)
    wp = wp.reshape(1, -1, 1)  # Shape (1, 22, 1)
    w = w.reshape(1, 1, -1)    # Shape (1, 1, 75)
    
    # Pierson-Moskowitz Spectrum (S_PM)
    S_PM = (5 / 16) * (hs ** 2) * (wp ** 4) * (w ** -5) * np.exp(-(5 / 4) * (w / wp) ** -4)

    # JONSWAP constant
    C = 0.2 / (0.065 * (gamma ** 0.803) + 0.135)

    # JONSWAP spectrum (S_J)
    sigma = sigma_width(wp, w)  # Assuming sigma_width function is vectorized
    S_J = C * S_PM * gamma ** (np.exp(-0.5 * ((w - wp) / (sigma * wp)) ** 2))

    return S_J

def response_function(w, h, S, exponent):
    return (w ** exponent) * (h ** 2) * S

def response_function_vect(w, h, S, exponent):
    # Reshape w to (1, 1, 75, 1) to broadcast with h and S
    w = w.reshape(1, 1, -1, 1)  # Now w has shape (1, 1, 75, 1)
    
    # Reshape h to (1, 1, 75, 48) to align with w and S
    h = h.reshape(1, 1, h.shape[0], h.shape[1])  # Now h has shape (1, 1, 75, 48)
    
    # Broadcasted operation for the response
    response = (w ** exponent) * (h ** 2) * S[:, :, :, np.newaxis]
    
    # print(response.shape)  # Check the shape for correctness

    return response

# def moment_integration_manual(H_w, hs, tp, gamma, scf):
#     # Initialize variables
#     moments = [0, 0]
#     wp = 2 * np.pi / tp  # angular spectral peak frequency
    
#     # Convert wave periods (T) to angular frequencies (w)
#     wave_periods = H_w.iloc[:, 0].values  # Extract wave periods
#     angular_frequencies = 2 * np.pi / wave_periods  # Convert to angular frequencies
    
#     # Get stresses and apply scale factor (scf)
#     stresses = H_w.iloc[:, 1].values * scf
    
#     # Calculate the spectrum values vectorially for all frequencies
#     spectrum_values = np.array([spectrum_function(hs, wp, w, gamma) for w in angular_frequencies])
    
#     # Calculate response functions for moments 0 and 2 for all frequencies
#     response_0 = np.array([response_function(w, H, s, 0) for w, H, s in zip(angular_frequencies, stresses, spectrum_values)])
#     response_2 = np.array([response_function(w, H, s, 2) for w, H, s in zip(angular_frequencies, stresses, spectrum_values)])
    
#     # Vectorized difference between adjacent angular frequencies
#     dw = np.diff(angular_frequencies)
    
#     # Vectorized integration using the trapezoidal rule
#     moments[0] = np.sum(((response_0[:-1] + response_0[1:]) / 2) * dw)
#     moments[1] = np.sum(((response_2[:-1] + response_2[1:]) / 2) * dw)
    
#     # Handle the triangular area between 0 and the first frequency
#     w1 = angular_frequencies[0]
#     f1_0 = response_0[0]
#     f1_2 = response_2[0]

#     # Triangular area for moment 0
#     m_0 = -f1_0 / w1 if w1 != 0 else 0
#     w0_0 = -f1_0 / m_0 if m_0 != 0 else 0
#     w0_0 = max(0, min(w1, w0_0))  # Ensure w0_0 is between 0 and w1
#     moments[0] += (f1_0 / 2) * (w1 - w0_0)

#     # Triangular area for moment 2
#     m_2 = -f1_2 / w1 if w1 != 0 else 0
#     w0_2 = -f1_2 / m_2 if m_2 != 0 else 0
#     w0_2 = max(0, min(w1, w0_2))  # Ensure w0_2 is between 0 and w1
#     moments[1] += (f1_2 / 2) * (w1 - w0_2)
    
#     return moments


def moment_integration(H_w, hs, tp, gamma, scf):
    # Initialize variables
    moments = [0, 0]
    wp = 2 * np.pi / tp  # Angular spectral peak frequency
    
    # Convert wave periods (T) to angular frequencies (w)
    wave_periods = H_w.iloc[:, 0].values  # Extract wave periods
    angular_frequencies = 2 * np.pi / wave_periods  # Convert to angular frequencies
    
    # Get stresses and apply scale factor (scf)
    stresses = H_w.iloc[:, 1].values * scf
    
    # Vectorized calculation of spectrum values
    spectrum_values = spectrum_function(hs, wp, angular_frequencies, gamma)  

    # Calculate response functions for moments 0 and 2 for all frequencies
    response_0 = response_function(angular_frequencies, stresses, spectrum_values, 0)
    response_2 = response_function(angular_frequencies, stresses, spectrum_values, 2)

    # Add initial point (0, 0) directly
    response_0 = np.insert(response_0, 0, 0)
    response_2 = np.insert(response_2, 0, 0)
    angular_frequencies = np.insert(angular_frequencies, 0, 0)
    
    # Vectorized integration using the trapezoidal rule
    moments[0] = np.trapz(response_0, x=angular_frequencies)
    moments[1] = np.trapz(response_2, x=angular_frequencies)
    
    return moments

def moment_integration_vect(H_w, wave_periods, hs, tp, gamma):
    # Initialize variables
    wp = 2 * np.pi / tp  # Angular spectral peak frequency

    angular_frequencies = 2 * np.pi / wave_periods  # Convert to angular frequencies

    # Vectorized calculation of spectrum values (3D array: Hs, Tp, N_freq)
    spectrum_values = spectrum_function_vect(hs, wp, angular_frequencies, gamma)

    # Calculate response functions for moments 0 and 2 for all non-zero frequencies
    response_0 = response_function_vect(angular_frequencies, H_w, spectrum_values, 0)
    response_2 = response_function_vect(angular_frequencies, H_w, spectrum_values, 2)

    # Now, insert zero into angular_frequencies after the responses are calculated
    angular_frequencies = np.insert(angular_frequencies, 0, 0)  # Insert zero at index 0

    # Create zero response arrays for the first frequency (zero frequency)
    zero_response_shape = (response_0.shape[0], response_0.shape[1], 1, response_0.shape[3])
    zero_response = np.zeros(zero_response_shape)  # Shape is (Hs, Tp, 1, dir)

    # Concatenate the zero response at the start of the frequency axis (axis=2)
    response_0 = np.concatenate((zero_response, response_0), axis=2)
    response_2 = np.concatenate((zero_response, response_2), axis=2)

    # Perform the trapezoidal integration along the 3rd axis (frequency axis)
    moments = [
        np.trapz(response_0, x=angular_frequencies, axis=2),
        np.trapz(response_2, x=angular_frequencies, axis=2)
    ]

    return moments

def moment_integration_vect_trap(H_w, wave_periods, hs, tp, gamma):
    # Initialize variables
    moments = [0, 0]
    wp = 2 * np.pi / tp  # Angular spectral peak frequency

    # Ensure wave_periods is not empty
    if wave_periods.size == 0:
        raise ValueError("wave_periods array is empty, cannot calculate angular_frequencies.")

    angular_frequencies = 2 * np.pi / wave_periods  # Convert to angular frequencies

    # Vectorized calculation of spectrum values
    spectrum_values = spectrum_function_vect(hs, wp, angular_frequencies, gamma)

    # Calculate response functions for moments 0 and 2 for all frequencies
    response_0 = response_function_vect(angular_frequencies, H_w, spectrum_values, 0)
    response_2 = response_function_vect(angular_frequencies, H_w, spectrum_values, 2)

    # Now, insert zeros after the responses are calculated

    # Insert zero into angular_frequencies
    angular_frequencies = np.insert(angular_frequencies, 0, 0)  # Insert zero at the at index 0

    # Create an initial zero response with the shape (50, 22, 1, 48)
    zero_response = np.zeros((response_0.shape[0], response_0.shape[1], 1, response_0.shape[3]))

    # Concatenate the initial zero-point to the response arrays along the 3rd axis (frequency axis)
    response_0 = np.concatenate((zero_response, response_0), axis=2)  # Shape becomes (50, 22, 76, 48)
    response_2 = np.concatenate((zero_response, response_2), axis=2)  # Same shape adjustment

    # Perform the trapezoidal integration along the 3rd axis (the frequency axis)
    moments[0] = np.trapz(response_0, x=angular_frequencies, axis=2)
    moments[1] = np.trapz(response_2, x=angular_frequencies, axis=2)

    return moments

def moment_integration_vect_simp(H_w, wave_periods, hs, tp, gamma):
    # Simpson Method [better for non-linear reponses]
    # Initialize variables
    moments = [0, 0]
    wp = 2 * np.pi / tp  # Angular spectral peak frequency

    # Ensure wave_periods is not empty
    if wave_periods.size == 0:
        raise ValueError("wave_periods array is empty, cannot calculate angular_frequencies.")

    angular_frequencies = 2 * np.pi / wave_periods  # Convert to angular frequencies

    # Vectorized calculation of spectrum values
    spectrum_values = spectrum_function_vect(hs, wp, angular_frequencies, gamma)

    # Calculate response functions for moments 0 and 2 for all frequencies
    response_0 = response_function_vect(angular_frequencies, H_w, spectrum_values, 0)
    response_2 = response_function_vect(angular_frequencies, H_w, spectrum_values, 2)

    # Now, insert zeros after the responses are calculated

    # Insert zero into angular_frequencies
    angular_frequencies = np.insert(angular_frequencies, 0, 0)  # Insert zero at the at index 0

    # Create an initial zero response with the shape (50, 22, 1, 48)
    zero_response = np.zeros((response_0.shape[0], response_0.shape[1], 1, response_0.shape[3]))

    # Concatenate the initial zero-point to the response arrays along the 3rd axis (frequency axis)
    response_0 = np.concatenate((zero_response, response_0), axis=2)  # Shape becomes (50, 22, 76, 48)
    response_2 = np.concatenate((zero_response, response_2), axis=2)  # Same shape adjustment

    # If angular_frequencies has an even number of points, add an additional point with equal values
    if len(angular_frequencies) % 2 == 0:
        angular_frequencies = np.append(angular_frequencies, angular_frequencies[-1])
        response_0 = np.concatenate((response_0, response_0[:, :, -1:, :]), axis=2)  # Duplicate last response_0 along the frequency axis
        response_2 = np.concatenate((response_2, response_2[:, :, -1:, :]), axis=2)  # Duplicate last response_2


    # Perform Simpson's rule integration along the 3rd axis (frequency axis)
    moments[0] = simpson(response_0, x=angular_frequencies, axis=2)
    moments[1] = simpson(response_2, x=angular_frequencies, axis=2)
    
    return moments


# def moment_integration_old(H_w, hs, tp, gamma, scf):
#     # Initialize variables
#     moments = [0, 0]
#     wp = 2 * np.pi / tp  # Angular spectral peak frequency
    
#     # Convert wave periods (T) to angular frequencies (w)
#     wave_periods = H_w.iloc[:, 0].values  # Extract wave periods
#     angular_frequencies = 2 * np.pi / wave_periods  # Convert to angular frequencies
    
#     # Get stresses and apply scale factor (scf)
#     stresses = H_w.iloc[:, 1].values * scf
    
#     # Calculate the spectrum values vectorially for all frequencies
#     spectrum_values = np.array([spectrum_function(hs, wp, w, gamma) for w in angular_frequencies])
    
#     # Calculate response functions for moments 0 and 2 for all frequencies
#     response_0 = np.array([response_function(w, H, S, 0) for w, H, S in zip(angular_frequencies, stresses, spectrum_values)])
#     response_2 = np.array([response_function(w, H, S, 2) for w, H, S in zip(angular_frequencies, stresses, spectrum_values)])
    
#     # Add initial point (0, 0) to the beginning of angular_frequencies and response arrays
#     angular_frequencies = np.concatenate(([0], angular_frequencies))
#     response_0 = np.concatenate(([0], response_0))
#     response_2 = np.concatenate(([0], response_2))
    
#     # Vectorized integration using the trapezoidal rule
#     moments[0] = np.trapz(response_0, x=angular_frequencies)
#     moments[1] = np.trapz(response_2, x=angular_frequencies)
    
#     return moments



def thickness_effect(thk, t_ref, k):
    return (thk / t_ref)**k

def get_t_ref(code, type):
    return 25 if code == "EC" else 16 if (type == "joint") & (code == "DNV") else 25

def get_tag_ID(column_name, ID):
    # Mapping conditions based on column_name
    if "CHD1" in column_name:
        return "chd_THK1" if ID == "THK" else "brace1"
    elif "CHD2" in column_name:
        return "chd_THK1" if ID == "THK" else "brace2"
    elif "CHD3" in column_name:
        return "chd_THK1" if ID == "THK" else "brace3"
    elif "BRC1" in column_name:
        return "brc_THK1" if ID == "THK" else "brace1"
    elif "BRC2" in column_name:
        return "brc_THK2" if ID == "THK" else "brace2"
    elif "BRC3" in column_name:
        return "brc_THK3" if ID == "THK" else "brace3"
    else:
        return "Unknown thickness"  # Default case if column_name doesn't match any condition
    
def rayleigh_sigma(sigma, RMS):
    return (sigma / (RMS ** 2)) * np.exp(- (sigma ** 2) / (2 * RMS ** 2))

# def sn_curve_integration_manual(sigma_1, sigma_2, log_a, M, RMS):
#     d_s0 = 1  # Replace this with the actual value for d_s0 if different
#     sigma1 = sigma_1
#     FN_sum = 0
#     a = 10 ** log_a
    
#     while sigma1 < sigma_2:
#         d_s = d_s0
#         FN_1 = (sigma1 ** M) * rayleigh_sigma(sigma1, RMS)
#         sigma2 = sigma1 + d_s0
#         if sigma2 > sigma_2:  # Check if sigma is beyond the upper limit
#             sigma2 = sigma_2
#             d_s = sigma2 - sigma1
        
#         FN_2 = (sigma2 ** M) * rayleigh_sigma(sigma2, RMS)
        
#         FN_sum += ((FN_2 + FN_1) / 2) * d_s  # Trapezoidal method integration
        
#         sigma1 = sigma1 + d_s0
    
#     FN_sum = FN_sum / a
#     return FN_sum

def sn_curve_integration(sigma_1, sigma_2, log_a, M, RMS, num_points=500):
    # Using the trapezoidal method [quickest]
    a = 10 ** log_a

    # Generate discrete sigma values
    sigma_values = np.linspace(sigma_1, sigma_2, num_points)
    
    # Compute integrand values
    integrand_values = (sigma_values ** M) * rayleigh_sigma(sigma_values, RMS)
    
    # Compute the integral using trapezoidal rule
    result = np.trapz(integrand_values, sigma_values)
    
    return result / a

def sn_curve_integration_vect_trap(sigma_1, sigma_2, log_a, M, RMS, num_points=100):
    # Using the trapezoidal method [quickest]
    a = 10 ** log_a

    # Generate discrete sigma values
    sigma_values = np.linspace(sigma_1, sigma_2, num_points)

    # Calculate Rayleigh sigma for each sigma value
    r = rayleigh_sigma(sigma_values, RMS)  # Shape: (500,50,22,48)
    moment = (sigma_values ** M)[:, np.newaxis, np.newaxis, np.newaxis]  # Shape: (500, 1, 1, 1)

    # Compute integrand values
    integrand_values = r * moment  # Shape: (500,50,22,48)

    # Compute the integral using trapezoidal rule
    result = np.trapz(integrand_values, x=sigma_values, axis=0)  # Integrate along the first axis

    return result / a  # This will yield a shape of (50,22,48)

def sn_curve_integration_vect(sigma_1, sigma_2, log_a, M, RMS):
    a = 10 ** log_a
    
    # Integrating with adaptive quadrature
    def integrand(sigma_values):
        r = rayleigh_sigma(sigma_values, RMS)  # Shape: (50,22,48)
        moment = sigma_values ** M
        return r * moment[..., np.newaxis, np.newaxis, np.newaxis]
    
    result = quad_vec(integrand, sigma_1, sigma_2)[0]
    
    return result / a





# def sn_curve_integration_quad(sigma_1, sigma_2, log_a, M, RMS):
#     a = 10 ** log_a
#     def integrand(sigma, M, RMS):
#         return (sigma ** M) * rayleigh_sigma(sigma, RMS)

#     result, _ = quad(integrand, sigma_1, sigma_2, args=(M, RMS))
#     return result / a

# def calc_damages(directions, TFs, scatter_diagrams, fractions, spectrum_gamma, check, SN_Curves, Life, safety_factor):
#     # TF columns 
#     joint_columns = ["joint", "type", "chord1", "brace1", "brace2", "brace3"]
#     thickness_columns = ["chd_THK1", "brc_THK1", "brc_THK2", "brc_THK3"]
#     tf_columns = [f"{prefix}{i}_HS{j}" for prefix in ["BRC", "CHD"] for i in range(1, 4) for j in range(1, 9)]
#     SN_Curve_row = (SN_Curves["SN_Curve"] == check["SN_Curve"]) & (SN_Curves["Environment"] == check["Environment"])

#     SN_Curve = SN_Curves[SN_Curve_row]
#     s_inflection = SN_Curve["s_inflection"].values[0] / safety_factor
#     n_inflection = SN_Curve["Inflection"].values[0]
#     m1 = SN_Curve["m1"].values[0]
#     m2 = SN_Curve["m2"].values[0]

#     log_a1 = np.log10(n_inflection) + m1 * np.log10(s_inflection) 
#     log_a2 = np.log10(n_inflection) + m2 * np.log10(s_inflection) 

#     # Group by the joint and thickness columns from the first direction (since it's common for all)
#     grouped_TF = TFs[str(directions[0])].groupby(joint_columns + thickness_columns, dropna=False)
    
#     # Create a list to store rows for all Hs and Tp combinations for this joint and direction
#     results_list = []
#     idx = 0
#     # Iterate over the grouped TF data
#     for (joint, type, chord1, brace1, brace2, brace3, chd_THK1, brc_THK1, brc_THK2, brc_THK3), joint_data in grouped_TF:
#         joint_index = joint_data.index
#         joint_damages = []
#         results_row = {
#             'joint': joint,
#             'type': type,
#             'chord1': chord1,
#             'brace1': brace1,
#             'brace2': brace2,
#             'brace3': brace3,
#             'Code': check["Code"],
#             'Code Check': check["type"],
#             'side': check["side"],
#             'Environment': check["Environment"], 
#             'safety_factor': safety_factor,
#             'SN_Curve': check["SN_Curve"]
#         }

      
#         # Iterate over each direction
#         for direction in directions:
#             print(f"Analyzing Joint {joint} Type {type} Direction {direction}")
#             direction = str(direction)
#             fraction = fractions[direction].values[0]
            
#             # Load Hs and Tp columns and their associated percentages (fractions) from scatter_diagrams
#             hs_series = scatter_diagrams[direction]['HS'].values.astype(float)
#             tp_columns = scatter_diagrams[direction].columns[1:].values.astype(float)

#             # Retrieve the scatter diagram DataFrame for the direction
#             scatter_data = scatter_diagrams[direction]

#             # Iterate over the Hs and Tp values
#             for hs_idx, hs in enumerate(hs_series):
#                 for tp_idx, tp in enumerate(tp_columns):
#                     # Retrieve the corresponding percentage (fraction) from scatter_data for the current Hs and Tp
#                     percentage = scatter_data.iloc[hs_idx, tp_idx + 1]  # Assuming the percentage is in the data
#                     alpha = fraction * percentage / 100
#                     # Create a dictionary for this specific joint, direction, Hs, and Tp combination
#                     if alpha != 0:
#                         row_data = {
#                             'joint': joint,
#                             'type': type,
#                             'chord1': chord1,
#                             'brace1': brace1,
#                             'brace2': brace2,
#                             'brace3': brace3,
#                             'direction': direction,
#                             'Hs': hs,
#                             'Tp': tp,
#                             'Fraction': alpha,  
#                             'SN_Curve': check["SN_Curve"]
#                         }

#                         # Iterate over all transfer function columns (tf_columns)
#                         for tf_column in tf_columns:
#                             thk = joint_data[get_tag_ID(tf_column, "THK")]
#                             scf = thickness_effect(thk, get_t_ref(SN_Curve["Code"].values[0], check["type"]), SN_Curve["k"].values[0])
                            
#                             braces = joint_data[get_tag_ID(tf_column, "BRACE")]
#                             # Check if H_w contains any non-NaN values
#                             if braces.isna().all().all():  # Checks if all values in the DataFrame are NaN
#                                 # Skip row if all values are NaN
#                                 continue
#                             else:
#                                 # Perform the spectral moment calculation
#                                 H_w = TFs[direction].loc[joint_index, ["wave_period", tf_column]]
#                                 moments = moment_integration(H_w, hs, tp, spectrum_gamma, scf)
#                                 RMS = moments[0] ** 0.5
#                                 Tz = 2 * np.pi * (moments[0] / moments[1]) ** 0.5
#                                 n_cycles = alpha * Life * 365 * 24 * 60 * 60 / Tz
                                
#                                 # Add the moment for the current tf_column to the row_data
#                                 row_data[f'{tf_column}_n_cycles'] = n_cycles
#                                 row_data[f'{tf_column}_RMS'] = RMS
#                                 row_data[f'{tf_column}_Tz'] = Tz

#                                 D_m_5 = n_cycles * sn_curve_integration(0, s_inflection, log_a2, m2, RMS)
#                                 D_m_3 = n_cycles * sn_curve_integration(s_inflection, 2000, log_a1, m1, RMS)
#                                 row_data[f'{tf_column}_Damage'] = D_m_3 + D_m_5
#                                 # print(f"Joint {joint} Type {type} Direction {direction} D_5 {D_m_5} D_3 {D_m_3}")

#                         # After processing all tf_columns, append the row_data dictionary to the list
#                         joint_damages.append(row_data)
#             break
#         joint_damages_df = pd.DataFrame(joint_damages)
        
#         idx += 1
#         # Save joint_damages_df to a temporary CSV file
#         temp_csv_path = f'joint_{idx}.csv'
#         joint_damages_df.to_csv(temp_csv_path, index=False)
#         print(f"Temporary CSV saved to {temp_csv_path}")
        
        
#         # Create a Series to contain the sum of each column ending in '_Damage'
#         damage_sum_series = joint_damages_df.filter(like='_Damage').sum()
        
#         # Populate results_row with values from damage_sum_series
#         for key in damage_sum_series.index:
#             results_row[key] = damage_sum_series[key]
        
#         # Convert the list of rows into a DataFrame
#         results_list.append(results_row) 
#         break

#     results_df = pd.DataFrame(results_list)
    
#     # Add a Total_Damage column to results_df
#     damage_columns = [col for col in results_df.columns if col.endswith('_Damage')]
#     results_df['Max_Damage'] = results_df[damage_columns].max(axis=1)
    
#     CSV_filename = "FLS_RESULTS.csv"
#     results_df.to_csv(f"{CSV_filename}", index=False)
#     print(f"Results saved to {CSV_filename}")
#     return results_df
def calc_inline_deterministic_damages_vect(directions, SRs_original, scatter, check, SN_Curves):    
    results_df = pd.DataFrame()
    SRs = {key: df.copy(deep=True) for key, df in SRs_original.items()}
    safety_factor = check["safety_factor"]
    Life = check["Life"]
    spectrum_gamma = check["spectrum_gamma"]
    inline_type = check["type"]
    # SR columns 
    if inline_type == "inline": 
        joint_columns = ["joint", "memb1", "memb2"]
        OD_columns = ["memb1_OD", "memb2_OD"]
        THK_columns = ["memb1_THK", "memb2_THK"]
        prefixes = ["memb1", "memb2"]
        SFC_columns = ["SCFtoe", "SCFroot"] 
        SR_columns = np.array([f"{prefix}_HS{j}" for prefix in prefixes for j in range(1, 9)])
    else:
        joint_columns = ["joint", "cone", "tubular"]
        OD_columns = ["cone_OD", "tub_OD"]
        THK_columns = ["cone_THK", "tub_THK"]
        prefixes = ["cone", "tub"]
        SFC_columns = ["SCFcone", "SCFtubular"] 
        SR_columns = np.array([f"{prefix}_HS{j}" for prefix in prefixes for j in range(1, 9)]) 

    SN_Curve_row = (SN_Curves["SN_Curve"] == check["SN_Curve"]) & (SN_Curves["Environment"] == check["Environment"])

    SN_Curve = SN_Curves[SN_Curve_row]
    s_inflection = SN_Curve["s_inflection"].values[0] / safety_factor
    n_inflection = SN_Curve["Inflection"].values[0]
    m1 = SN_Curve["m1"].values[0]
    m2 = SN_Curve["m2"].values[0]

    log_a1 = np.log10(n_inflection) + m1 * np.log10(s_inflection) 
    log_a2 = np.log10(n_inflection) + m2 * np.log10(s_inflection) 
    
    # Define the elevation range
    min_ele = check["Elevation_min"]
    max_ele = check["Elevation_max"]
    
    # DEFINE CURRENT SCOPE BY [TARGET ELEVATION, ID < 600 FOR ROOT CHECKS]
        # Identify rows where elevation condition is met for each column

    for direction in directions:
        direction = str(direction)
        elevation_condition = (SRs[direction][f'Elevation'] < min_ele) | (SRs[direction][f'Elevation'] > max_ele) 
        # Zero out the transfer function columns where conditions are met (iterate over range 1 to 3)
        # Filter columns that match the prefix and brace index i
        columns_to_zero = SRs[direction].filter(regex='_HS').columns
        # Set all the matching columns to 0 where condition is True
        SRs[direction].loc[elevation_condition, columns_to_zero] = 0

    # Group by the joint and thickness columns from the first direction (since it's common for all)
    grouped_SR = SRs[str(directions[0])].groupby(joint_columns, dropna=False)
    
    # Create a list to store rows for all Hs and Tp combinations for this joint and direction
    results_list = []
    joint_n = 0
    # Iterate over the grouped SR data
    for (joint, memb1, memb2), joint_data in grouped_SR:
        group_key = (joint, memb1, memb2)
        within_bounds = (joint_data["Elevation"] >= min_ele) | (joint_data["Elevation"] <= max_ele)
        within_bounds = within_bounds.any().any()
        if within_bounds:
            n_columns = len(SR_columns)
            damages_sum = np.zeros(n_columns, dtype=float)
            joint_n += 1

            results_row = {
                'joint': joint,
                joint_columns[1]: memb1,
                joint_columns[2]: memb2,
                f'{prefixes[0]}_OD': joint_data[OD_columns[0]].iloc[0],
                f'{prefixes[1]}_OD': joint_data[OD_columns[1]].iloc[0],
                f'{prefixes[0]}_THK': joint_data[THK_columns[0]].iloc[0],
                f'{prefixes[1]}_THK': joint_data[THK_columns[1]].iloc[0],
                'elevation': joint_data["Elevation"].iloc[0],
                'Code': check["Code"],
                'Code Check': check["type"],
                'side': check["side"],
                'Environment': check["Environment"], 
                'safety_factor': safety_factor,
                'SN_Curve': check["SN_Curve"],
                SFC_columns[0]: joint_data[SFC_columns[0]].iloc[0],
                SFC_columns[1]: joint_data[SFC_columns[1]].iloc[0]
            }
            # Create a dictionary to store SCFs using dictionary comprehension
            thickness_values = [joint_data[THK_columns[0]].iloc[0], joint_data[THK_columns[1]].iloc[0]]

            thickness_SCF = {
                key: thickness_effect(value, get_t_ref(SN_Curve["Code"].values[0], check["type"]), SN_Curve["k"].values[0])
                for key, value in zip(prefixes, thickness_values)
            }
        
            # Iterate over each direction
            for direction in directions:
                check_type = check["type"]
                side = check["side"]
                print(f"Analyzing Joint {joint} Type {check_type} Side {side}  Direction {direction}")
                direction = str(direction)
                grouped_SR_dir = SRs[direction].groupby(joint_columns, dropna=False)
                joint_data_dir = grouped_SR_dir.get_group(group_key)
                joint_index_dir = joint_data_dir.index
                wave_periods = SRs[direction].loc[joint_index_dir, "wave_period"].astype(float)
                wave_heights = SRs[direction].loc[joint_index_dir, "wave_height"].astype(float)
                SR = SRs[direction].loc[joint_index_dir, SR_columns].copy(deep=True)
                # Extract wave data from SRs
                wave_periods = SRs[direction].loc[joint_index_dir, "wave_period"].to_numpy(dtype=float)
                wave_heights = SRs[direction].loc[joint_index_dir, "wave_height"].to_numpy(dtype=float)

                # Extract scatter diagram data
                H_series = scatter['H'].values.astype(float)
                T_series = scatter['T'].values.astype(float)
                scatter_column = scatter[direction].values

                # --- STEP 1: Check if wave_periods/heights match scatter H/T directly ---
                same_heights = np.array_equal(wave_heights, H_series)
                same_periods = np.array_equal(wave_periods, T_series)

                if same_heights and same_periods:
                    # Perfect match → use full series directly
                    H_filtered = H_series
                    T_filtered = T_series
                    scatter_filtered = scatter_column

                else:
                    # Mismatch → fall back to mask-based filtering
                    scatter_mask = scatter_column != 0
                    H_filtered = H_series[scatter_mask]
                    T_filtered = T_series[scatter_mask]
                    scatter_filtered = scatter_column[scatter_mask]

                # Optional: safety check
                if len(H_filtered) != len(wave_heights):
                    raise ValueError(
                        f"Mismatch after masking: H_filtered={len(H_filtered)}, wave_heights={len(wave_heights)}"
                    )

                if len(T_filtered) != len(wave_periods):
                    raise ValueError(
                        f"Mismatch after masking: T_filtered={len(T_filtered)}, wave_periods={len(wave_periods)}"
                    )

                # Apply thickness correction
                for column in SR_columns:
                    SR[column] *= thickness_SCF[column.split("_")[0]]

                # Compute cycles
                n_cycles = scatter_filtered * Life 
                # n_cycles = scatter_column * Life
                # S–N curve selection
                log_a = np.where(SR > s_inflection, log_a1, log_a2)
                m = np.where(SR > s_inflection, m1, m2)
                SR = SR.apply(pd.to_numeric, errors='coerce').astype(float)
                print(getattr(SR, "dtype", type(SR)))
                print(SR.head())
                # Fatigue damage
                N_to_failure = 10 ** (log_a - m * np.log10(SR))
                n_columns = N_to_failure.shape[1]
                n_cycles_extended = np.tile(n_cycles[:, np.newaxis], (1, n_columns))
                damages_direction = n_cycles_extended / N_to_failure
                damages_sum += damages_direction.sum(axis=0)
        
            for i, col in enumerate(SR_columns):
                results_row[f"{col}_Damage"] = damages_sum[i]
            
            # Convert the list of rows into a DataFrame
            results_list.append(results_row) 

    results_df = pd.DataFrame(results_list)
    results_df = results_df.reset_index(drop=True)

    
    # Add a Total_Damage column to results_df
    memb1_damage_columns = [col for col in results_df.columns if (col.endswith('_Damage') & col.startswith(prefixes[0]))]
    memb2_damage_columns = [col for col in results_df.columns if (col.endswith('_Damage') & col.startswith(prefixes[1]))]


    results_df[f'Max_{prefixes[0]}_Damage'] = results_df[memb1_damage_columns].max(axis=1)
    results_df[f'Max_{prefixes[1]}_Damage'] = results_df[memb2_damage_columns].max(axis=1)
   
    return results_df

def calc_deterministic_damages_vect(directions, SRs_original, scatter, check, SN_Curves,path_SCF, full_report = False):
    results_df = pd.DataFrame()
    SRs = {key: df.copy(deep=True) for key, df in SRs_original.items()}
    # SR columns 
    safety_factor = check["safety_factor"]
    Life = check["Life"]
    spectrum_gamma = check["spectrum_gamma"]

    # SCF_df = pd.read_parquet(path_SCF)
    SCF_df = pd.read_csv(path_SCF)
    joint_columns = ["joint", "type", "chord1", "brace1", "brace2", "brace3"]
    thickness_columns = ["chd_THK1", "brc_THK1", "brc_THK2", "brc_THK3"]
    elevation_columns = ["elevation1", "elevation2", "elevation3"]
    SR_columns = np.array([f"{prefix}{i}_HS{j}" for prefix in ["BRC", "CHD"] for i in range(1, 4) for j in range(1, 9)])
    SN_Curve_row = (SN_Curves["SN_Curve"] == check["SN_Curve"]) & (SN_Curves["Environment"] == check["Environment"])

    SN_Curve = SN_Curves[SN_Curve_row]
    s_inflection = SN_Curve["s_inflection"].values[0] / safety_factor
    n_inflection = SN_Curve["Inflection"].values[0]
    m1 = SN_Curve["m1"].values[0]
    m2 = SN_Curve["m2"].values[0]

    log_a1 = np.log10(n_inflection) + m1 * np.log10(s_inflection) 
    log_a2 = np.log10(n_inflection) + m2 * np.log10(s_inflection) 
    
    # Define the elevation range
    min_ele = check["Elevation_min"]
    max_ele = check["Elevation_max"]
    
    # DEFINE CURRENT SCOPE BY [TARGET ELEVATION, ID < 600 FOR ROOT CHECKS]
        # Identify rows where elevation condition is met for each column
    
    for direction in directions:
        direction = str(direction)
        elevation_conditions = [(SRs[direction][f'elevation{i}'] < min_ele) | (SRs[direction][f'elevation{i}'] > max_ele) for i in range(1, 4)]
        # For root checks, identify where brc_ID is <= 600
        if check["side"] == "root":
            brc_conditions = [ (SRs[direction][f'brc_OD{i}'] - 2 * SRs[direction][f'brc_THK{i}']) > 600 for i in range(1, 4)]
        else:
            # brc_conditions = [False] * len(SRs[direction])
            brc_conditions = [pd.Series([False] * len(SRs[direction])) for _ in range(1, 4)]
        # Zero out the transfer function columns where conditions are met (iterate over range 1 to 3)
        for i in range(1, 4):
            for prefix in ['BRC', 'CHD']:
                # Apply the combined elevation and brc_ID conditions
                condition = elevation_conditions[i-1] | brc_conditions[i-1]    
                # Filter columns that match the prefix and brace index i
                # columns_to_zero = SRs[direction].filter(like=f'{prefix}{i}_HS').columns
                columns_to_zero = SRs[direction].filter(like=f'{prefix}{i}_HS').columns.tolist()
                # Set all the matching columns to 0 where condition is True    
                SRs[direction].loc[condition, columns_to_zero] = 0

    # Group by the joint and thickness columns from the first direction (since it's common for all)
    grouped_SR = SRs[str(directions[0])].groupby(joint_columns, dropna=False)
    
    # Create a list to store rows for all Hs and Tp combinations for this joint and direction
    results_list = []
    scf_series_list = []
    # joint_n = 0
    
    for (joint, type, chord1, brace1, brace2, brace3), joint_data in grouped_SR:
        group_key = (joint, type, chord1, brace1, brace2, brace3)
        within_bounds = joint_data[elevation_columns].apply(lambda col: (col >= min_ele) & (col <= max_ele))
        within_bounds = within_bounds.any().any()  # True if any value is within bounds
        if within_bounds:
            SCFs_row = SCF_df.loc[(SCF_df["joint"] == joint) & (SCF_df["brace1"] == brace1)].iloc[0]
            scf_series = SCFs_row.filter(regex="^SCF")
            # If the check is for "root"
            if check["side"] == "root":
                # Extract R columns (those starting with R)
                R_series = SCFs_row.filter(regex="^R")
                # Multiply corresponding columns: R containing FX with SCF containing FX, and similarly for IPB and OPB
                for suffix in ["FX", "IPB", "OPB"]:
                    # Filter SCF and R columns by suffix
                    scf_filtered = scf_series.filter(regex=f"{suffix}")
                    R_filtered = R_series.filter(regex=f"{suffix}")
                    # Multiply corresponding SCF and R columns and update scf_series directly
                    for col in scf_filtered.index:
                        R_col = R_filtered.index[R_filtered.index.str.contains(suffix)][0]  # Find the matching R column for the suffix
                        scf_series[col] = scf_filtered[col] * R_series[R_col]
            
            scf_series_list.append(scf_series)
            joint_index = joint_data.index
            n_columns = len(SR_columns)
            damages_sum = np.zeros(n_columns, dtype=float)
            # joint_n += 1
            chd_THK1 = joint_data["chd_THK1"].iloc[0]
            brc_THK1 = joint_data["brc_THK1"].iloc[0]
            brc_THK2 = joint_data["brc_THK2"].iloc[0]
            brc_THK3 = joint_data["brc_THK3"].iloc[0]
            results_row = {
                'joint': joint,
                'type': type,
                'chord1': chord1,
                'brace1': brace1,
                'brace2': brace2,
                'brace3': brace3,
                'chd_OD1': joint_data["chd_OD1"].iloc[0],
                'chd_THK1': joint_data["chd_THK1"].iloc[0],
                'brc_OD1': joint_data["brc_OD1"].iloc[0],
                'brc_THK1': joint_data["brc_THK1"].iloc[0],
                'brc_OD2': joint_data["brc_OD2"].iloc[0],
                'brc_THK2': joint_data["brc_THK2"].iloc[0],
                'brc_OD3': joint_data["brc_OD3"].iloc[0],
                'brc_THK3': joint_data["brc_THK3"].iloc[0],
                'Code': check["Code"],
                'Code Check': check["type"],
                'side': check["side"],
                'Environment': check["Environment"], 
                'safety_factor': safety_factor,
                'SN_Curve': check["SN_Curve"]
            }
            # Create a dictionary to store SCFs using dictionary comprehension
            thickness_keys = ['CHD1', 'CHD2', 'CHD3', 'BRC1', 'BRC2', 'BRC3']
            thickness_values = [chd_THK1, chd_THK1, chd_THK1, brc_THK1, brc_THK2, brc_THK3]

            thickness_SCF = {
                key: thickness_effect(value, get_t_ref(SN_Curve["Code"].values[0], check["type"]), SN_Curve["k"].values[0])
                for key, value in zip(thickness_keys, thickness_values)
            }
        
            # Iterate over each direction
            for direction in directions:
                print(f"Analyzing Joint {joint} Type {type} Direction {direction}")
                direction = str(direction)

                grouped_SR_dir = SRs[direction].groupby(joint_columns, dropna=False)
                joint_data_dir = grouped_SR_dir.get_group(group_key)   # if you're sure it exists
                joint_index_dir = joint_data_dir.index
                SR = SRs[direction].loc[joint_index_dir, SR_columns].copy(deep=True)
                
                # Extract wave data from SRs
                wave_periods = SRs[direction].loc[joint_index_dir, "wave_period"].to_numpy(dtype=float)
                wave_heights = SRs[direction].loc[joint_index_dir, "wave_height"].to_numpy(dtype=float)

                # Extract scatter diagram data
                H_series = scatter['H'].values.astype(float)
                T_series = scatter['T'].values.astype(float)
                scatter_column = scatter[direction].values

                # --- STEP 1: Check if wave_periods/heights match scatter H/T directly ---
                same_heights = np.array_equal(wave_heights, H_series)
                same_periods = np.array_equal(wave_periods, T_series)

                if same_heights and same_periods:
                    # Perfect match → use full series directly
                    H_filtered = H_series
                    T_filtered = T_series
                    scatter_filtered = scatter_column

                else:
                    # Mismatch → fall back to mask-based filtering
                    scatter_mask = scatter_column != 0
                    H_filtered = H_series[scatter_mask]
                    T_filtered = T_series[scatter_mask]
                    scatter_filtered = scatter_column[scatter_mask]

                # Optional: safety check
                if len(H_filtered) != len(wave_heights):
                    raise ValueError(
                        f"Mismatch after masking: H_filtered={len(H_filtered)}, wave_heights={len(wave_heights)}"
                    )

                if len(T_filtered) != len(wave_periods):
                    raise ValueError(
                        f"Mismatch after masking: T_filtered={len(T_filtered)}, wave_periods={len(wave_periods)}"
                    )
                # Apply Thickness Correction to the relevant columns in H_w
                for column in SR_columns:  # Loop through all transfer function columns
                        SR[column] *= thickness_SCF[column[:4]]  # Multiply the column by its corresponding SCF
                n_cycles = scatter_filtered * Life 
                log_a = np.where(SR > s_inflection, log_a1, log_a2)
                m = np.where(SR > s_inflection, m1, m2)
                N_to_failure = 10 ** (log_a - m * np.log10(SR))
                n_columns = N_to_failure.shape[1]
                n_cycles_extended = np.tile(n_cycles[:, np.newaxis], (1, n_columns))
                damages_direction = n_cycles_extended / N_to_failure
                damages_sum += damages_direction.sum(axis=0)

                a = 1
            
            for i, SR_column in enumerate(SR_columns):
                results_row[f"{SR_column}_Damage"] = damages_sum[i]
            
            # Convert the list of rows into a DataFrame
            results_list.append(results_row) 
            # if joint_n == 50:
            # break

    results_df = pd.DataFrame(results_list)
    scf_series_df = pd.DataFrame(scf_series_list)
    results_df = results_df.reset_index(drop=True)
    scf_series_df = scf_series_df.reset_index(drop=True)
    print("Results DF Index:", results_df.index)
    print("SCF Series DF Index:", scf_series_df.index)
    # # Concatenate the results_df and scf_series_df side by side
    final_df = pd.concat([results_df, scf_series_df], axis=1)

    # print(results_df)
    
    # Add a Total_Damage column to results_df
    brace_damage_columns = [col for col in final_df.columns if (col.endswith('_Damage') & col.startswith('BRC'))]
    chord_damage_columns = [col for col in final_df.columns if (col.endswith('_Damage') & col.startswith('CHD'))]


    final_df['Max_BRC_Damage'] = final_df[brace_damage_columns].max(axis=1)
    final_df['Max_CHD_Damage'] = final_df[chord_damage_columns].max(axis=1)
   
    return final_df

def calc_joint_damages_vect(directions, TFs_original, scatter_diagrams, check, SN_Curves,path_SCF, full_report = False):
    results_df = pd.DataFrame()
    TFs = {key: df.copy(deep=True) for key, df in TFs_original.items()}
    # TF columns 
    safety_factor = check["safety_factor"]
    Life = check["Life"]
    spectrum_gamma = check["spectrum_gamma"]

    SCF_df = pd.read_csv(path_SCF)
    joint_columns = ["joint", "type", "chord1", "brace1", "brace2", "brace3"]
    thickness_columns = ["chd_THK1", "brc_THK1", "brc_THK2", "brc_THK3"]
    elevation_columns = ["elevation1", "elevation2", "elevation3"]
    tf_columns = np.array([f"{prefix}{i}_HS{j}" for prefix in ["BRC", "CHD"] for i in range(1, 4) for j in range(1, 9)])
    SN_Curve_row = (SN_Curves["SN_Curve"] == check["SN_Curve"]) & (SN_Curves["Environment"] == check["Environment"])

    SN_Curve = SN_Curves[SN_Curve_row]
    s_inflection = SN_Curve["s_inflection"].values[0] / safety_factor
    n_inflection = SN_Curve["Inflection"].values[0]
    m1 = SN_Curve["m1"].values[0]
    m2 = SN_Curve["m2"].values[0]

    log_a1 = np.log10(n_inflection) + m1 * np.log10(s_inflection) 
    log_a2 = np.log10(n_inflection) + m2 * np.log10(s_inflection) 
    T_year = 365 * 24 * 60 * 60 
    
    # Define the elevation range
    min_ele = check["Elevation_min"]
    max_ele = check["Elevation_max"]
    
    # DEFINE CURRENT SCOPE BY [TARGET ELEVATION, ID < 600 FOR ROOT CHECKS]
        # Identify rows where elevation condition is met for each column
    direction = str(directions[0])
    elevation_conditions = [(TFs[direction][f'elevation{i}'] < min_ele) | (TFs[direction][f'elevation{i}'] > max_ele) for i in range(1, 4)]

    # For root checks, identify where brc_ID is <= 600
    if check["side"] == "root":
        brc_conditions = [ (TFs[direction][f'brc_OD{i}'] - 2 * TFs[direction][f'brc_THK{i}']) > 900 for i in range(1, 4)]
    else:
        brc_conditions = [False] * len(TFs[direction])
    for direction in directions:
        direction = str(direction)
        # Zero out the transfer function columns where conditions are met (iterate over range 1 to 3)
        for i in range(1, 4):
            for prefix in ['BRC', 'CHD']:
                # Apply the combined elevation and brc_ID conditions
                condition = elevation_conditions[i-1] | brc_conditions[i-1]    
                # Filter columns that match the prefix and brace index i
                columns_to_zero = TFs[direction].filter(like=f'{prefix}{i}_HS').columns
                # Set all the matching columns to 0 where condition is True
                TFs[direction].loc[condition, columns_to_zero] = 0

    # Group by the joint and thickness columns from the first direction (since it's common for all)
    grouped_TF = TFs[str(directions[0])].groupby(joint_columns + thickness_columns, dropna=False)
    
    # Create a list to store rows for all Hs and Tp combinations for this joint and direction
    results_list = []
    scf_series_list = []
    # joint_n = 0
    for (joint, type, chord1, brace1, brace2, brace3, chd_THK1, brc_THK1, brc_THK2, brc_THK3), joint_data in grouped_TF:
        within_bounds = joint_data[elevation_columns].apply(lambda col: (col >= min_ele) & (col <= max_ele))
        within_bounds = within_bounds.any().any()  # True if any value is within bounds
        if within_bounds:
            SCFs_row = SCF_df.loc[(SCF_df["joint"] == joint) & (SCF_df["brace1"] == brace1)].iloc[0]
            scf_series = SCFs_row.filter(regex="^SCF")
            # If the check is for "root"
            if check["side"] == "root":
                # Extract R columns (those starting with R)
                R_series = SCFs_row.filter(regex="^R")
                # Multiply corresponding columns: R containing FX with SCF containing FX, and similarly for IPB and OPB
                for suffix in ["FX", "IPB", "OPB"]:
                    # Filter SCF and R columns by suffix
                    scf_filtered = scf_series.filter(regex=f"{suffix}")
                    R_filtered = R_series.filter(regex=f"{suffix}")
                    # Multiply corresponding SCF and R columns and update scf_series directly
                    for col in scf_filtered.index:
                        R_col = R_filtered.index[R_filtered.index.str.contains(suffix)][0]  # Find the matching R column for the suffix
                        scf_series[col] = scf_filtered[col] * R_series[R_col]
            
            scf_series_list.append(scf_series)
            joint_index = joint_data.index
            total_damage = np.zeros(len(tf_columns))
            # joint_n += 1
            results_row = {
                'joint': joint,
                'type': type,
                'chord1': chord1,
                'brace1': brace1,
                'brace2': brace2,
                'brace3': brace3,
                'chd_OD1': joint_data["chd_OD1"].iloc[0],
                'chd_THK1': joint_data["chd_THK1"].iloc[0],
                'brc_OD1': joint_data["brc_OD1"].iloc[0],
                'brc_THK1': joint_data["brc_THK1"].iloc[0],
                'brc_OD2': joint_data["brc_OD2"].iloc[0],
                'brc_THK2': joint_data["brc_THK2"].iloc[0],
                'brc_OD3': joint_data["brc_OD3"].iloc[0],
                'brc_THK3': joint_data["brc_THK3"].iloc[0],
                'Code': check["Code"],
                'Code Check': check["type"],
                'side': check["side"],
                'Environment': check["Environment"], 
                'safety_factor': safety_factor,
                'SN_Curve': check["SN_Curve"]
            }
            # Create a dictionary to store SCFs using dictionary comprehension
            thickness_keys = ['CHD1', 'CHD2', 'CHD3', 'BRC1', 'BRC2', 'BRC3']
            thickness_values = [chd_THK1, chd_THK1, chd_THK1, brc_THK1, brc_THK2, brc_THK3]

            thickness_SCF = {
                key: thickness_effect(value, get_t_ref(SN_Curve["Code"].values[0], check["type"]), SN_Curve["k"].values[0])
                for key, value in zip(thickness_keys, thickness_values)
            }
        
            # Iterate over each direction
            for direction in directions:
                print(f"Analyzing Joint {joint} Type {type} Direction {direction}")
                direction = str(direction)
                # Load Hs and Tp columns and their associated percentages (fractions) from scatter_diagrams
                hs_series = scatter_diagrams[direction]['HS'].values.astype(float)
                tp_columns = scatter_diagrams[direction].columns[1:].values.astype(float)
                scatter_data = scatter_diagrams[direction].iloc[:, 1:].values

                wave_periods = TFs[direction].loc[joint_index, "wave_period"]
                wave_periods = pd.to_numeric(wave_periods, errors="coerce")
                
                H_w = TFs[direction].loc[joint_index, tf_columns].copy(deep=True)
                # Apply Thickness Correction to the relevant columns in H_w
                for column in tf_columns:  # Loop through all transfer function columns
                        H_w[column] *= thickness_SCF[column[:4]]  # Multiply the column by its corresponding SCF
        
                moments = moment_integration_vect(H_w.values, wave_periods.values, hs_series, tp_columns, spectrum_gamma)
                RMS = moments[0] ** 0.5
                Tz = 2 * np.pi * (moments[0] / moments[1]) ** 0.5
                n_cycles = scatter_data[:, :, np.newaxis] * Life * T_year/ Tz
                D_m_5 = n_cycles * sn_curve_integration_vect(0, s_inflection, log_a2, m2, RMS)
                D_m_3 = n_cycles * sn_curve_integration_vect(s_inflection, 2000, log_a1, m1, RMS)
                total_damage += np.sum((D_m_5+D_m_3), axis=(0, 1)) 
                a = 1
            
            for i, tf_column in enumerate(tf_columns):
                results_row[f"{tf_column}_Damage"] = total_damage[i]
            
            # Convert the list of rows into a DataFrame
            results_list.append(results_row) 
            # if joint_n == 50:
            # break

    results_df = pd.DataFrame(results_list)
    scf_series_df = pd.DataFrame(scf_series_list)
    results_df = results_df.reset_index(drop=True)
    scf_series_df = scf_series_df.reset_index(drop=True)
    print("Results DF Index:", results_df.index)
    print("SCF Series DF Index:", scf_series_df.index)
    # # Concatenate the results_df and scf_series_df side by side
    final_df = pd.concat([results_df, scf_series_df], axis=1)

    # print(results_df)
    
    # Add a Total_Damage column to results_df
    brace_damage_columns = [col for col in final_df.columns if (col.endswith('_Damage') & col.startswith('BRC'))]
    chord_damage_columns = [col for col in final_df.columns if (col.endswith('_Damage') & col.startswith('CHD'))]


    final_df['Max_BRC_Damage'] = final_df[brace_damage_columns].max(axis=1)
    final_df['Max_CHD_Damage'] = final_df[chord_damage_columns].max(axis=1)
   
    return final_df

def calc_inline_damages_vect(directions, TFs_original, scatter_diagrams, check, SN_Curves):    
    results_df = pd.DataFrame()
    TFs = {key: df.copy(deep=True) for key, df in TFs_original.items()}
    safety_factor = check["safety_factor"]
    Life = check["Life"]
    spectrum_gamma = check["spectrum_gamma"]
    inline_type = check["type"]
    # TF columns 
    if inline_type == "inline": 
        joint_columns = ["joint", "memb1", "memb2"]
        OD_columns = ["memb1_OD", "memb2_OD"]
        THK_columns = ["memb1_THK", "memb2_THK"]
        prefixes = ["memb1", "memb2"]
        SFC_columns = ["SCFtoe", "SCFroot"] 
        tf_columns = np.array([f"{prefix}_HS{j}" for prefix in prefixes for j in range(1, 9)])
    else:
        joint_columns = ["joint", "cone", "tubular"]
        OD_columns = ["cone_OD", "tub_OD"]
        THK_columns = ["cone_THK", "tub_THK"]
        prefixes = ["cone", "tub"]
        SFC_columns = ["SCFcone", "SCFtubular"] 
        tf_columns = np.array([f"{prefix}_HS{j}" for prefix in prefixes for j in range(1, 9)]) 

    SN_Curve_row = (SN_Curves["SN_Curve"] == check["SN_Curve"]) & (SN_Curves["Environment"] == check["Environment"])

    SN_Curve = SN_Curves[SN_Curve_row]
    s_inflection = SN_Curve["s_inflection"].values[0] / safety_factor
    n_inflection = SN_Curve["Inflection"].values[0]
    m1 = SN_Curve["m1"].values[0]
    m2 = SN_Curve["m2"].values[0]

    log_a1 = np.log10(n_inflection) + m1 * np.log10(s_inflection) 
    log_a2 = np.log10(n_inflection) + m2 * np.log10(s_inflection) 
    T_year = 365 * 24 * 60 * 60 
    
    # Define the elevation range
    min_ele = check["Elevation_min"]
    max_ele = check["Elevation_max"]
    
    # DEFINE CURRENT SCOPE BY [TARGET ELEVATION, ID < 600 FOR ROOT CHECKS]
        # Identify rows where elevation condition is met for each column
    direction = str(directions[0])
    elevation_condition = (TFs[direction][f'Elevation'] < min_ele) | (TFs[direction][f'Elevation'] > max_ele) 

    for direction in directions:
        direction = str(direction)
        # Zero out the transfer function columns where conditions are met (iterate over range 1 to 3)
        # Filter columns that match the prefix and brace index i
        columns_to_zero = TFs[direction].filter(regex='_HS').columns
        # Set all the matching columns to 0 where condition is True
        TFs[direction].loc[elevation_condition, columns_to_zero] = 0

    # Group by the joint and thickness columns from the first direction (since it's common for all)
    grouped_TF = TFs[str(directions[0])].groupby(joint_columns, dropna=False)
    
    # Create a list to store rows for all Hs and Tp combinations for this joint and direction
    results_list = []
    joint_n = 0
    # Iterate over the grouped TF data
    for (joint, memb1, memb2), joint_data in grouped_TF:
        within_bounds = (joint_data["Elevation"] >= min_ele) | (joint_data["Elevation"] <= max_ele)
        within_bounds = within_bounds.any().any()
        if within_bounds:
            joint_index = joint_data.index
            total_damage = np.zeros(len(tf_columns))
            joint_n += 1

            results_row = {
                'joint': joint,
                joint_columns[1]: memb1,
                joint_columns[2]: memb2,
                f'{prefixes[0]}_OD': joint_data[OD_columns[0]].iloc[0],
                f'{prefixes[1]}_OD': joint_data[OD_columns[1]].iloc[0],
                f'{prefixes[0]}_THK': joint_data[THK_columns[0]].iloc[0],
                f'{prefixes[1]}_THK': joint_data[THK_columns[1]].iloc[0],
                'elevation': joint_data["Elevation"].iloc[0],
                'Code': check["Code"],
                'Code Check': check["type"],
                'side': check["side"],
                'Environment': check["Environment"], 
                'safety_factor': safety_factor,
                'SN_Curve': check["SN_Curve"],
                SFC_columns[0]: joint_data[SFC_columns[0]].iloc[0],
                SFC_columns[1]: joint_data[SFC_columns[1]].iloc[0]
            }
            # Create a dictionary to store SCFs using dictionary comprehension
            thickness_values = [joint_data[THK_columns[0]].iloc[0], joint_data[THK_columns[1]].iloc[0]]

            thickness_SCF = {
                key: thickness_effect(value, get_t_ref(SN_Curve["Code"].values[0], check["type"]), SN_Curve["k"].values[0])
                for key, value in zip(prefixes, thickness_values)
            }
        
            # Iterate over each direction
            for direction in directions:
                check_type = check["type"]
                side = check["side"]
                print(f"Analyzing Joint {joint} Type {check_type} Side {side}  Direction {direction}")
                direction = str(direction)
                # Load Hs and Tp columns and their associated percentages (fractions) from scatter_diagrams
                hs_series = scatter_diagrams[direction]['HS'].values.astype(float)
                tp_columns = scatter_diagrams[direction].columns[1:].values.astype(float)
                scatter_data = scatter_diagrams[direction].iloc[:, 1:].values
                wave_periods = TFs[direction].loc[joint_index, "wave_period"] 
                wave_periods = pd.to_numeric(wave_periods, errors="coerce")
                H_w = TFs[direction].loc[joint_index, tf_columns].copy(deep=True)
                # Apply Thickness Correction to the relevant columns in H_w
                for column in tf_columns:  # Loop through all transfer function columns
                        H_w[column] *= thickness_SCF[column.split("_")[0]]  # Multiply the column by its corresponding SCF
        
                moments = moment_integration_vect(H_w.values, wave_periods.values, hs_series, tp_columns, spectrum_gamma)
                RMS = moments[0] ** 0.5
                Tz = 2 * np.pi * (moments[0] / moments[1]) ** 0.5
                n_cycles = scatter_data[:, :, np.newaxis] * Life * T_year/ Tz
                D_m_5 = n_cycles * sn_curve_integration_vect(0, s_inflection, log_a2, m2, RMS)
                D_m_3 = n_cycles * sn_curve_integration_vect(s_inflection, 2000, log_a1, m1, RMS)
                total_damage += np.sum((D_m_5+D_m_3), axis=(0, 1))
                # break
        
            for i, tf_column in enumerate(tf_columns):
                results_row[f"{tf_column}_Damage"] = total_damage[i]
            
            # Convert the list of rows into a DataFrame
            results_list.append(results_row) 

    results_df = pd.DataFrame(results_list)
    results_df = results_df.reset_index(drop=True)

    
    # Add a Total_Damage column to results_df
    memb1_damage_columns = [col for col in results_df.columns if (col.endswith('_Damage') & col.startswith(prefixes[0]))]
    memb2_damage_columns = [col for col in results_df.columns if (col.endswith('_Damage') & col.startswith(prefixes[1]))]


    results_df[f'Max_{prefixes[0]}_Damage'] = results_df[memb1_damage_columns].max(axis=1)
    results_df[f'Max_{prefixes[1]}_Damage'] = results_df[memb2_damage_columns].max(axis=1)
   
    return results_df


def IRS_SCF_calculation(df):
    # Group by 'joint' and 'brace' columns
    # print(df.columns.tolist())
    df = df.fillna(0)
    grouped = df.groupby(['joint','type','brace'])



    # Iterate through each group
    for (joint, type, brace), group_df in grouped:
        # Create boolean masks where 'brace' matches 'brace1', 'brace2', or 'brace3'
        original_df = group_df.copy()
        # Create a vector to track which suffix (1, 2, or 3) corresponds to each matched brace
        chd_OD = group_df['chd_OD1'].iloc[0]
        chd_THK  = group_df['chd_THK1'].iloc[0]
        web_height = group_df['web_height'].iloc[0]
        web_THK = group_df['web_THK'].iloc[0]
        flange_width = group_df['flange_width'].iloc[0] 
        flange_THK = group_df['flange_THK'].iloc[0] 
        eff_chord = group_df['eff_chord'].iloc[0] * 1000
        n_rings = group_df['n_rings'].iloc[0] 
        p = group_df['spacing'].iloc[0]  


        'Target Brace'
        if (type == "T") or (type == "T-X"):
            brc = 1
        else:
            angles = group_df[['angle1', 'angle2', 'angle3']].iloc[0].to_numpy()

            brace_columns = ['brace1', 'brace2', 'brace3']
            # Find which column matches the brace value
            brc = None
            for i, col in enumerate(['brace1', 'brace2', 'brace3']):
                if group_df[col].iloc[0] == brace:
                    brc = i + 1
                    break

            print(angles)
            print(group_df[['angle1', 'angle2', 'angle3']].dtypes)
            print(f"Brace Number: {brc}")
        if brace == "6216-6239":
            a = 1
 
        
        brc_OD1 = group_df[f'brc_OD{brc}'].iloc[0]
        brc_THK1 = group_df[f'brc_THK{brc}'].iloc[0]
        θ = group_df[f'angle{brc}'].iloc[0]

        τ = brc_THK1/chd_THK
        β = brc_OD1/chd_OD
        γ = chd_OD/(2*chd_THK)
        α = 2*eff_chord/chd_OD

        d_p = brc_OD1/np.sin((np.radians(θ)))
        be = min(1.56*chd_THK*γ**0.5,eff_chord,p) 
        Rtau = n_rings*web_THK/chd_THK
        
        Ac = chd_THK*d_p + n_rings*(web_height*web_THK + flange_width*flange_THK)
        Ar = be*chd_THK + web_height*web_THK + flange_width*flange_THK

        y_cgc = n_rings*((web_height*web_THK*(web_height + chd_THK)/2) + (flange_width*flange_THK*(web_height + chd_THK/2 + flange_THK/2)))/Ac
        y_cgr = (be*chd_THK*(web_height + flange_THK + chd_THK/2) + web_height*web_THK*(web_height/2 + flange_THK) + (flange_width*flange_THK**2)/2)/Ar

        I_OC = (((d_p*chd_THK**3)/12) + (((web_height*web_THK*(web_height + chd_THK)**2)/4) + (flange_width*flange_THK*(web_height + chd_THK/2 + flange_THK/2)**2) + ((web_THK*web_height**3)/12) + ((flange_width*flange_THK**3)/12))*n_rings) - Ac*y_cgc**2
        I_OR = (((be*chd_THK**3)/12) + (web_height*web_THK*(web_height/2 + flange_THK)**2) + (be*chd_THK*(web_height + flange_THK + chd_THK/2)**2) + ((web_height**3)*web_THK)/12 + (flange_width*flange_THK**3)/3) - (Ar*y_cgr**2)
        I_mod = (I_OR/y_cgr)/((chd_THK**2)*(chd_OD/6))
        
        Rtau_ = 2*web_THK/chd_THK
        Ac_ = chd_THK*d_p + 2*(web_height*web_THK + flange_width*flange_THK)
        y_cgc_ = 2*((web_height*web_THK*(web_height + chd_THK)/2) + (flange_width*flange_THK*(web_height + chd_THK/2 + flange_THK/2)))/Ac
        I_OC_ = (((d_p*chd_THK**3)/12) + (((web_height*web_THK*(web_height + chd_THK)**2)/4) + (flange_width*flange_THK*(web_height + chd_THK/2 + flange_THK/2)**2) + ((web_THK*web_height**3)/12) + ((flange_width*flange_THK**3)/12))*2) - Ac*y_cgc**2

        K_1 = Ar/(chd_OD*chd_THK)
        K_2 = (((12*I_OC)/d_p)**(1/3))/chd_THK
        K_2_ = (((12*I_OC_)/d_p)**(1/3))/chd_THK
        
        if n_rings == 1:
            CHD_SDL_FX = np.exp( - 0.3*(p/d_p)**0.2)
            BRC_SDL_FX = np.exp( - 0.2*(p/d_p)**0.2)
            IRS_FX = np.exp( - 0.1*(p/d_p)**2)
            CHD_SDL_OPB = np.exp( - 1.4*(p/d_p)**0.5)
            BRC_SDL_OPB = np.exp( - 0.3*(p/d_p))
            IRS_OPB = np.exp( - 0.25*(p/d_p)**2)
            IPB_CRN = max(p/d_p,0.65)
        elif n_rings == 2:
            CHD_SDL_FX = 2*np.exp( - (p/d_p)**0.2)
            BRC_SDL_FX = 2*np.exp( - 0.5*(p/d_p)**0.2)
            IRS_FX = 2*np.exp( - 0.25*(p/d_p)**2)
            CHD_SDL_OPB = 2*np.exp( - 1.4*(p/d_p)**0.5)
            BRC_SDL_OPB = 2*np.exp( - (p/d_p))
            IRS_OPB = 2*np.exp( - 0.25*(p/d_p)**2)
            IPB_CRN = max((n_rings - 1)*(p/d_p),0.65)
        elif n_rings == 3:
            CHD_SDL_FX = 1 + 2*np.exp( - 2.5*(p/d_p)**0.2)
            BRC_SDL_FX = 1 + 2*np.exp( - 3*(p/d_p)**0.2)
            IRS_FX = 1 + 2*np.exp( - 3*(p/d_p)**2)
            CHD_SDL_OPB = 1 + 2*np.exp( - 3*(p/d_p)**0.5)
            BRC_SDL_OPB = 1 + 2*np.exp( - 9*(p/d_p))
            IRS_OPB = 1 + 2*np.exp( - 4*(p/d_p)**2)
            IPB_CRN = max((n_rings - 1)*(p/d_p),0.65)
        elif n_rings == 4:
            CHD_SDL_FX = 2*np.exp( - (p/d_p)**0.2) + 2*np.exp( - 5*(p/d_p)**0.2)
            BRC_SDL_FX = 2*np.exp( - 0.5*(p/d_p)**0.2) + 2*np.exp( - 6*(p/d_p)**0.2)
            IRS_FX = 2*np.exp( - 0.25*(p/d_p)**2) + 2*np.exp( - 30*(p/d_p)**2)
            CHD_SDL_OPB = 2*np.exp( - 1.4*(p/d_p)**0.5) + 2*np.exp( - 6*(p/d_p)**0.5)
            BRC_SDL_OPB = 2*np.exp( - (p/d_p)**0.2) + 2*np.exp( - 18*(p/d_p))
            IRS_OPB = 2*np.exp( - 0.25*(p/d_p)**2) + 2*np.exp( - 30*(p/d_p)**2)
            IPB_CRN = max((n_rings - 1)*(p/d_p),0.6)
        
        T_ratios = {}
    
        T_BRC_CRN_FX_EQ = (τ**0.8)*(γ**0.5)*np.sin(np.radians(θ))*(4*β - 3*(β**2) - 0.5)
        T_BRC_SDL_FX_EQ = (0.12/((τ**0.25)*BRC_SDL_FX))*(10/K_2 + 1/Rtau)
        T_BRC_CRN_IPB_EQ = 0.8 + (IPB_CRN*β*γ/20)
        T_BRC_SDL_OPB_EQ = (0.36/((BRC_SDL_OPB**0.5)*(τ*β*γ)**0.3))*((3/K_2) + 1/(Rtau**(1.1 - (n_rings - 1)*p/d_p)))
        T_CHD_CRN_FX_EQ = 1.5*β**0.5
        T_CHD_SDL_FX_EQ = (0.1/((τ**0.2)*CHD_SDL_FX))*(5.5/K_2 + 1/Rtau)
        T_CHD_CRN_IPB_EQ = (0.1/IPB_CRN)*((β*γ/τ)**0.2)*(3.7/(K_2_**0.35) + 1/Rtau_)
        T_CHD_SDL_OPB_EQ = (1/(8*CHD_SDL_OPB*(τ*β*γ)**0.2))*((6/K_2) + 1)
        T_IRS_FX_EQ = (τ*0.3*(γ**0.5)*((np.sin(np.radians(θ)))**0.5)/IRS_FX)*(((6*β**0.15)/I_mod) + ((β**0.5)/K_1))
        T_IRS_IPB_EQ = 0.73*τ*β*((np.sin(np.radians(θ)))**0.5)*(IPB_CRN**0.5)*(3/I_mod + 1/K_1)
        T_IRS_OPB_EQ = ((0.15*τ*β*(γ**0.75)*((np.sin(np.radians(θ)))**0.5))/IRS_OPB)*(8/I_mod + (β**0.5)/K_1)
        
        T_BRC_CRN_FX_DEV = 0
        T_BRC_SDL_FX_DEV = 0.05
        T_BRC_CRN_IPB_DEV= 0
        T_BRC_SDL_OPB_DEV = 0.06
        T_CHD_CRN_FX_DEV = 0
        T_CHD_SDL_FX_DEV = 0.05
        T_CHD_CRN_IPB_DEV = 0.07
        T_CHD_SDL_OPB_DEV = 0.05
        T_IRS_FX_DEV = 0.14
        T_IRS_IPB_DEV = 0.155
        T_IRS_OPB_DEV = 0.186

        T_ratios["T_BRC_CRN_FX_RAT"] = T_BRC_CRN_FX_EQ + 1.28*T_BRC_CRN_FX_DEV
        T_ratios["T_BRC_SDL_FX_RAT"] = T_BRC_SDL_FX_EQ + 1.28*T_BRC_SDL_FX_DEV
        T_ratios["T_BRC_CRN_IPB_RAT"] = T_BRC_CRN_IPB_EQ + 1.28*T_BRC_CRN_IPB_DEV
        T_ratios["T_BRC_SDL_OPB_RAT"] = T_BRC_SDL_OPB_EQ + 1.28*T_BRC_SDL_OPB_DEV
        T_ratios["T_CHD_CRN_FX_RAT"] = T_CHD_CRN_FX_EQ + 1.28*T_CHD_CRN_FX_DEV
        T_ratios["T_CHD_SDL_FX_RAT"] = T_CHD_SDL_FX_EQ + 1.28*T_CHD_SDL_FX_DEV
        T_ratios["T_CHD_CRN_IPB_RAT"] = T_CHD_CRN_IPB_EQ + 1.28*T_CHD_CRN_IPB_DEV
        T_ratios["T_CHD_SDL_OPB_RAT"] = T_CHD_SDL_OPB_EQ + 1.28*T_CHD_SDL_OPB_DEV
        SCF_T_IRS_FX_DEV = T_IRS_FX_EQ*(1 + 2*T_IRS_FX_DEV)
        SCF_T_IRS_IPB = T_IRS_IPB_EQ*(1 + 2*T_IRS_IPB_DEV)
        SCF_T_IRS_OPB = T_IRS_OPB_EQ*(1 + 2*T_IRS_OPB_DEV)

            
        if ("K" in type):
            K_ratios = {}
            other_brcs = [x for x in [1, 2, 3] if x != brc]
            
            g1 = group_df['gap1'].iloc[0]
            g2 = group_df['gap2'].iloc[0]

            if brc == 1:
                gap = g1
                brc2 = 2
            elif brc == 3:
                gap = g2
                brc2 = 2
            elif brc == 2:
                gap, brc2 = (g1, 1) if g1 <= g2 else (g2, 3)
                
            ζ = gap/chd_OD

            θ2 = group_df[f'angle{brc2}'].iloc[0]

            K_BRC_CRN_FX_ONE_EQ = 1.6*β**2 - 0.4*β + 0.8
            K_BRC_SDL_FX_ONE_EQ = (0.24/BRC_SDL_FX)*((β/(τ*γ))**0.25)*(7/(K_2**0.5) + 1/Rtau)
            K_BRC_CRN_IPB_ONE_EQ = 0.8 + (IPB_CRN*β*γ/20)
            K_BRC_SDL_OPB_ONE_EQ = (0.36/((BRC_SDL_OPB**0.5)*(τ*β*γ)**0.3))*((3/K_2) + 1/(Rtau**(1.1 - (n_rings - 1)*p/d_p)))*1.1
            K_CHD_CRN_FX_ONE_EQ = 1.2*β**0.3
            K_CHD_SDL_FX_ONE_EQ = (0.1/((τ**0.2)*CHD_SDL_FX))*(5.5/K_2 + 1/Rtau)
            K_CHD_CRN_IPB_ONE_EQ = (0.1/IPB_CRN)*((β*γ/τ)**0.2)*(3.7/(K_2_**0.35) + 1/Rtau_)*1.2
            K_CHD_SDL_OPB_ONE_EQ = (1/(8*CHD_SDL_OPB*(τ*β*γ)**0.2))*((6/K_2) + 1)
            K_IRS_FX_ONE_EQ = (τ*0.3*(γ**0.5)*((np.sin(np.radians(θ)))**(0.5 + np.sin(np.radians(θ2))))/IRS_FX)*(((6*β**0.15)/I_mod) + ((β**0.5)/K_1))
            K_IRS_IPB_ONE_EQ = 0.73*τ*β*((np.sin(np.radians(θ)))**0.5)*(IPB_CRN**0.5)*(3/I_mod + 1/K_1)
            K_IRS_OPB_ONE_EQ = ((0.15*τ*β*(γ**0.75)*((np.sin(np.radians(θ)))**(0.5 + np.sin(np.radians(θ2)))))/IRS_OPB)*(8/I_mod + (β**0.5)/K_1)

            K_BRC_CRN_FX_ONE_DEV = 0
            K_BRC_SDL_FX_ONE_DEV = 0.07
            K_BRC_CRN_IPB_ONE_DEV = 0
            K_BRC_SDL_OPB_ONE_DEV = 0.07
            K_CHD_CRN_FX_ONE_DEV = 0
            K_CHD_SDL_FX_ONE_DEV = 0.05
            K_CHD_CRN_IPB_ONE_DEV = 0.09
            K_CHD_SDL_OPB_ONE_DEV = 0.06
            K_IRS_FX_ONE_DEV = 0.253
            K_IRS_IPB_ONE_DEV = 0.214
            K_IRS_OPB_ONE_DEV = 0.218

            K_ratios[f"K{brc}{brc2}_BRC_CRN_FX_ONE_RAT"] = K_BRC_CRN_FX_ONE_EQ + 1.28*K_BRC_CRN_FX_ONE_DEV
            K_ratios[f"K{brc}{brc2}_BRC_SDL_FX_ONE_RAT"] = K_BRC_SDL_FX_ONE_EQ + 1.28*K_BRC_SDL_FX_ONE_DEV
            K_ratios[f"K{brc}{brc2}_BRC_CRN_IPB_ONE_RAT"] = K_BRC_CRN_IPB_ONE_EQ + 1.28*K_BRC_CRN_IPB_ONE_DEV
            K_ratios[f"K{brc}{brc2}_BRC_SDL_OPB_ONE_RAT"] = K_BRC_SDL_OPB_ONE_EQ + 1.28*K_BRC_SDL_OPB_ONE_DEV
            K_ratios[f"K{brc}{brc2}_CHD_CRN_FX_ONE_RAT"] = K_CHD_CRN_FX_ONE_EQ + 1.28*K_CHD_CRN_FX_ONE_DEV
            K_ratios[f"K{brc}{brc2}_CHD_SDL_FX_ONE_RAT"] = K_CHD_SDL_FX_ONE_EQ + 1.28*K_CHD_SDL_FX_ONE_DEV
            K_ratios[f"K{brc}{brc2}_CHD_CRN_IPB_ONE_RAT"] = K_CHD_CRN_IPB_ONE_EQ + 1.28*K_CHD_CRN_IPB_ONE_DEV
            K_ratios[f"K{brc}{brc2}_CHD_SDL_OPB_ONE_RAT"] = K_CHD_SDL_OPB_ONE_EQ + 1.28*K_CHD_SDL_OPB_ONE_DEV
            

            SCF_K_IRS_FX_ONE = K_IRS_FX_ONE_EQ*(1 + 2*K_IRS_FX_ONE_DEV)
            SCF_K_IRS_IPB_ONE = K_IRS_IPB_ONE_EQ*(1 + 2*K_IRS_IPB_ONE_DEV)
            SCF_K_IRS_OPB_ONE = K_IRS_OPB_ONE_EQ*(1 + 2*K_IRS_OPB_ONE_DEV)


            K_BRC_CRN_FX_BALANCED_EQ = 0.5*β**2 + 0.8
            K_BRC_SDL_FX_BALANCED_EQ = K_BRC_SDL_FX_ONE_EQ*1.4
            K_BRC_CRN_IPB_BALANCED_EQ = 0.8 + (IPB_CRN*β*γ/20)
            K_BRC_SDL_OPB_BALANCED_EQ = (0.36/((BRC_SDL_OPB**0.5)*(τ*β*γ)**0.3))*((3/K_2) + 1/(Rtau**(1.1 - (n_rings - 1)*p/d_p)))*(0.75/β**0.2)
            K_CHD_CRN_FX_BALANCED_EQ = 1.2*β**0.3
            K_CHD_SDL_FX_BALANCED_EQ = (0.05/((τ**0.2)*CHD_SDL_FX**2))*(10/(K_2**0.35) + 1/Rtau)
            K_CHD_CRN_IPB_BALANCED_EQ = (0.1/IPB_CRN)*((β*γ/τ)**0.2)*(3.7/(K_2_**0.35) + 1/Rtau_)*max(1.15*(τ/np.sin(np.radians(max(θ,θ2))))**0.6,1.2)
            K_CHD_SDL_OPB_BALANCED_EQ = (1/(8*CHD_SDL_OPB*(τ*β*γ)**0.2))*((6/K_2) + 1)*0.8*((np.sin(np.radians(θ))/(np.sin(np.radians(θ2))))**0.5)
            K_IRS_FX_BALANCED_EQ = (τ*0.3*(γ**0.5)*((np.sin(np.radians(θ)))**0.5)/IRS_FX)*(((6*β**0.15)/I_mod) + ((β**0.5)/K_1))*2*τ*(β**0.3)*(IRS_FX**0.5)/(γ**0.3)
            K_IRS_IPB_BALANCED_EQ = 0.73*τ*β*((np.sin(np.radians(θ)))**0.5)*(IPB_CRN**0.5)*(3/I_mod + 1/K_1)
            K_IRS_OPB_BALANCED_EQ = ((0.15*τ*β*(γ**0.75)*((np.sin(np.radians(θ)))**0.5))/IRS_OPB)*(8/I_mod + (β**0.5)/K_1)*0.8

            K_BRC_CRN_FX_BALANCED_DEV = 0
            K_BRC_SDL_FX_BALANCED_DEV = 0.1
            K_BRC_CRN_IPB_BALANCED_DEV = 0
            K_BRC_SDL_OPB_BALANCED_DEV = 0.06
            K_CHD_CRN_FX_BALANCED_DEV = 0
            K_CHD_SDL_FX_BALANCED_DEV = 0.07
            K_CHD_CRN_IPB_BALANCED_DEV = 0.09
            K_CHD_SDL_OPB_BALANCED_DEV = 0.05
            K_IRS_FX_BALANCED_DEV = 0.233
            K_IRS_IPB_BALANCED_DEV = 0.214
            K_IRS_OPB_BALANCED_DEV = 0.246

            K_ratios[f"K{brc}{brc2}_BRC_CRN_FX_BALANCED_RAT"] = K_BRC_CRN_FX_BALANCED_EQ + 1.28*K_BRC_CRN_FX_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_BRC_SDL_FX_BALANCED_RAT"] = K_BRC_SDL_FX_BALANCED_EQ + 1.28*K_BRC_SDL_FX_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_BRC_CRN_IPB_BALANCED_RAT"] = K_BRC_CRN_IPB_BALANCED_EQ + 1.28*K_BRC_CRN_IPB_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_BRC_SDL_OPB_BALANCED_RAT"] = K_BRC_SDL_OPB_BALANCED_EQ + 1.28*K_BRC_SDL_OPB_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_CHD_CRN_FX_BALANCED_RAT"] = K_CHD_CRN_FX_BALANCED_EQ + 1.28*K_CHD_CRN_FX_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_CHD_SDL_FX_BALANCED_RAT"] = K_CHD_SDL_FX_BALANCED_EQ + 1.28*K_CHD_SDL_FX_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_CHD_CRN_IPB_BALANCED_RAT"] = K_CHD_CRN_IPB_BALANCED_EQ + 1.28*K_CHD_CRN_IPB_BALANCED_DEV
            K_ratios[f"K{brc}{brc2}_CHD_SDL_OPB_BALANCED_RAT"] = K_CHD_SDL_OPB_BALANCED_EQ + 1.28*K_CHD_SDL_OPB_BALANCED_DEV
            SCF_K_IRS_FX_BALANCED = K_IRS_FX_BALANCED_EQ*(1 + 2*K_IRS_FX_BALANCED_DEV)
            SCF_K_IRS_IPB_BALANCED = K_IRS_IPB_BALANCED_EQ*(1 + 2*K_IRS_IPB_BALANCED_DEV)
            SCF_K_IRS_OPB_BALANCED = K_IRS_OPB_BALANCED_EQ*(1 + 2*K_IRS_OPB_BALANCED_DEV)
            
            
        for prefix in ["BRC", "CHD"]:
            for suffix in ["CRN_FX", "SDL_FX", "CRN_IPB", "SDL_OPB"]:
                # Update all T_SCF independent of joint type
                if "T" in type: 
                    t_scf_key = f"T_SCF{brc}_{prefix}_{suffix}"
                    group_df[t_scf_key] = original_df[t_scf_key] * T_ratios[f"T_{prefix}_{suffix}_RAT"]
                if ("X" in type) and ("IPB" not in suffix): 
                    x_scf_key = f"X_SCF{brc}_{prefix}_{suffix}_BALANCED"
                    group_df[x_scf_key] = original_df[x_scf_key] * T_ratios[f"T_{prefix}_{suffix}_RAT"]
                    x_scf_key = f"X_SCF{brc}_{prefix}_{suffix}_ONE"
                    group_df[x_scf_key] = original_df[x_scf_key] * T_ratios[f"T_{prefix}_{suffix}_RAT"]

                if "K" in type:  # Update all IRS ONE SCFs 
                    # Update K_SCF for IRS ONE
                    k_scf_one_key = f"K_SCF{brc}{brc2}_{prefix}_{suffix}_ONE"
                    group_df[k_scf_one_key] = original_df[k_scf_one_key] * K_ratios[f"K{brc}{brc2}_{prefix}_{suffix}_ONE_RAT"]

                    # Determine if we are dealing with bending or axial
                    if "FX" in suffix:  # Update Axial IRS SCFs  
                        k_scf_balanced_key = f"K_SCF{brc}{brc2}_{prefix}_{suffix}_BALANCED"
                        group_df[k_scf_balanced_key] = original_df[k_scf_balanced_key] * K_ratios[f"K{brc}{brc2}_{prefix}_{suffix}_BALANCED_RAT"]
                    else:  # Update Bending IRS SCFs
                        group_df[t_scf_key] = original_df[t_scf_key] * K_ratios[f"K{brc}{brc2}_{prefix}_{suffix}_BALANCED_RAT"]

                        


            
        df.update(group_df)


    return df


# Mapping function for the hotspot values
def map_hotspot(hs_value):
    if hs_value in ['HS1', 'HS5']:
        return 'CRN'
    elif hs_value in ['HS3', 'HS7']:
        return 'SDL'
    else:
        return 'DIA'
    
# def max_INL_damage_columns(row, keys):
#     hs_values = [1, 2, 3, 4, 5, 6, 7, 8]  # HS identifiers

#     result = {}
#     # Dynamically generate memb1/column names for memb1 side damage
#     memb1_columns = [f"{keys[0]}_HS{hs}_Damage" for hs in hs_values]
#     memb2_columns = [f"{keys[1]}_HS{hs}_Damage" for hs in hs_values]
    
#     # Find the column name where the maximum damage occurs 
#     max_memb1_col = row[memb1_columns].idxmax() if row[memb1_columns].notna().any() else None
#     max_memb2_col = row[memb2_columns].idxmax() if row[memb2_columns].notna().any() else None

#     if max_memb1_col:  # Same as memb2
#         # Extract the HS part and map it
#         max_memb1_hs = max_memb1_col.split("_")[1]
#         result[f'{keys[0]}'] = [map_hotspot(max_memb1_hs), row[max_memb1_col]]
#         max_memb2_hs = max_memb2_col.split("_")[1]
#         result[f'{keys[1]}'] = [map_hotspot(max_memb2_hs), row[max_memb2_col]]
#     else:
#         result[f'{keys[0]}'] = None
#         result[f'{keys[1]}'] = None
#     print(result)
#     return result        

def max_damage_columns(row):
    braces = [1, 2, 3]  # Brace identifiers
    hs_values = [1, 2, 3, 4, 5, 6, 7, 8]  # HS identifiers

    result = {}
    # Loop through each brace
    for brace in braces:
        # Dynamically generate brace/column names for brace side damage
        brace_columns = [f"BRC{brace}_HS{hs}_Damage" for hs in hs_values]
        chord_columns = [f"CHD{brace}_HS{hs}_Damage" for hs in hs_values]
        
        # Find the column name where the maximum damage occurs 
        max_brace_col = row[brace_columns].idxmax() if row[brace_columns].notna().any() else None
        max_chord_col = row[chord_columns].idxmax() if row[chord_columns].notna().any() else None

        if max_brace_col:  # Same as chord
            # Extract the HS part and map it
            max_brace_hs = max_brace_col.split("_")[1]
            result[f'brace{brace}'] = [map_hotspot(max_brace_hs), row[max_brace_col]]
            max_chord_hs = max_chord_col.split("_")[1]
            result[f'chord{brace}'] = [map_hotspot(max_chord_hs), row[max_chord_col]]
        else:
            result[f'brace{brace}'] = None
            result[f'chord{brace}'] = None

    print(result)
    return result

def area_ratio (OD1, THK1, OD2, THK2):
    # Calculate inner diameters
    ID1 = OD1 - 2 * THK1
    ID2 = OD2 - 2 * THK2

    # Calculate cross-sectional areas
    area1 = math.pi * (OD1**2 - ID1**2) / 4
    area2 = math.pi * (OD2**2 - ID2**2) / 4

    # Calculate the ratio of areas A2 / A1
    area_ratio =  area1 / area2 
    return area_ratio

def process_inline_optimization(df_damages, original_data, target_columns, D_max, folder, is_cone = False):
    os.makedirs(os.path.join(folder, folder), exist_ok=True)
    final_optimization_df = pd.DataFrame()
    scf_columns = df_damages.filter(like="SCF").columns.tolist()
    damage_columns = df_damages.filter(like="Max").columns.tolist()
    # df_damages['max_SCF'] = df_damages[scf_columns].max(axis=1)
    thk_cols = ["cone_THK", "tub_THK"] if is_cone else ["memb1_THK", "memb2_THK"] 
    # Iterate over each row in df_damages
    for index, row in df_damages.iterrows():
        output_path = os.path.join(folder, folder, f'Optimization_{row["joint"]}.csv')
        original_data_row = original_data[(original_data[target_columns] == row[target_columns].values).all(axis=1)]
        original_data_row["expected_damage"] = row[damage_columns].max()
            # Convert the row (Series) to a DataFrame before concatenating
        row_df = pd.DataFrame([row], columns=df_damages.columns)
        if is_cone:
            row_df["length"] = original_data["length"]
            row_df["cone_L"] = original_data["cone_L"]
            row_df["cone_S"] = original_data["cone_S"]

        final_optimization_df = pd.concat([final_optimization_df, original_data_row], ignore_index=True)

        print(final_optimization_df)
        keys = ["cone", "tub"] if is_cone else ["memb1", "memb2"]
        target_memb = ["cone", "tubular"] if is_cone else ["memb1", "memb2"]
        if is_cone:
            increment_columns = {"length": 0, f'{keys[0]}_THK': 0, f'{keys[1]}_THK': 0}
        else:
            increment_columns = {f'{keys[0]}_THK': 0, f'{keys[1]}_THK': 0}
        
        n_iterations = 0 
        
        optimizations = []
        # while n_iterations < 5:
        while n_iterations < 5:
            n_iterations += 1
            optimization_rows = []
            for col, increment in increment_columns.items():
                new_data_row = original_data_row.copy()   

                # Make sure to access scalar values
                if is_cone:
                    new_data_row['length'] +=  increment_columns['length'] 
                new_data_row[f'{keys[0]}_THK'] +=  increment_columns[f'{keys[0]}_THK'] 
                new_data_row[f'{keys[1]}_THK'] +=  increment_columns[f'{keys[1]}_THK'] 
                if col == "length":
                    new_data_row[col] +=  100   # Otherwise, increment
                else:
                    new_data_row[col] +=  5   # Otherwise, increment

                if is_cone:
                    side = "L" if original_data_row["joint"].iloc[0] == original_data_row["cone"].iloc[0].split("-")[0] else "S"
                
                memb_area_ratios = []
                # Adjust OD for inner flushed members               
                for i in [0, 1]:
                    if (new_data_row[f'{keys[i]}_THK'].iloc[0] != original_data_row[f'{keys[i]}_THK'].iloc[0]):
                        # only adjust if there has been a thickness increase
                        if new_data_row[f'{keys[i]}_THK'].iloc[0] > original_data_row[f'{keys[i]}_THK'].iloc[0]: 
                            # If on Large Side of Cone or Non-Leg Inline Weld
                            if (("cone" in keys[i]) and (side == "L")):
                                new_data_row["cone_L"] = original_data_row["cone_L"] + 2 * (new_data_row[f'{keys[i]}_THK'] - original_data_row[f'{keys[i]}_THK'] ) 
                            elif (("cone" not in keys[i]) and (original_data_row[f"{keys[i]}_group"].iloc[0][:1] != "L")):
                                new_data_row[f'{keys[i]}_OD'] = original_data_row[f'{keys[i]}_OD'] + 2 * (new_data_row[f'{keys[i]}_THK'] - original_data_row[f'{keys[i]}_THK'] ) 
                    if is_cone:
                        OD_tag = "cone_L" if side == "L" else "cone_S"
                        memb_area_ratios.append(area_ratio(original_data_row[OD_tag], original_data_row[f'{keys[i]}_THK'], new_data_row[OD_tag] , new_data_row[f'{keys[i]}_THK']).iloc[0])
                    else:
                        memb_area_ratios.append(area_ratio(original_data_row[f'{keys[i]}_OD'], original_data_row[f'{keys[i]}_THK'], new_data_row[f'{keys[i]}_OD'] , new_data_row[f'{keys[i]}_THK']).iloc[0]) 

                if is_cone:
                    scf1, scf2 = calc_cone_SCF(side, new_data_row['cone_L'].iloc[0], new_data_row['cone_S'].iloc[0], new_data_row['cone_THK'].iloc[0], new_data_row["length"].iloc[0], new_data_row['tub_OD'].iloc[0], new_data_row['tub_THK'].iloc[0])
                else:
                    scf1, scf2 = calc_inline_SCF(new_data_row["memb1_OD"].iloc[0], new_data_row["memb1_THK"].iloc[0], new_data_row["memb2_OD"].iloc[0], new_data_row["memb2_THK"].iloc[0]) 
                
                new_data_row[scf_columns] = [scf1, scf2] 
                    # print(scf_toe, original_data_row["SCFtoe"].iloc[0], scf_root,  original_data_row["SCFroot"].iloc[0])
                ratios = np.array([scf1 / original_data_row[scf_columns[0]].iloc[0], scf2 / original_data_row[scf_columns[1]].iloc[0] ]  )
                target_ratio =  (ratios if is_cone else np.array([ratios[0], ratios[1]]))
                damages = row[damage_columns].values

                new_data_row["property_iter"] = col
                new_data_row["ratio0"] = (target_ratio[0] * memb_area_ratios[0])
                new_data_row["ratio1"] = (target_ratio[1] * memb_area_ratios[1])
            
                exp_damages1 = damages[0] * (target_ratio[0] * memb_area_ratios[0]) ** (5)
                exp_damages2 = damages[1] * (target_ratio[1] * memb_area_ratios[1]) ** (5)
                new_data_row["expected_damage"] = max([exp_damages1, exp_damages2])
                optimization_rows.append(new_data_row.iloc[0])
                print(n_iterations, col,max([exp_damages1, exp_damages2]))
                optimizations.append(new_data_row.iloc[0])

            optimization_rows_df = pd.DataFrame(optimization_rows)
            optimization_rows_df = optimization_rows_df.sort_values(by='expected_damage', ascending=False)
            print(n_iterations, optimization_rows_df)
            # Calculate the maximum value between "SCFtubular" and "SCFcone" for each row
            # optimization_rows_df['max_SCF'] = optimization_rows_df[scf_columns].max(axis=1)
            # Sort by the 'max_SCF' column in descending order
            # Select the last row (the one with the highest max_SCF)
            best_option = optimization_rows_df.tail(1)
            print(best_option)
            a = 1
            for col in thk_cols:
                target_col = col.replace("THK", "FY")
                SY_original = material(original_data_row[col], original_data_row[target_col])
                SY_new = yield_strength(best_option[col], SY_original)
                best_option[target_col] = SY_new

            # if (best_option['max_SCF'].iloc[0] < row['max_SCF']) and (new_data_row["expected_damage"].iloc[0]  > D_max):
            final_optimization_df = pd.concat([final_optimization_df, best_option], ignore_index=True)
            if (new_data_row["expected_damage"].iloc[0]  > D_max):
                # print(increment_columns)
                if best_option["property_iter"].iloc[0] == "length":
                    increment_columns[best_option["property_iter"].iloc[0]] += 100                    
                else:
                    increment_columns[best_option["property_iter"].iloc[0]] += 5

                
  
            else:
                n_iterations = 5
        
        optimizations_df = pd.DataFrame(optimizations)
        optimizations_df.to_csv(output_path, index=False)
                
    return final_optimization_df


# Function identifies max chord_OD on either side of brace to determine
# <<<<<<<<<<<<< NOT IN USE 
def get_brace_OD_Max(joint_Id, brace_Id, model, z_max, chord_OD1):
    current_joint_Id = other_item(brace_Id, joint_Id)
    current_joint = model.FindJoint(current_joint_Id)
    curent_member = model.FindMember(brace_Id)
    current_OD = get_geo(curent_member, "OD")
    current_THK = get_geo(curent_member, "THK")
    brace_OD = current_OD.copy()
    Max_OD1 = chord_OD1
    Max_OD2 = 0
    brace_ID = current_OD - 2 * current_THK
    angle_tolerance = 1
    dist_tolerance = 0.1
    angle = 0
    dist = 0
    chord_found = False
    cone_found = False
    while angle_ < angle_tolerance and dist < dist_tolerance and istube:
        members = current_joint.AttachedMembers
        n_members = len(members)
        for next_member in members:
            next_member_OD = get_geo(next_member, "OD")
            next_member_THK = get_geo(next_member, "THK")
            if left(next_member.group.Id, 1) == "L" and left(current_member.group.Id, 1) != "L":
                angle_ = 10  # Reset angle to exit loop
                chord_found = True
                break
            angle = math.degrees(member_angle_min(curent_member, next_member))
            dist = distance.euclidean(current_joint.coord , common_joint_coords(current_joint, next_member))
            istube = next_member.group.Segments[0].IsSimpleTube
            section_ = next_member.group.Segments[0].Section
            if section_:
                shape_ = next_member.group.Segments[0].Section.Shape
            if istube == False and shape_ == "CON":
                angle = 10  # Reset angle to exit loop
                cone_found = True
                break
            if angle < angle_tolerance and dist < dist_tolerance and istube:
                next_joint_id = other_item(next_member.Id, current_joint.Id)
                current_member = next_member
                current_joint = model.FindJoint(next_joint_id)
                current_OD = get_geo(curent_member, "OD")
                current_THK = get_geo(curent_member, "THK")
                break
            elif angle > angle_tolerance and dist < dist_tolerance and istube:
                angle = 10  
                # Other Chord Found
                chord_found = True
                break
    
    if chord_found:
        Max_OD2 = (next_member_OD - current_OD) + brace_OD

    Max_OD = max(Max_OD1, Max_OD2)
    
    return Max_OD





def process_joint_optimization(df_damages, original_SCF, original_data, braces, target_columns, D_max, folder):
    final_optimization_df = pd.DataFrame()
    # Define the file name
    previous_optimization_filename = "Optimized Member Summary.csv"
    group_OD_filename = "MAX_Group_OD.csv"
    # Read the file if it exists
    opti_df = pd.read_csv(previous_optimization_filename) if os.path.exists(previous_optimization_filename) else None 
    group_OD_df = pd.read_csv(group_OD_filename) if os.path.exists(group_OD_filename) else None
    limit_OD_df = pd.read_csv("limit_OD_groups.txt")
    limit_OD_map = dict(zip(limit_OD_df["group_id"], limit_OD_df["OD_max"]))
    if opti_df is not None and group_OD_df is not None:
        # Find rows in `group_OD_df` that match `group` values in `opti_df`
        merged_df = group_OD_df.merge(opti_df, on="group", how="inner")
        
        # Duplicate rows with updated `group` column from `new_group`
        new_rows = merged_df.copy()
        new_rows["group"] = new_rows["new_group"]
        new_rows = new_rows[group_OD_df.columns]  # Ensure new rows match `group_OD_df` structure
        
        # Combine original and new rows into a single DataFrame
        updated_group_OD_df = pd.concat([group_OD_df, new_rows], ignore_index=True)
        
        # Save the updated DataFrame
        updated_group_OD_df.to_csv("MAX_Group_OD.csv", index=False)
        print("Updated MAX_Group_OD.csv has been created.")
        
        # Delete the previous optimization file
        os.remove(previous_optimization_filename)

    # Iterate over each row in df_damages
    for index, row in df_damages.iterrows():
        # Get the original SCF for the current row
        original_SCF_row = original_SCF[(original_SCF[target_columns] == row[target_columns].values).all(axis=1)]
        original_data_row = original_data[(original_data[target_columns] == row[target_columns].values).all(axis=1)]
        final_optimization_df = pd.concat([final_optimization_df, original_data_row.tail(1)], ignore_index=True)
        target_locs = max_damage_columns(row)
        thk_cols = ['chd_THK1', 'chd_THK2', 'brc_THK1', 'brc_THK2', 'brc_THK3']
        i_columns = {'chd_OD1': 0, 'chd_THK1': 0, 'brc_OD1': 0, 'brc_THK1': 0, 'brc_OD2': 0, 'brc_THK2': 0, 'brc_OD3': 0, 'brc_THK3': 0}
        print(row)
        for brc in braces:
            output_path = os.path.join(folder, folder, f'Optimization_{row["joint"]}_{row[f"brace{brc}"]}.csv')
            optimization_rows = []
            print(row[f"brace{brc}"] )
            # if pd.notna(row[f"brace{brc}"]) and row[f"brace{brc}"] == "6216-6239":
            if pd.notna(row[f"brace{brc}"]):
                # Define the columns to increment
                target_memb = [f"chord{brc}", f"brace{brc}"]
                n_iterations = 0
                
                while (n_iterations < 10):
                # while (n_iterations < 10):
                    n_iterations += 1
                    increment_columns = {'chd_THK1': i_columns['chd_THK1'], f'brc_OD{brc}': i_columns[f'brc_OD{brc}'], f'brc_THK{brc}': i_columns[f'brc_THK{brc}'], 'chd_OD1': i_columns['chd_OD1'] }
                    # increment_columns = {'chd_THK1': i_columns['chd_THK1'], f'brc_OD{brc}': i_columns[f'brc_OD{brc}'] }
                    # Loop over each increment column (OD and THK)
                    for col, increment in increment_columns.items():
                        new_data_row = original_data_row.copy(deep=True) 


                        new_data_row['chd_THK1'] = new_data_row['chd_THK1'] + i_columns['chd_THK1']
                        if not new_data_row['chord1_group'].iloc[0].startswith(('L', 'V')):
                            new_data_row['chd_OD1'] = new_data_row['chd_OD1'] + i_columns['chd_OD1']
                        new_data_row['brc_OD1'] = new_data_row['brc_OD1'] + i_columns['brc_OD1']
                        new_data_row['brc_THK1'] = new_data_row['brc_THK1'] + i_columns['brc_THK1']

                        if pd.notna(new_data_row["brace2"].iloc[0]):
                            new_data_row['brc_OD2'] = new_data_row['brc_OD2'] + i_columns['brc_OD2']
                            new_data_row['brc_THK2'] = new_data_row['brc_THK2'] + i_columns['brc_THK2']
                        if pd.notna(new_data_row["brace3"].iloc[0]):
                            new_data_row['brc_OD3'] = new_data_row['brc_OD3'] + i_columns['brc_OD3']
                            new_data_row['brc_THK3'] = new_data_row['brc_THK3'] + i_columns['brc_THK3']

                        # Make sure to access scalar values
                        if 'THK' in col:
                            new_data_row[col] += 5   # Otherwise, increment
                        else:
                            # Increment for non-thickness properties (OD)
                            new_data_row[col] += 50 


                        # # Unlock OD if OD/THK < 18
                        # delta_OD = 0.0
                        # if col == "chd_THK1":
                        #     current_OD = new_data_row['chd_OD1'].iloc[0]
                        #     current_THK = new_data_row['chd_THK1'].iloc[0]
                        #     if current_OD / current_THK < 18:
                        #         target_OD = current_THK * 18
                        #         # Apply OD limit
                        #         new_data_row['chd_OD1'] = min(3000, target_OD)

                        # Apply limits while updating new_data_row
                        new_data_row['chd_OD1'] = min(3000, new_data_row['chd_OD1'].iloc[0] )  # Limit chord OD to 3000
                        new_data_row['chd_THK1'] = min(120, new_data_row['chd_THK1'].iloc[0] , (new_data_row['chd_OD1'].iloc[0]) / 18)  # Limit chord thickness to 120
                        new_data_row['chd_THK1']  = max(new_data_row['chd_THK1'].iloc[0], original_data_row['chd_THK1'].iloc[0]) # Guarantee thickness does not diminish
                        new_data_row['chd_THK1'] = 5 * round(new_data_row['chd_THK1'].iloc[0] / 5)

                        new_data_row['chd_THK2'] = new_data_row['chd_THK1']
                        new_data_row['chd_OD2'] = new_data_row['chd_OD1']

                        if pd.notna(new_data_row[f'brace{brc}'].iloc[0]):
                            group = new_data_row[f'brace{brc}_group'].iloc[0]
                            # Get Max possible OD
                            # Program-learned OD limit (MAX_Group_OD.csv)
                            group_row = group_OD_df.loc[group_OD_df["group"] == group, "max_OD"] if group_OD_df is not None else None  # lookup row or None
                            max_brace_OD = group_row.squeeze() if (group_OD_df is not None and group_row is not None and not group_row.empty) else 2500  # scalar or default

                            # Engineering OD limit (limit_OD_groups.txt)
                            limit_file_max = limit_OD_map.get(group, None)  # lookup engineering limit
                            max_brace_OD = min(max_brace_OD, limit_file_max) if limit_file_max is not None else max_brace_OD  # combine limits

                            
                             # Limit brace thickness to 80
                            new_data_row[f'brc_OD{brc}'] = min(new_data_row['chd_OD1'].iloc[0], max_brace_OD, new_data_row[f'brc_OD{brc}'].iloc[0])  
                            new_data_row[f'brc_THK{brc}'] = min(80, new_data_row[f'brc_THK{brc}'].iloc[0] , (new_data_row[f'brc_OD{brc}'].iloc[0]) / 18) 
                            new_data_row[f'brc_THK{brc}'] = max(new_data_row[f'brc_THK{brc}'].iloc[0], original_data_row[f'brc_THK{brc}'].iloc[0]) # Guarantee thickness does not diminish
                            new_data_row[f'brc_THK{brc}'] = 5 * round(new_data_row[f'brc_THK{brc}'].iloc[0] / 5)
                        # Recalculate the SCF with the incremented values
                        new_SCF = calculate_tubular_scf(new_data_row)
                        
                        damages = []
                        ratios = []
                        brace_ratio = area_ratio(original_data_row[f'brc_OD{brc}'], original_data_row[f'brc_THK{brc}'], new_data_row[f'brc_OD{brc}'] , new_data_row[f'brc_THK{brc}']).iloc[0]
                        # brace_ratio = area_ratio(original_data_row[f'brc_OD{brc}'], original_data_row[f'brc_THK{brc}'], new_data_row[f'brc_OD{brc}'] + increment_columns[f'brc_OD{brc}'] , new_data_row[f'brc_THK{brc}']).iloc[0]
                        damage_estimates = []
                        for i, memb in enumerate(target_memb):
                            if target_locs[memb]:
                                # damage_column = "Max_BRC_Damage" if row["Max_BRC_Damage"] > row["Max_CHD_Damage"] else "Max_CHD_Damage"
                                SCF_columns = []
                                key1 = "CHD" if "chord" in memb else "BRC"
                                key2 = target_locs[memb][0]
                                key3 = "IPB" if key2 == "CRN" else "OPB" if key2 == "SDL" else None
                                print(target_locs, target_locs[memb][1])
                                damages.append(target_locs[memb][1]) 
                                SCF_columns.append(f"T_SCF{brc}_{key1}_{key2}_FX")
                                if key3:
                                    SCF_columns.append(f"T_SCF{brc}_{key1}_{key2}_{key3}")
                                else:
                                    SCF_columns.append(f"T_SCF{brc}_{key1}_{key2}_IPB")
                                    SCF_columns.append(f"T_SCF{brc}_{key1}_{key2}_OPB")
                                
                                # Perform replacement if "DIA" exists in any column
                                if any("DIA" in col for col in SCF_columns):
                                    SCF_columns = [replacement for col in SCF_columns
                                                for replacement in ([col.replace("DIA", "CRN"), col.replace("DIA", "SDL")] if "FX" in col
                                                                    else [col.replace("DIA", "CRN")] if "IPB" in col
                                                                    else [col.replace("DIA", "SDL")] if "OPB" in col else [col])]

                                
                                # Extract the original SCF values from the original_SCF_row for the corresponding columns
                                original_scf_values = original_SCF_row[SCF_columns].values.flatten()
                                # Extract the new SCF values from the current row in new_scf_df
                                new_scf_values = new_SCF[SCF_columns].values.flatten()
                                # Calculate the ratio for each SCF column (new / original)
                                HS_ratios = new_scf_values / original_scf_values  
                                HS_ratios =  HS_ratios * brace_ratio
                                ratios.append(sum(HS_ratios) / len(HS_ratios))
                                # Get the maximum ratio
                                damage_estimates.append( damages[i] * (ratios[i]) ** (5) )
                

                        if ratios:
                            final_ratio = min(ratios)
                            max_damage = max(damage_estimates)
                            new_SCF["scf_ratio"] = final_ratio
                            new_SCF["expected_damage"] = max_damage
                            new_SCF["property_iter"] = col 
                            if max(damages) > D_max:          
                                optimization_rows.append(new_SCF.iloc[0])

                    if optimization_rows:
                        optimization_rows_df = pd.DataFrame(optimization_rows)
                        print(optimization_rows_df)
                        # Sort by the 'max_SCF' column in descending order
                        # Select the last row (the one with the highest max_SCF)
                        best_option = optimization_rows_df.sort_values(by='expected_damage', ascending=False).tail(1)
                        for col in thk_cols:
                            target_col = col.replace("THK", "SY")
                            SY_original = material(original_data_row[col], original_data_row[target_col])
                            SY_new = yield_strength(best_option[col], SY_original)
                            best_option[target_col] = SY_new

                            
                        print(n_iterations, brc, best_option)
                        if "THK" in best_option["property_iter"].iloc[0]:
                            if "chd" in best_option["property_iter"].iloc[0]:
                                current_OD = best_option["chd_OD1"].iloc[0]
                                original_OD = original_data_row["chd_OD1"].iloc[0]
                                delta_OD = current_OD - original_OD
                                if (delta_OD - i_columns['chd_OD1']) > 0:
                                    i_columns["chd_OD1"] = delta_OD
                            i_columns[best_option["property_iter"].iloc[0]] += 5  
                        else:
                            i_columns[best_option["property_iter"].iloc[0]] += 50 
                    
                    # Check for termination condition
                    if ratios and (max_damage < D_max):
                        print(f"Terminating loop: max_damage ({max_damage}) < D_max ({D_max})")
                        break  # Exit the while loop  

                            
                if optimization_rows:
                    # chd_ΔOD = best_option["chd_OD1"] - original_data_row["chd_OD1"]
                    # chd_ΔTHK = best_option["chd_THK1"] - original_data_row["chd_THK1"]
                    # best_option["chd_OD2"] = original_data_row["chd_OD2"] + chd_ΔOD
                    # best_option["chd_THK2"] = original_data_row["chd_THK2"] + chd_ΔTHK
                    optimization_df = pd.DataFrame(optimization_rows)
                    optimization_df.to_csv(output_path, index=False)
                    # Append the last row of optimization_df to final_optimization_df
                    # best_option.to_csv("test.csv", index=False)
                    final_optimization_df = pd.concat([final_optimization_df, best_option], ignore_index=True)

    print(final_optimization_df)
    return final_optimization_df


def optimize_members_FLS(df, check, filename):
    folder = "OPTIMIZATIONS"
    df_created = False
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, folder), exist_ok=True)
    D_max = 0.90
    braces = [1, 2, 3]  # Brace identifiers
    hs_values = [1, 2, 3, 4, 5, 6, 7, 8]  # HS identifiers
    # Dynamically generate column names for brace side damage
    brace_columns = [f"BRC{brace}_HS{hs}_Damage" for hs in hs_values for brace in braces]
    # Dynamically generate column names for chord side damage
    chord_columns = [f"CHD{brace}_HS{hs}_Damage" for hs in hs_values for brace in braces]
    filename = filename.replace("FLS", "Optimization")
    output_path = os.path.join(folder, filename)
    if not os.path.exists(output_path):
        # Define the keys based on the 'check' type
        if check['type'] == "joint":
            keys = ["BRC", "CHD"]
            target_columns = ["joint", "brace1"]
        elif check['type'] == "inline":
            keys = ["memb1", "memb2"]
            target_columns = ["joint", "memb1"]
        else:
            keys = ["cone", "tub"]
            target_columns = ["joint", "cone"]

        # Define the filename for CSV loading
        CSV_filename = f"{check['type']}_data.csv"
        
        # Filter df based on the max_damage condition
        df_damages = df[(df[f'Max_{keys[0]}_Damage'] > D_max) | (df[f'Max_{keys[1]}_Damage'] > D_max)]
        
        # Load the original CSV
        original_data = pd.read_csv(CSV_filename)
        # Filter-Reduce original data dataframe to rows with damages over D_max
        original_data = original_data[original_data[target_columns].isin(df_damages[target_columns].to_dict(orient='list')).all(axis=1)]

        if check['type'] == "joint" and (check['side'] == "toe"):
            original_SCF = pd.read_csv("joint_SCF.csv")
            original_SCF = original_SCF[original_SCF[target_columns].isin(df_damages[target_columns].to_dict(orient='list')).all(axis=1)]
            optimization_df = process_joint_optimization(df_damages, original_SCF, original_data, braces, target_columns, D_max, folder)   
            df_created = True if (len(optimization_df) > 0) else False
            a = 1
        elif check['type'] == "joint" and (check['side'] == "root"):    
            original_SCF = pd.read_csv("joint_SCF.csv")
            original_SCF = original_SCF[original_SCF[target_columns].isin(df_damages[target_columns].to_dict(orient='list')).all(axis=1)]
            optimization_df = process_joint_optimization(df_damages, original_SCF, original_data, braces, target_columns, D_max, folder)   
            df_created = True if (len(optimization_df) > 0) else False
        elif check['type'] == "inline" and (check['side'] == "toe"):
            optimization_df = process_inline_optimization(df_damages, original_data, target_columns, D_max, folder, False)
            df_created = True if (len(optimization_df) > 0) else False
            a = 1
        elif check['type'] == "cone" and (check['side'] == "toe"):
            optimization_df = process_inline_optimization(df_damages, original_data, target_columns, D_max, folder, True)
            df_created = True if (len(optimization_df) > 0) else False

        if df_created:
            print(optimization_df)
            optimization_df.to_csv(output_path, index=False)

def joint_differences(df):
    # Group by "joint" and "brace1"
    grouped = df.groupby(["joint", "brace1"])
    
    # Dictionary to define columns to compare for each member
    columns_to_compare = {
        "chord1": ["chd_OD1", "chd_THK1"],
        "chord2": ["chd_OD2", "chd_THK2"],
        "brace1": ["brc_OD1", "brc_THK1"],
        "brace2": ["brc_OD2", "brc_THK2"],
        "brace3": ["brc_OD3", "brc_THK3"]
    }
    
    # List to collect the results
    results = []
    
    # Iterate through each group
    for (joint, brace1), group in grouped:   
        # Separate original and updated rows
        original_row = group.iloc[0]
        updated_row = group.iloc[1]
        
        # Check for differences based on the dictionary structure
        for member, columns in columns_to_compare.items():
            # Use notna() to skip NaNs in comparisons
            differences = [
                (original_row[col] != updated_row[col]) and 
                (pd.notna(original_row[col]) or pd.notna(updated_row[col]))
                for col in columns
            ]
            
            if any(differences):  # If there's a difference
                prefix = "chd" if ("chord" in member) else "brc"
                suffix = member[-1]
                # Create a row with the required information
                result = {
                    "member": original_row[member],
                    "group": original_row[f"{member}_group"],
                    "check_type": "joint_FLS",
                    "OD_after": updated_row[f"{prefix}_OD{suffix}"],
                    "OD_before": original_row[f"{prefix}_OD{suffix}"],
                    "THK_after": updated_row[f"{prefix}_THK{suffix}"],
                    "THK_before": original_row[f"{prefix}_THK{suffix}"],
                    "SY_after": updated_row[f"{prefix}_SY{suffix}"], 
                    "SY_before": original_row[f"{prefix}_SY{suffix}"] 
                }
                results.append(result)
    
    # Convert the results into a DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

def inline_differences(df):
    # Dictionary to define columns to compare for each member
    grouped = df.groupby(["joint", "memb1"])
    columns_to_compare = {
        "memb1": ["memb1_OD", "memb1_THK"],
        "memb2": ["memb2_OD", "memb2_THK"],
    }
    
    # List to collect the results
    results = []
    
    # Iterate through each group
    for (joint, brace1), group in grouped:   
        # Separate original and updated rows
        original_row = group.iloc[0]
        updated_row = group.iloc[-1]
        
        # Check for differences based on the dictionary structure
        for member, columns in columns_to_compare.items():
            # Use notna() to skip NaNs in comparisons
            differences = [
                (original_row[col] != updated_row[col]) and 
                (pd.notna(original_row[col]) or pd.notna(updated_row[col]))
                for col in columns
            ]
            
            prefix = member.replace("tubular", "tub") 

            if any(differences):  # If there's a difference
                # Create a row with the required information
                result = {
                    "member": original_row[member],
                    "group": original_row[f"{prefix}_group"],
                    "check_type": "member_FLS",
                    "OD_after": updated_row[f"{prefix}_OD"],
                    "OD_before": original_row[f"{prefix}_OD"],
                    "THK_after": updated_row[f"{prefix}_THK"],
                    "THK_before": original_row[f"{prefix}_THK"],
                    "SY_after": updated_row[f"{prefix}_SY"], 
                    "SY_before": original_row[f"{prefix}_SY"],
                }
                results.append(result)
    
    # Convert the results into a DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df


def cone_differences(df):
    # Dictionary to define columns to compare for each member
    grouped = df.groupby(["joint", "cone"])
    columns_to_compare = {
        "cone": ["cone_L", "cone_S", "cone_THK", "length"],
        "tubular": ["tub_OD", "tub_THK"],
    }
    
    # Lists to collect the results
    results_cone = []
    results_tub = []
    
    # Iterate through each group
    for (joint, memb_type), group in grouped:   
        # Separate original and updated rows
        original_row = group.iloc[0]
        updated_row = group.iloc[-1]
        
        # Check for differences based on the dictionary structure
        for member, columns in columns_to_compare.items():    
            prefix = member.replace("tubular", "tub") 

            # Iterate over each column to check for differences
            differences = [
                (original_row[col] != updated_row[col]) and 
                (pd.notna(original_row[col]) or pd.notna(updated_row[col]))
                for col in columns
            ]

            if "cone" in member and any(differences):  # If there's a difference for cone
                result = {
                    "member": original_row[member],
                    "group": original_row[f"{prefix}_group"],
                    "check_type": "cone_FLS",
                    "OD_L_after": updated_row[f"{prefix}_L"],
                    "OD_L_before": original_row[f"{prefix}_L"],
                    "OD_S_after": updated_row[f"{prefix}_S"],
                    "OD_S_before": original_row[f"{prefix}_S"],
                    "THK_after": updated_row[f"{prefix}_THK"],
                    "THK_before": original_row[f"{prefix}_THK"],
                    "SY_after": updated_row[f"{prefix}_SY"], 
                    "SY_before": original_row[f"{prefix}_SY"],
                    "length_after": updated_row["length"],
                    "length_before": original_row["length"]
                }
                results_cone.append(result)

            elif "tubular" in member and any(differences):  # If there's a difference for tubular
                result = {
                    "member": original_row[member],
                    "group": original_row[f"{prefix}_group"],
                    "check_type": "member_FLS",
                    "OD_after": updated_row[f"{prefix}_OD"],
                    "OD_before": original_row[f"{prefix}_OD"],
                    "THK_after": updated_row[f"{prefix}_THK"],
                    "THK_before": original_row[f"{prefix}_THK"],
                    "SY_after": updated_row[f"{prefix}_SY"], 
                    "SY_before": original_row[f"{prefix}_SY"],
                }
                results_tub.append(result)

    return [pd.DataFrame(results_cone), pd.DataFrame(results_tub)]

def combine_damages(df1, df2):
    # Identify columns containing "Damage" in their name
    damage_columns = [col for col in df1.columns if "Damage" in col]
    # Create a copy of df1 to store combined results
    df_comb = df1.copy()
    # Add damage columns from both dataframes
    for col in damage_columns:
        df_comb[col] = df1[col] + df2[col]
    # Combine the "Environment" column with "-" separator
    if "Environment" in df1.columns:
        df_comb["Environment"] = df1["Environment"] + "-" + df2["Environment"]
    # Return the combined dataframe
    return df_comb

def get_dataframes(subdir_path):
    # Get all .csv files in the subdirectory and its subdirectories
    file_paths = glob.glob(os.path.join(subdir_path, "**/*.csv"), recursive=True)

    # Check the number of files found
    if len(file_paths) == 1:
        # If only one file, read it directly
        return pd.read_csv(file_paths[0])
    elif len(file_paths) > 1:
        # If two or more files, read and concatenate them
        dataframes = [pd.read_csv(file) for file in file_paths]
        return pd.concat(dataframes, ignore_index=True)
    else:
        # No files found
        return pd.DataFrame()
    

def remove_corrosion(df, models):   
    # Create an empty list to store updated groups
    updated_groups = []

    # Group by 'group' and 'analysis'
    groups = df.groupby(['member', 'group', 'check_type'])
    
    # Iterate through each group, modify it, and then append it back
    for (member, group, check_type), group_df in groups:
        analysis = check_type.split("_")[1]
        NC_model = models["NC"]
        model = models.get(analysis, None)
        
        NC_group_obj = NC_model.FindMemberGroup(group)
        group_obj = model.FindMemberGroup(group)


        
        # Check if the segment is a simple tube and calculate OD and THK changes
        if group_obj.Segments[0].IsSimpleTube :
            OD = round(NC_group_obj.Segments[0].Tube.OD * 10, 1)
            THK = round((NC_group_obj.Segments[0].Tube.T * 10), 1)
            
            ΔOD_optimization = round((group_df['OD_after'].iloc[0] - group_df['OD_before'].iloc[0]) / 5) * 5
            ΔTHK_optimization = round((group_df['THK_after'].iloc[0] - group_df['THK_before'].iloc[0]) / 5) * 5

            # Update the group DataFrame directly
            group_df['THK_after'] = min(THK + ΔTHK_optimization,120)
            group_df['THK_before'] = THK
            if group_obj.Id.startswith(("L", "V")):        
                group_df['OD_after'] = OD
                group_df['OD_before'] = OD
            else:
                group_df['OD_after'] = OD + ΔOD_optimization
                group_df['OD_before'] = OD
            
            # Round the OD and THK columns to one decimal place
            group_df[['OD_after', 'OD_before', 'THK_after', 'THK_before']] = group_df[['OD_after', 'OD_before', 'THK_after', 'THK_before']].round(1)
            
        elif group_obj.Segments[0].Section == "CON":
            raise ValueError("UPDATE CODE TO DEAL WITH CONE OPTIMIZATION")
        
        # Reset index of the group to avoid duplicate indices
        updated_groups.append(group_df.reset_index(drop=True))
    
    # Concatenate all updated groups back into a new DataFrame
    updated_df = pd.concat(updated_groups, ignore_index=True)
    
    # Use the 'group' and 'analysis' columns as keys to update the original DataFrame
    df.update(updated_df)

    return df


import os
import pandas as pd

def process_fls_csv_files(start_with):
    """
    Processes all CSV files in the current directory that start with start_with.
    Adds a 'max' column, sorts, saves, and produces a summary ONLY for files
    whose last two digits represent the TOTAL design life (41 years).
    """

    current_directory = os.getcwd()
    summary_rows = []

    for filename in os.listdir(current_directory):
        if filename.startswith(start_with) and filename.endswith(".csv"):
            filepath = os.path.join(current_directory, filename)

            # Extract design life from last two digits before ".csv"
            try:
                life = int(filename[-6:-4])   # e.g. FLS_ABC_41.csv → "41"
            except:
                life = None

            # Read CSV
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            if df.shape[1] < 2:
                print(f"Skipping {filename}: Not enough columns.")
                continue

            # Add max column
            try:
                df["max"] = df.iloc[:, -2:].max(axis=1)
            except Exception as e:
                print(f"Error calculating 'max' for {filename}: {e}")
                continue

            # Sort
            try:
                df = df.sort_values("max", ascending=False).reset_index(drop=True)
            except Exception as e:
                print(f"Error sorting {filename}: {e}")
                continue

            # Save updated file
            try:
                df.to_csv(filepath, index=False)
                print(f"Processed and saved {filename}.")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                continue

            # Only include TOTAL design life (41 years) in summary
            if life == 41:
                try:
                    count_above_1 = (df["max"] > 1).sum()
                    max_damage = df["max"].max()
                    summary_rows.append({
                        "filename": filename,
                        "design_life": life,
                        "count_damage_above_1": count_above_1,
                        "max_damage": max_damage
                    })
                except Exception as e:
                    print(f"Error building summary for {filename}: {e}")

    # Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(current_directory, f"{start_with}_SUMMARY_41YEARS.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")
    else:
        print("No 41-year files found; no summary created.")


# AUX FUNCTIONS

def safe_float(s):
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        # Try to fix malformed scientific notation like "-10.-3" or "-9.3-8"
        match = re.match(r'^(-?\d*\.?\d*)([-+]\d+)$', s)
        if match:
            base, exponent = match.groups()
            if base.endswith('.'):
                base += '0'  # fix case like '-10.' → '-10.0'
            try:
                return float(f"{base}e{exponent}")
            except ValueError:
                pass
        raise ValueError(f"Unrecognized float format: '{s}'")

def wp_distance(brace, chord):
    # 1) find central joint (shared joint ID)
    if brace.joint1_id in (chord.joint1_id, chord.joint2_id):
        central_joint_id = brace.joint1_id
        Pb0 = np.array(brace.coord1)  # brace offset end
    else:
        central_joint_id = brace.joint2_id
        Pb0 = np.array(brace.coord2)

    # 2) central joint coordinate from chord
    if chord.joint1_id == central_joint_id:
        Jc = np.array(chord.coord1)
    else:
        Jc = np.array(chord.coord2)

    # 3) direction vectors (unit, 3D)
    vb = brace.dir_vector(central_joint_id)   # brace axis
    vc = chord.dir_vector(central_joint_id)   # chord axis

    # 4) solve closest points between the two 3D lines
    w0 = Jc - Pb0
    a = np.dot(vb, vb)
    b = np.dot(vb, vc)
    c = np.dot(vc, vc)
    d = np.dot(vb, w0)
    e = np.dot(vc, w0)

    den = a * c - b * b
    t = (d * c - b * e) / den

    # 5) intersection point on brace axis
    P_int = Pb0 + t * vb

    # 6) wp_distance = distance from intersection to central joint
    return np.linalg.norm(P_int - Jc)

def saddle_point(brace, leg_center, R):
    # Convert coordinates (m → mm)
    lc = leg_center * 1000
    B1 = np.array(brace.coord1) * 1000 - lc
    B2 = np.array(brace.coord2) * 1000 - lc
    d  = B2 - B1

    # Convert radius (cm → mm)
    R = R * 10

    x1, y1 = B1[0], B1[1]
    dx, dy = d[0], d[1]

    A = dx*dx + dy*dy
    B = 2*(x1*dx + y1*dy)
    C = x1*x1 + y1*y1 - R*R

    t1, t2 = np.roots([A, B, C])
    t = t1 if 0 <= t1 <= 1 else t2

    S = B1 + t*d

    return S


def brace_saddles(brace, leg_center, R):
    S = saddle_point(brace, leg_center, R)

    # Central angle from axis intersection
    theta_c = np.arctan2(S[1], S[0])

    # OD in cm → mm
    Db = brace.OD * 10
    R_mm = R * 10

    # Correct angular half-width using arcsin
    ratio = (Db / 2.0) / R_mm
    ratio = max(-1.0, min(1.0, ratio))  # clamp
    dtheta = np.arcsin(ratio)

    theta_left  = theta_c - dtheta
    theta_right = theta_c + dtheta


    # DEBUG PRINTS
    print("\n--- BRACE SADDLES DEBUG ---")
    print(f"Brace {brace.Id}")
    print(f"  Saddle axis point (mm): {S}")
    print(f"  theta_c: {theta_c:.6f} rad")
    print(f"  Db (mm): {Db}")
    print(f"  R (mm): {R_mm}")
    print(f"  ratio Db/(2R): {ratio:.6f}")
    print(f"  dtheta (arcsin): {dtheta:.6f} rad")
    print(f"  theta_left: {theta_left:.6f} rad")
    print(f"  theta_right: {theta_right:.6f} rad")
    print("---------------------------\n")

    return theta_left, theta_right, theta_c


# def arc_gap(theta1, theta2, R):
#     """
#     Directed arc gap:
#     - Positive => real gap
#     - Negative => real overlap
#     - Wrap-around corrected by adding 2π
#     """
#     dtheta = theta2 - theta1
#     print(f"theta1={theta1:.6f}, theta2={theta2:.6f}, dtheta_raw={dtheta:.6f}")

#     gap_mm = R * 10 * dtheta
#     print(f"gap_mm={gap_mm:.3f}")
#     return gap_mm


def assign_unique_groups(model, z_max):
    """
    Update the model so that each member below z_max has a unique group_id.
    Uses model.make_unique_group_id() to generate valid 3-char IDs.
    Returns the updated model.
    """

    used_ids = set()   # track allocated group IDs

    for member_id, member in model.members.items():

        # --- Z filter: skip members above z_max ---
        z1 = member.coord1[2]
        z2 = member.coord2[2]
        if max(z1, z2) > z_max:
            continue

        base_gid = member.group_id

        # First time this group_id appears → keep it
        if base_gid not in used_ids:
            used_ids.add(base_gid)
            continue

        # Group already used → generate a unique 3-char ID
        new_gid = model.make_unique_group_id(
            base_id=base_gid,
            extra_used=used_ids,
            first_letter="Z"
        )

        # Clone the group with the new ID
        model.groups[base_gid].clone_group(
            model_out=model,
            new_id=new_gid,
            add_to_model=True
        )

        # Assign the new group_id to the member
        member.group_id = new_gid

        # Track the new ID
        used_ids.add(new_gid)

    return model


def count_members(model):
    n = len(model.members)
    print(f"Total members in model: {n}")
    return

def load_dataframe_UCs_csv(folder):
    """
    Load all CSV strength-check UC dataframes from the given folder.
    Returns a dict: {filename: dataframe}
    """

    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)

    dfs = {}

    if not files:
        print(f"[WARN] No CSV files found in: {folder}")
        return dfs

    for file in files:
        fname = os.path.basename(file)

        try:
            # Try comma first, fallback to any whitespace
            try:
                df = pd.read_csv(file)
            except:
                df = pd.read_csv(file, sep=r"\s+")

            dfs[fname] = df
            print(f"[LOAD] {fname}: {df.shape[0]} rows, {df.shape[1]} columns")

        except Exception as e:
            print(f"[ERROR] Could not load {fname}: {e}")

    print(f"\n[SUMMARY] Loaded {len(dfs)} CSV UC dataframes.")
    return dfs

def collect_strength_UCs(Strength_Results, model, z_max = 21.5, source = "strength"):
    """
    Collect maximum UC per member from:
      - cone files
      - member files
      - joint files (chord/brace UC pairs)
    Returns DataFrame:
        member_id, group_id, UC_max, UC_source, load_case
    """

    records = []

    for fname, df in Strength_Results.items():

        filename_lower = fname.lower()

        # ---------------------------------------------------------
        # CASE 1 — JOINT FILES
        # ---------------------------------------------------------
        if "joint" in filename_lower:
            print(f"[PROCESS JOINT] {fname}: {df.shape[0]} rows")

            chord_cols = [c for c in df.columns if c.lower().startswith("chord")]
            brace_cols = [c for c in df.columns if c.lower().startswith("brace")]
            uc_cols    = [c for c in df.columns if c.lower().startswith("uc_final")]

            if not chord_cols or not brace_cols or not uc_cols:
                print(f"[WARN] Missing chord/brace/UC columns in {fname}")
                continue

            for _, row in df.iterrows():

                load_case = row["loads"] if "loads" in df.columns else None
                uc_source = f"joint {source}"

                for chd in [1, 2]:
                    chord_col = f"chord{chd}"
                    if chord_col not in df.columns:
                        continue

                    chord_mem = row[chord_col]
                    if pd.isna(chord_mem) or chord_mem not in model.members:
                        continue

                    chord = model.members[chord_mem]
                    if min(chord.coord1[2], chord.coord2[2]) > z_max:
                        continue

                    for brc in [1, 2, 3]:
                        brace_col = f"brace{brc}"
                        if brace_col not in df.columns:
                            continue

                        brace_mem = row[brace_col]
                        if pd.isna(brace_mem) or brace_mem not in model.members:
                            continue
                        brace = model.members[brace_mem]
                        if min(brace.coord1[2], brace.coord2[2]) > z_max:
                            continue

                        uc_col = f"UC_final_chd{chd}_brc{brc}"
                        if uc_col not in df.columns:
                            continue

                        uc = row[uc_col]
                        if pd.isna(uc):
                            continue

                        gid_chord = model.members[chord_mem].group_id
                        gid_brace = model.members[brace_mem].group_id

                        records.append((chord_mem, gid_chord, uc, uc_source, load_case))
                        records.append((brace_mem, gid_brace, uc, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # CASE 2 — CONE / MEMBER FILES
        # ---------------------------------------------------------
        if "cone" not in filename_lower and "member" not in filename_lower:
            print(f"[SKIP] Not cone/member/joint: {fname}")
            continue

        if "member" not in df.columns or "UC_max" not in df.columns:
            print(f"[WARN] Missing required columns in {fname}")
            continue

        print(f"[PROCESS MEMBER] {fname}: {df.shape[0]} rows")

        for _, row in df.iterrows():
            mem = row["member"]
            uc  = row["UC_max"]

            if pd.isna(mem) or pd.isna(uc):
                continue

            if mem not in model.members:
                continue
            member = model.members[mem]
            if min(member.coord1[2], member.coord2[2]) > z_max:
                continue

            gid = model.members[mem].group_id
            load_case = row["loads"] if "loads" in df.columns else None

            if "cone" in filename_lower:
                uc_source = f"cone {source}"
            else:
                uc_source = f"member {source}"

            records.append((mem, gid, uc, uc_source, load_case))

    # ---------------------------------------------------------
    # Build final DataFrame
    # ---------------------------------------------------------
    df_all = pd.DataFrame(
        records,
        columns=["member_id", "group_id", "UC_max", "UC_source", "load_case"]
    )

    df_max = (
        df_all.sort_values("UC_max", ascending=False)
              .groupby(["member_id", "group_id"], as_index=False)
              .first()
    )

    print(f"\n[SUMMARY] Collected UCs for {df_max.shape[0]} members.")
    return df_max

def collect_strength_UCs_II(Strength_Results, model, z_max = 21.5, source = "strength"):
    """
    Collect maximum UC per member from:
      - cone files
      - member files
      - joint files (chord/brace UC pairs)
    Returns DataFrame:
        member_id, group_id, UC_max, UC_source, load_case
    """

    records = []

    for fname, df in Strength_Results.items():

        filename_lower = fname.lower()

        # ---------------------------------------------------------
        # CASE 1 — JOINT FILES
        # ---------------------------------------------------------
        if "joint" in filename_lower:
            print(f"[PROCESS JOINT] {fname}: {df.shape[0]} rows")

            chord_cols = [c for c in df.columns if c.lower().startswith("chord")]
            brace_cols = [c for c in df.columns if c.lower().startswith("brace")]
            uc_cols    = [c for c in df.columns if c.lower().startswith("uc_final")]

            if not chord_cols or not brace_cols or not uc_cols:
                print(f"[WARN] Missing chord/brace/UC columns in {fname}")
                continue

            for _, row in df.iterrows():

                load_case = row["loads"] if "loads" in df.columns else None
                uc_source = f"joint {source}"

                for chd in [1, 2]:
                    chord_col = f"chord{chd}"
                    if chord_col not in df.columns:
                        continue

                    chord_mem = row[chord_col]
                    if pd.isna(chord_mem) or chord_mem not in model.members:
                        continue

                    chord = model.members[chord_mem]
                    if min(chord.coord1[2], chord.coord2[2]) > z_max:
                        continue

                    for brc in [1, 2, 3]:
                        brace_col = f"brace{brc}"
                        if brace_col not in df.columns:
                            continue

                        brace_mem = row[brace_col]
                        if pd.isna(brace_mem) or brace_mem not in model.members:
                            continue
                        brace = model.members[brace_mem]
                        if min(brace.coord1[2], brace.coord2[2]) > z_max:
                            continue

                        uc_col = f"UC_final_chd{chd}_brc{brc}"
                        if uc_col not in df.columns:
                            continue

                        uc = row[uc_col]
                        if pd.isna(uc):
                            continue

                        gid_chord = model.members[chord_mem].group_id
                        gid_brace = model.members[brace_mem].group_id

                        records.append((chord_mem, gid_chord, uc, uc_source, load_case))
                        records.append((brace_mem, gid_brace, uc, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # CASE 2 — CONE / MEMBER FILES
        # ---------------------------------------------------------
        if "cone" not in filename_lower and "member" not in filename_lower:
            print(f"[SKIP] Not cone/member/joint: {fname}")
            continue

        if "member" not in df.columns or "UC_max" not in df.columns:
            print(f"[WARN] Missing required columns in {fname}")
            continue

        print(f"[PROCESS MEMBER] {fname}: {df.shape[0]} rows")

        for _, row in df.iterrows():
            mem = row["member"]
            uc  = row["UC_max"]

            if pd.isna(mem) or pd.isna(uc):
                continue

            if mem not in model.members:
                continue
            member = model.members[mem]
            if min(member.coord1[2], member.coord2[2]) > z_max:
                continue

            gid = model.members[mem].group_id
            load_case = row["loads"] if "loads" in df.columns else None

            if "cone" in filename_lower:
                uc_source = f"cone {source}"
            else:
                uc_source = f"member {source}"

            records.append((mem, gid, uc, uc_source, load_case))

    # ---------------------------------------------------------
    # Build final DataFrame
    # ---------------------------------------------------------
    df_all = pd.DataFrame(
        records,
        columns=["member_id", "group_id", "UC_max", "UC_source", "load_case"]
    )

    df_max = (
        df_all.sort_values("UC_max", ascending=False)
              .groupby(["member_id", "group_id"], as_index=False)
              .first()
    )

    print(f"\n[SUMMARY] Collected UCs for {df_max.shape[0]} members.")
    return df_max

def collect_fatigue_UCs(Fatigue_Results, model, joint_data):
    """
    Collect fatigue UCs from:
      - joint fatigue files
      - inline fatigue files
      - cone fatigue files

    Returns DataFrame:
        member_id, group_id, UC_max, UC_source, load_case
    """

    records = []

    for fname, df in Fatigue_Results.items():

        filename_lower = fname.lower()
        load_case = fname   # <── fatigue has no load cases, store filename

        # ---------------------------------------------------------
        # CASE 1 — JOINT FATIGUE FILES
        # ---------------------------------------------------------
        if "joint" in filename_lower:
            print(f"[PROCESS FATIGUE JOINT] {fname}: {df.shape[0]} rows")

            if not {"Member 1", "Member 2", "DAMAGE"}.issubset(df.columns):
                print(f"[WARN] Missing joint fatigue columns in {fname}")
                continue

            uc_source = "fatigue joint"

            for _, row in df.iterrows():
                uc = row["DAMAGE"]
                if pd.isna(uc):
                    continue

                joint = row["Joint"]
                type = row["Type"]

                # Chord 1
                if type == "CHD":
                    chd1 = row["Member 1"]
                    brc = row["Member 2"]

                else:
                    chd1 = row["Member 2"]
                    brc = row["Member 1"]

                mask = (joint_data["joint"] == joint) & ( (joint_data["chord1"] == chd1) | (joint_data["chord2"] == chd1))
                row = joint_data.loc[mask].iloc[0]  # assumes exactly one match
                chd2 = row["chord2"] if row["chord1"] == chd1 else row["chord1"]

                if not pd.isna(chd1) and chd1 in model.members:
                    gid = model.members[chd1].group_id
                    records.append((chd1, gid, uc, uc_source, load_case))

                if not pd.isna(chd2) and chd2 in model.members:
                    gid = model.members[chd2].group_id
                    records.append((chd2, gid, uc, uc_source, load_case))

                # Brace 2
                if not pd.isna(m2) and brc in model.members:
                    gid = model.members[brc].group_id
                    records.append((brc, gid, uc, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # CASE 2 — INLINE FATIGUE FILES
        # ---------------------------------------------------------
        if "inline" in filename_lower:
            print(f"[PROCESS FATIGUE INLINE] {fname}: {df.shape[0]} rows")

            required = {"memb1", "memb2", "Max_memb1_Damage", "Max_memb2_Damage"}
            if not required.issubset(df.columns):
                print(f"[WARN] Missing inline fatigue columns in {fname}")
                continue

            uc_source = "fatigue inline"

            for _, row in df.iterrows():

                # memb1
                m1 = row["memb1"]
                uc1 = row["Max_memb1_Damage"]
                if not pd.isna(m1) and not pd.isna(uc1) and m1 in model.members:
                    gid = model.members[m1].group_id
                    records.append((m1, gid, uc1, uc_source, load_case))

                # memb2
                m2 = row["memb2"]
                uc2 = row["Max_memb2_Damage"]
                if not pd.isna(m2) and not pd.isna(uc2) and m2 in model.members:
                    gid = model.members[m2].group_id
                    records.append((m2, gid, uc2, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # CASE 3 — CONE FATIGUE FILES
        # ---------------------------------------------------------
        if "cone" in filename_lower:
            print(f"[PROCESS FATIGUE CONE] {fname}: {df.shape[0]} rows")

            required = {"cone", "tubular", "Max_cone_Damage", "Max_tub_Damage"}
            if not required.issubset(df.columns):
                print(f"[WARN] Missing cone fatigue columns in {fname}")
                continue

            uc_source = "fatigue cone"

            for _, row in df.iterrows():

                # cone
                cone_mem = row["cone"]
                uc_cone = row["Max_cone_Damage"]
                if not pd.isna(cone_mem) and not pd.isna(uc_cone) and cone_mem in model.members:
                    gid = model.members[cone_mem].group_id
                    records.append((cone_mem, gid, uc_cone, uc_source, load_case))

                # tubular
                tub_mem = row["tubular"]
                uc_tub = row["Max_tub_Damage"]
                if not pd.isna(tub_mem) and not pd.isna(uc_tub) and tub_mem in model.members:
                    gid = model.members[tub_mem].group_id
                    records.append((tub_mem, gid, uc_tub, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # SKIP OTHER FILES
        # ---------------------------------------------------------
        print(f"[SKIP] Not joint/inline/cone fatigue: {fname}")

    # ---------------------------------------------------------
    # Build final DataFrame
    # ---------------------------------------------------------
    df_all = pd.DataFrame(
        records,
        columns=["member_id", "group_id", "UC_max", "UC_source", "load_case"]
    )

    df_max = (
        df_all.sort_values("UC_max", ascending=False)
              .groupby(["member_id", "group_id"], as_index=False)
              .first()
    )

    print(f"\n[SUMMARY] Collected fatigue UCs for {df_max.shape[0]} members.")
    return df_max

def collect_fatigue_UCs_II(Fatigue_Results, model, joint_data):
    """
    Collect fatigue UCs from:
      - joint fatigue files
      - inline fatigue files
      - cone fatigue files

    Returns DataFrame:
        member_id, group_id, UC_max, UC_source, load_case
    """

    records = []

    for fname, df in Fatigue_Results.items():

        filename_lower = fname.lower()
        load_case = fname   # <── fatigue has no load cases, store filename

        # ---------------------------------------------------------
        # CASE 1 — JOINT FATIGUE FILES
        # ---------------------------------------------------------
        if "joint" in filename_lower:
            print(f"[PROCESS FATIGUE JOINT] {fname}: {df.shape[0]} rows")

            if not {"Member 1", "Member 2", "DAMAGE"}.issubset(df.columns):
                print(f"[WARN] Missing joint fatigue columns in {fname}")
                continue

            uc_source = "fatigue joint"

            for _, row in df.iterrows():
                uc = row["DAMAGE"]
                if pd.isna(uc):
                    continue

                joint = row["Joint"]
                type = row["Type"]

                # Chord 1
                if type == "CHD":
                    chd1 = row["Member 1"]
                    brc = row["Member 2"]

                else:
                    chd1 = row["Member 2"]
                    brc = row["Member 1"]

                mask = (joint_data["joint"] == joint) & ( (joint_data["chord1"] == chd1) | (joint_data["chord2"] == chd1))
                row = joint_data.loc[mask].iloc[0]  # assumes exactly one match
                chd2 = row["chord2"] if row["chord1"] == chd1 else row["chord1"]

                if not pd.isna(chd1) and chd1 in model.members:
                    gid = model.members[chd1].group_id
                    records.append((chd1, gid, uc, uc_source, load_case))

                if not pd.isna(chd2) and chd2 in model.members:
                    gid = model.members[chd2].group_id
                    records.append((chd2, gid, uc, uc_source, load_case))

                # Brace 2
                if not pd.isna(m2) and brc in model.members:
                    gid = model.members[brc].group_id
                    records.append((brc, gid, uc, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # CASE 2 — INLINE FATIGUE FILES
        # ---------------------------------------------------------
        if "inline" in filename_lower or "cone" in filename_lower:
            print(f"[PROCESS FATIGUE INLINE] {fname}: {df.shape[0]} rows")

            required = {"Member 1", "Member 1", "DAMAGEs"}
            if not required.issubset(df.columns):
                print(f"[WARN] Missing inline fatigue columns in {fname}")
                continue

            uc_source = "fatigue inline"

            for _, row in df.iterrows():

                # memb1
                m1 = row["Member 1"]
                uc1 = row["DAMAGE"]
                if not pd.isna(m1) and not pd.isna(uc1) and m1 in model.members:
                    gid = model.members[m1].group_id
                    records.append((m1, gid, uc1, uc_source, load_case))

                # memb2
                m2 = row["Member 2"]
                uc2 = row["DAMAGE"]
                if not pd.isna(m2) and not pd.isna(uc2) and m2 in model.members:
                    gid = model.members[m2].group_id
                    records.append((m2, gid, uc2, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # CASE 3 — CONE FATIGUE FILES
        # ---------------------------------------------------------
        if "cone" in filename_lower:
            print(f"[PROCESS FATIGUE CONE] {fname}: {df.shape[0]} rows")

            required = {"cone", "tubular", "Max_cone_Damage", "Max_tub_Damage"}
            if not required.issubset(df.columns):
                print(f"[WARN] Missing cone fatigue columns in {fname}")
                continue

            uc_source = "fatigue cone"

            for _, row in df.iterrows():

                # cone
                cone_mem = row["Member 1"]
                uc_cone = row["DAMAGE"]
                if not pd.isna(cone_mem) and not pd.isna(uc_cone) and cone_mem in model.members:
                    gid = model.members[cone_mem].group_id
                    records.append((cone_mem, gid, uc_cone, uc_source, load_case))

                # tubular
                tub_mem = row["Member 2"]
                uc_tub = row["DAMAGE"]
                if not pd.isna(tub_mem) and not pd.isna(uc_tub) and tub_mem in model.members:
                    gid = model.members[tub_mem].group_id
                    records.append((tub_mem, gid, uc_tub, uc_source, load_case))

            continue

        # ---------------------------------------------------------
        # SKIP OTHER FILES
        # ---------------------------------------------------------
        print(f"[SKIP] Not joint/inline/cone fatigue: {fname}")

    # ---------------------------------------------------------
    # Build final DataFrame
    # ---------------------------------------------------------
    df_all = pd.DataFrame(
        records,
        columns=["member_id", "group_id", "UC_max", "UC_source", "load_case"]
    )

    df_max = (
        df_all.sort_values("UC_max", ascending=False)
              .groupby(["member_id", "group_id"], as_index=False)
              .first()
    )

    print(f"\n[SUMMARY] Collected fatigue UCs for {df_max.shape[0]} members.")
    return df_max


UC_COLOR_SCALE = {
    0.0: "173 216 230",   # light blue
    0.1: "135 206 250",   # sky blue
    0.2: "0   191 255",   # deep sky blue
    0.3: "0   128 255",   # blue
    0.4: "0   255 255",   # cyan
    0.5: "0   255 128",   # green-cyan
    0.6: "0   255 0",     # green
    0.7: "255 255 0",     # yellow
    0.8: "255 100 0",     # orange
    0.9: "255 80 0",     # deep orange
    1.0: "255 40  0",      # orange-red
}

UC_OVER_1_COLOR = "255 0   0"
  # red



def uc_to_color(uc):
    if uc > 1.0:
        return UC_OVER_1_COLOR

    bucket = round(float(uc) * 10) / 10.0
    bucket = max(0.0, min(bucket, 1.0))

    return UC_COLOR_SCALE[bucket]

def insert_group_colors(file_path, member_UCs_df):

    with open(file_path, "r") as f:
        lines = f.readlines()

    insert_index = len(lines) - 1

    for _, row in member_UCs_df.iterrows():
        gid = row["group_id"]
        color = row["color"]
        lines.insert(insert_index, f" **GCOL** {gid}M{color}\n")
        insert_index += 1

    with open(file_path, "w") as f:
        f.writelines(lines)

    print(f"[GCOL] Inserted {member_UCs_df.shape[0]} UC color lines.")



def compute_possible_reduction(OD, THK, strength_allowance, fatigue_allowance, target, m=3):
    """
    Estimate the max reduction (in OD or THK, per `target`) that keeps the
    member within its strength and fatigue allowances, holding the other
    dimension fixed.

    UC_strength ~ 1/A        -> A_new = A_old * (1 - strength_allowance)
    UC_fatigue  ~ 1/A^m      -> A_new = A_old * (1 - fatigue_allowance) ** (1/m)

    Governing case = whichever gives the larger A_new (smaller reduction),
    since exceeding it would violate the other criterion.

    A minimum thickness of 1.5 cm is enforced, and OD is floored at
    THK * 18 to preserve the OD/THK >= 18 ratio.

    OD reduction is rounded down to the nearest 1 cm (10 mm).
    THK reduction is rounded down to the nearest 0.5 cm (5 mm).

    target: "OD" or "THK" - which dimension's reduction to compute and return.
    """
    MIN_THK = 1.5
    A_old = math.pi * THK * (OD - THK)  # exact annulus area
    strength_factor = (1-strength_allowance) ** (1/4)
    fatigue_factor = (1-fatigue_allowance) ** (1/10)
    A_new_strength = A_old *  strength_factor
    A_new_fatigue = A_old * fatigue_factor 

    A_new = min(max(A_new_strength, A_new_fatigue), A_old)

    if target == "OD":
        # A = pi * THK * (OD - THK)  =>  OD_new = A_new / (pi * THK) + THK
        OD_new = A_new / (math.pi * THK) + THK

        min_OD = THK * 18.0  # preserves OD/THK >= 18 at this THK
        OD_new = max(OD_new, min_OD)

        raw_reduction = max(0.0, OD - OD_new)
        rounded_reduction = math.floor(raw_reduction / 1.0) * 1.0
        return rounded_reduction

    elif target == "THK":
        # A = pi * (OD*THK - THK^2)  =>  pi*THK_new^2 - pi*OD*THK_new + A_new = 0
        a, b, c = math.pi, -math.pi * OD, A_new
        disc = b**2 - 4 * a * c
        if disc < 0:
            THK_new = THK  # no real solution -> no reduction
        else:
            THK_new = (-b - math.sqrt(disc)) / (2 * a)  # smaller root = physical one

        THK_new = max(THK_new, MIN_THK)

        raw_reduction = max(0.0, THK - THK_new)
        rounded_reduction = math.floor(raw_reduction / 0.5) * 0.5
        return rounded_reduction

    else:
        raise ValueError(f"target must be 'OD' or 'THK', got {target!r}")
    
def get_final_colinear_sections_Old(colinear_members_data, brace_member_data, chord_member_data,
                                is_leg,
                                sl_threshold=0.5,
                                fls_threshold=0.3):
    """Returns a dictionary of only the colinear members that received section updates 
    with their new_OD and new_THK, maintaining internal diameter and minimum thickness.
    
    - OD & THK are in cm. 
    - OD reduction rounded to multiples of 10mm (1 cm).
    - THK reduction rounded to multiples of 5mm (0.5 cm).
    - Checks are only run for members where od_red > 0 or thk_red > 0.
    - If verification fails, OD reduction is eliminated (set to 0.0).
    - Returns only members with updated dimensions.
    - is_leg: if True, the chain is outer-flushed (OD stays constant across the
      chain; only THK is reduced). If False, the chain is inner-flushed as before
      (OD can reduce, with THK reduction applied on top).
    """
    end_ids = set(chord_member_data.keys())
    MIN_THK = 1.5

    def ok(members):
        return all((0.0 if pd.isna(uc) else uc) < sl_threshold and
                   (0.0 if pd.isna(f) else f) < fls_threshold
                   for uc, f in members)

    # --- 1. Chain & End Chords (Global OD Reduction Evaluation) ---
    # Legs are outer-flushed - OD never reduces, so skip this evaluation entirely.
    chain_skip_od = any(d["skip_od"] for d in colinear_members_data.values())
    if is_leg or chain_skip_od:
        can_reduce_OD = 0.0
    else:
        chain = [(d["UC_max"], d["UC_FLS_max"]) for d in colinear_members_data.values()]
        end_chords = [(c["chord_UCmax"], c["chord_FLS_UCmax"])
                      for j in end_ids for c in chord_member_data.get(j, [])]

        all_od_group = chain + end_chords

        if ok(chain) and ok(end_chords):
            max_uc = max((0.0 if pd.isna(uc) else uc) for uc, _ in all_od_group)
            max_fls = max((0.0 if pd.isna(f) else f) for _, f in all_od_group)

            strength_allowance = 1.0 - max_uc
            fatigue_allowance = 1.0 - max_fls

            member_od_reductions = [
                compute_possible_reduction(d["OD"], d["THK"], strength_allowance, fatigue_allowance, target="OD")
                for d in colinear_members_data.values()
            ]
            can_reduce_OD = min(member_od_reductions) if member_od_reductions else 0.0
        else:
            can_reduce_OD = 0.0

    # --- 2. Individual Members (Local Thickness Reduction Evaluation) ---
    can_reduce_THK = {}
    for member_id, d in colinear_members_data.items():
        # Thickness is blocked only by the member's own THK flag.
        if d["skip_thk"]:
            can_reduce_THK[member_id] = 0.0
            continue    
        joints = {member_id[:4], member_id[-4:]}
        members = [(d["UC_max"], d["UC_FLS_max"])]
        members += [(b["brace_UCmax"], b["brace_FLS_UCmax"])
                    for j in joints for b in brace_member_data.get(j, [])]

        if ok(members):
            m_max_uc = max((0.0 if pd.isna(uc) else uc) for uc, _ in members)
            m_max_fls = max((0.0 if pd.isna(f) else f) for _, f in members)

            m_strength_allowance = 1.0 - m_max_uc
            m_fatigue_allowance = 1.0 - m_max_fls

            can_reduce_THK[member_id] = compute_possible_reduction(
                d["OD"], d["THK"], m_strength_allowance, m_fatigue_allowance, target="THK"
            )
        else:
            can_reduce_THK[member_id] = 0.0

    # --- 3. Final Verification Step (Runs only when reductions are present) ---
    def compute_verification(od_red, thk_red_dict):
        passed = True
        details = {}

        for member_id, d in colinear_members_data.items():
            thk_red = thk_red_dict.get(member_id, 0.0)

            if od_red == 0 and thk_red == 0:
                details[member_id] = {
                    "new_OD": d["OD"],
                    "new_THK": d["THK"]
                }
                continue

            if is_leg:
                # Outer-flushed: OD stays constant, only THK reduces.
                new_od = d["OD"]
                new_thk = max(d["THK"] - thk_red, MIN_THK)
            else:
                # Inner-flushed: OD can reduce; ID is decremented by od_red,
                # THK reduction applied on top.
                suggested_id = d["OD"] - 2 * d["THK"] - od_red
                new_thk = max(d["THK"] - thk_red, MIN_THK)
                new_od = suggested_id + 2 * new_thk

            # Check 1: OD / THK >= 18
            ratio_valid = ((new_od / new_thk) >= 18.0)

            member_joints = {member_id[:4], member_id[-4:]}

            # Check 2: at end joints, colinear OD must not exceed chord OD
            chord_valid = True
            for j in member_joints & end_ids:
                for c in chord_member_data.get(j, []):
                    if 'OD' in c and new_od > c['OD']:
                        chord_valid = False
                        break
                if not chord_valid:
                    break

            # Check 3: at middle joints, colinear OD must be >= attached brace OD
            brace_valid = True
            for j in member_joints - end_ids:
                for b in brace_member_data.get(j, []):
                    if 'OD' in b and new_od < b['OD']:
                        brace_valid = False
                        break
                if not brace_valid:
                    break

            is_valid = ratio_valid and chord_valid and brace_valid

            details[member_id] = {
                "new_OD": new_od,
                "new_THK": new_thk
            }
            if not is_valid:
                passed = False

        return passed, details

    has_reductions = (can_reduce_OD > 0) or any(t > 0 for t in can_reduce_THK.values())

    if has_reductions:
        verification_passed, final_chain = compute_verification(can_reduce_OD, can_reduce_THK)
        if not verification_passed:
            can_reduce_OD = 0.0
            _, final_chain = compute_verification(can_reduce_OD, can_reduce_THK)
    else:
        final_chain = {
            member_id: {"new_OD": d["OD"], "new_THK": d["THK"]}
            for member_id, d in colinear_members_data.items()
        }

    updated_chain = {
        m_id: dims for m_id, dims in final_chain.items()
        if dims["new_OD"] != colinear_members_data[m_id]["OD"] or dims["new_THK"] != colinear_members_data[m_id]["THK"]
    }

    return updated_chain

def get_final_colinear_sections(colinear_members_data, brace_member_data, chord_member_data,
                                is_leg,
                                sl_threshold=0.5,
                                fls_threshold=0.3):
    """Returns a dictionary of only the colinear members that received section updates.

    - OD & THK are in cm.
    - Tubular members are updated first:
        * chain OD reduction (global), decided from tubular UCs and end chords only
        * own THK reduction, from own UCs plus braces at their joints
        * legs are outer-flushed (OD constant), other chains inner-flushed (ID shifts by od_red)
        * OD / THK >= 18 checked on tubular members only
    - Cones are adjusted afterwards, from the final tubular sections:
        * large end (joint1) inner-flushed with its neighbour: OD_L = neighbour ID + 2 * cone THK
        * small end (joint2) outer-flushed with its neighbour: OD_S = neighbour OD
        * THK1 / THK2 = new THK of the tubular neighbour at joint1 / joint2
        * cone THK reduced from its own UCs (plus braces at its joints) only when below
          thresholds; otherwise left unchanged. Cones are never increased.
        * if an end has no tubular neighbour, it keeps its own ID (large end) / OD (small end)
          and its THK1 / THK2 is returned as None (leave unchanged)
    - If a cone fails its checks, its own THK reduction is dropped first; if it still fails,
      the chain OD reduction is eliminated (set to 0.0) and everything is re-checked.
      If it still fails, no member of the chain is updated.
    - Returns only members whose section changed:
      tubulars -> new_OD, new_THK
      cones    -> new_OD_L, new_OD_S, new_THK, new_THK1, new_THK2
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

    def neighbour_id(member_id, j):
        # Tubular chain member sharing joint j with member_id (None if there isn't one)
        return next((n for n in tubes if n != member_id and j in (n[:4], n[-4:])), None)

    # --- 1. Chain & End Chords (Global OD Reduction Evaluation, tubular members only) ---
    chain_skip_od = any(d["skip_od"] for d in colinear_members_data.values())
    if is_leg or chain_skip_od or not tubes:
        can_reduce_OD = 0.0
    else:
        chain = [(d["UC_max"], d["UC_FLS_max"]) for d in tubes.values()]
        end_chords = [(c["chord_UCmax"], c["chord_FLS_UCmax"])
                      for j in end_ids for c in chord_member_data.get(j, [])]

        if ok(chain) and ok(end_chords):
            group = chain + end_chords
            strength_allowance = 1.0 - max(val(uc) for uc, _ in group)
            fatigue_allowance = 1.0 - max(val(f) for _, f in group)
            can_reduce_OD = min(
                compute_possible_reduction(d["OD"], d["THK"], strength_allowance,
                                           fatigue_allowance, target="OD")
                for d in tubes.values()
            )
        else:
            can_reduce_OD = 0.0

    # --- 2. Individual Members (Local Thickness Reduction Evaluation, cones included) ---
    can_reduce_THK = {}
    for member_id, d in colinear_members_data.items():
        if d["skip_thk"]:
            can_reduce_THK[member_id] = 0.0
            continue
        joints = {member_id[:4], member_id[-4:]}
        members = [(d["UC_max"], d["UC_FLS_max"])]
        members += [(b["brace_UCmax"], b["brace_FLS_UCmax"])
                    for j in joints for b in brace_member_data.get(j, [])]

        if ok(members):
            m_strength_allowance = 1.0 - max(val(uc) for uc, _ in members)
            m_fatigue_allowance = 1.0 - max(val(f) for _, f in members)
            ods = [d["OD_L"], d["OD_S"]] if d["is_cone"] else [d["OD"]]
            can_reduce_THK[member_id] = min(
                compute_possible_reduction(od, d["THK"], m_strength_allowance,
                                           m_fatigue_allowance, target="THK")
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
                "new_THK": new_thk, "new_THK1": None, "new_THK2": None}

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
                "new_THK": new_thk, "new_THK1": thk1, "new_THK2": thk2}

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
        j1, j2 = member_id[:4], member_id[-4:]
        if colinear_members_data[member_id]["is_cone"]:
            od_at = {j1: sec["new_OD_L"], j2: sec["new_OD_S"]}
        else:
            od_at = {j1: sec["new_OD"], j2: sec["new_OD"]}
            # Check 1: OD / THK >= 18 (tubular members only)
            if sec["new_OD"] / sec["new_THK"] < 18.0:
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

    def compute_verification(od_red):
        details = {}

        # 3a. Tubular members first
        for member_id, d in tubes.items():
            thk_red = can_reduce_THK[member_id]
            sec = new_section(d, od_red, thk_red)
            if (od_red or thk_red) and not is_valid(member_id, sec):
                return False, {}
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
                return False, {}
            details[member_id] = sec

        return True, details

    passed, details = compute_verification(can_reduce_OD)
    if not passed and can_reduce_OD > 0:
        passed, details = compute_verification(0.0)
    if not passed:
        return {}

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
                new_id = generate_unique_group(existing_ids, old_group)
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
                new_group_id = generate_unique_group(existing_group_ids, old_group)
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

def apply_updated_sections(model_out, updated_members, member_zones, tol=0.01):
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
        if m.is_cone:
            return ("CON", float(m.OD_L), float(m.OD_S), float(m.THK), *cone_end_thks(m))
        return ("TUB", float(m.OD), float(m.THK))

    def new_size_of(m, new):
        if m.is_cone:
            cur_thk1, cur_thk2 = cone_end_thks(m)
            thk1 = cur_thk1 if new["new_THK1"] is None else float(new["new_THK1"])
            thk2 = cur_thk2 if new["new_THK2"] is None else float(new["new_THK2"])
            return ("CON", float(new["new_OD_L"]), float(new["new_OD_S"]),
                    float(new["new_THK"]), thk1, thk2)
        return ("TUB", float(new["new_OD"]), float(new["new_THK"]))

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

        # a matching size already exists in this zone: reuse its group
        group_id = find_group(zone, size)
        if group_id is not None:
            m.group_id = group_id
            changes[member_id] = (old_group, group_id, False)
            continue

        # otherwise clone the member's group and give the clone the new size
        new_group_id = generate_unique_group(existing_group_ids, old_group)
        new_group = model_out.groups[old_group].clone_group(model_out, new_group_id)

        seg = new_group.segments[0]
        sec = model_out.sections[seg.section_id] if seg.section_id != "" else None

        if m.is_cone:
            _, od_l, od_s, thk, thk1, thk2 = size
            if sec:
                sec.OD_L, sec.OD_S, sec.THK = od_l, od_s, thk
                sec.THK1, sec.THK2 = thk1, thk2
            else:
                print(f"Cone {member_id}: no section, cone not written")
        else:
            _, od, thk = size
            seg.OD, seg._THK = od, thk
            if sec:
                sec.OD, sec.THK = od, thk

        catalogue[(zone, size)] = new_group_id
        m.group_id = new_group_id
        changes[member_id] = (old_group, new_group_id, True)

    return model_out, changes

