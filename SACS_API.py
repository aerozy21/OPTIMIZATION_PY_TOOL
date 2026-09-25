import math
import sacs_extension as aux
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
from pathlib import Path
import shutil

class SACSModel:
    def __init__(self, input_file=None):
        self.joints = {}
        self.sections = {}
        self.groups = {}
        self.members = {}


        if input_file:
            self._read_model(input_file)
            self.path = input_file
            self.assign_attached_members()

    def _read_model(self, filepath):
        with open(filepath, 'r') as f:
            line = next_non_comment_line(f)
            while line:
                if line.startswith("*"):
                    line = next_non_comment_line(f)
                    continue
                if line.startswith("JOINT"):
                    # Joint id is common to all JOINT line types (cols 7-10)
                    joint_id = line[6:10].strip()
                    if not joint_id:
                        line = next_non_comment_line(f)
                        continue
                    fixity_field = line[54:60].strip()  
                    if fixity_field not in ("ELASTI", "PERSET"):
                        joint_id = line[6:10].strip()
                        x_str = line[11:18].strip()
                        y_str = line[18:25].strip()
                        z_str = line[25:32].strip()
                        x_dec = line[32:39].strip()
                        y_dec = line[39:46].strip()
                        z_dec = line[46:53].strip()   
                        x = aux.safe_float(x_str) + aux.safe_float(x_dec) / 100
                        y = aux.safe_float(y_str) + aux.safe_float(y_dec) / 100
                        z = aux.safe_float(z_str) + aux.safe_float(z_dec) / 100
                        joint = Joint(joint_id, self, x, y, z, fixity_field) 
                        self.add_joint(joint)
                        # 2) ELASTI / PERSET line -> assume joint already exists and attach raw line
                    else:
                        joint = self.joints[joint_id]  # will raise KeyError if not found (fine if you're confident)
                        if fixity_field == "ELASTI":
                            joint.elasti = line.rstrip("\n")
                        elif fixity_field == "PERSET":
                            joint.perset = line.rstrip("\n")
                
                if line.startswith("MEMBER") and line.strip() != "MEMBER" and not line.startswith("MEMBER OFFSET"):
                    memb2_line = ""
                    joint1_id = line[7:11].strip()
                    joint2_id = line[11:15].strip()
                    member_id = joint1_id + "-" +  joint2_id
                    offset_option = line[6:7].strip()
                    group_id = line[16:19].strip()
                    fixity1 = line[22:28].strip()
                    fixity2 = line[28:34].strip()
                    chord_angle = line[35:41].strip()
                    ref_joint = line[41:45].strip()
                    Length_option = line[46:47].strip()
                    Ly = line[51:55].strip()
                    Lz = line[55:59].strip()
                    flooded = line[45:46].strip()
                    offset1 = [0, 0, 0]
                    offset2 = [0, 0, 0]
                    offset_type = None
                    memb2_line = ""


                    # ---- Look ahead for MEMB2 regardless of offsets ----
                    line = next_non_comment_line(f)
                    if line.strip().startswith("MEMB2"):
                        memb2_line = line
                        line = next_non_comment_line(f)

                    # ---- OFFSET IMPLEMENTATION (only if offset option exists) ----
                    if joint1_id != "" and offset_option in ("1", "2"):
                        offset_type = {"1": "global", "2": "local"}[offset_option]
                        offset1 = None
                        offset2 = None

                        if line and line.strip().startswith("MEMBER OFFSETS"):
                            x1_offset = line[35:41].strip()
                            y1_offset = line[41:47].strip()
                            z1_offset = line[47:53].strip()
                            x2_offset = line[53:59].strip()
                            y2_offset = line[59:65].strip()
                            z2_offset = line[65:71].strip()
                            offset1 = [
                                aux.safe_float(x1_offset) / 100,
                                aux.safe_float(y1_offset) / 100,
                                aux.safe_float(z1_offset) / 100,
                            ]
                            offset2 = [
                                aux.safe_float(x2_offset) / 100,
                                aux.safe_float(y2_offset) / 100,
                                aux.safe_float(z2_offset) / 100,
                            ]
                    if member_id != "-":
                        member = Member(
                            member_id, self,
                            joint1_id, joint2_id, group_id,
                            offset_type, offset1, offset2,
                            fixity1, fixity2,
                            chord_angle, ref_joint,
                            Length_option, Ly, Lz,
                            flooded, memb2_line
                        )
    
                        self.add_member(member)
                        continue

                
                if line.startswith("GRUP"):
                    group_id = line[5:8].strip()
                    section_id = line[9:16].strip()
                    OD = line[17:23].strip()
                    THK = line[23:29].strip()
                    E = line[30:35].strip()
                    G = line[35:40].strip()
                    FY = line[40:45].strip()
                    Ky = line[51:55].strip()
                    Kz = line[55:59].strip()
                    flooded = line[69:70].strip()
                    density = line[70:76].strip()
                    seg_ratio = line[76:80].strip()
                    if group_id:
                        # get or create Group
                        group = self.groups.get(group_id)
                        if group is None:
                            group = Group(group_id, self)
                            self.add_group(group)

                        seg = GroupSegment(index=len(group.segments) + 1,section_id=section_id, OD=OD, _THK=THK, E=E, G=G, FY=FY, Ky=Ky, Kz=Kz, flooded=flooded, density=density, segment_ratio=seg_ratio, model=self)
                        group.add_segment(seg)
                
                if line.startswith("SECT"):
                    section_id = line[5:12].strip()
                    stype = line[15:18].strip()
                    section_data = {"section_id": section_id, "stype": stype}

                    if stype == "TUB":
                        section_data["OD"] = line[49:55].strip()
                        section_data["THK"] = line[55:60].strip()
                        section_data["OD_I"] = line[60:66].strip()
                        section_data["THK_I"] = line[66:71].strip()
                    elif stype == "CON":
                        section_data["OD_L"] = line[49:55].strip()
                        section_data["THK"] = line[55:60].strip()
                        section_data["OD_S"] = line[60:66].strip()
                        section_data["THK1"] = line[66:71].strip()
                        section_data["THK2"] = line[71:76].strip()

                    if section_id:
                        section = Section(**section_data)
                        self.add_section(section)
                line = next_non_comment_line(f)

    def add_joint(self, joint):
        self.joints[joint.Id] = joint
    
    def add_member(self, member):
        self.members[member.Id] = member

    def add_section(self, section):
        self.sections[section.Id] = section

    def add_group(self, group):
        self.groups[group.Id] = group
    
    def assign_attached_members(self):
        for member_id, member in self.members.items():
            start_joint_id = member_id[:4]
            end_joint_id = member_id[-4:]

            start_joint = self.joints.get(start_joint_id)
            end_joint = self.joints.get(end_joint_id)

            if start_joint:
                start_joint.AttachedMembers.append(member)
            if end_joint and end_joint_id != start_joint_id:
                end_joint.AttachedMembers.append(member)

    def save_as(self, new_name):
        """
        Save a clean copy of the original model file under a new name.
        If only a filename is provided, save it in the same directory as the original file.
        """

        if not hasattr(self, "path") or self.path is None:
            raise ValueError("This model was not loaded from a file, so it cannot be saved as a copy.")

        # Directory of the original file
        original_dir = os.path.dirname(self.path)

        # If user passed only a filename, save it in the same folder
        if not os.path.isabs(new_name):
            new_path = os.path.join(original_dir, new_name)
        else:
            new_path = new_name
        # Extract directory part of the final path
        target_dir = os.path.dirname(new_path)
        # Only create directory if it is not empty
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        # Copy the file
        shutil.copy2(self.path, new_path)
        # Update internal path
        self.path = new_path
        return new_path

    
    def make_unique_joint_id_4(self, joint_id):
        # Use the first character of the provided joint ID as prefix
        prefix = joint_id[0]
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # 1) Try classic numeric IDs first: prefix000–prefix999
        for i in range(1000):
            jid = f"{prefix}{i:03d}"
            if jid not in self.joints:
                return jid

        # 2) If numeric IDs exhausted, use full alphanumeric space
        for a in chars:
            for b in chars:
                for c in chars:
                    jid = f"{prefix}{a}{b}{c}"
                    if jid not in self.joints:
                        return jid

        raise RuntimeError(
            f"No available joint IDs left for prefix '{prefix}' "
            f"(all 46,656 combinations exhausted)"
        )


    def make_unique_group_id(self, base_id, extra_used=None, first_letter=None):
        """
        Unique 3‑character group ID generator using:
        - Capital letters A‑Z
        - Digits 0‑9

        Pattern: <L1><L2><C>
        Search order: C → L2 → L1
        """

        if len(base_id) != 3:
            raise ValueError(f"Base group id '{base_id}' is not 3 characters long")

        original_prefix = base_id[0]

        existing = set(self.groups.keys())
        if extra_used:
            existing |= set(extra_used)

        # Allowed characters
        first_letters  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        second_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        counter_chars  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Resolve starting indices
        if first_letter is not None:
            start_fl_index = first_letters.index(first_letter)
        else:
            start_fl_index = first_letters.index(original_prefix)

        start_sl_index = second_letters.index(original_prefix)

        # Wrap-around helper
        def wrap(seq, start):
            return seq[start:] + seq[:start]

        fl_seq = wrap(list(first_letters), start_fl_index)
        sl_seq = wrap(list(second_letters), start_sl_index)

        # FULL SEARCH SPACE: C → L2 → L1
        for ch in counter_chars:          # last digit changes fastest
            for sl in sl_seq:             # then second letter
                for fl in fl_seq:         # then first letter
                    gid = f"{fl}{sl}{ch}"
                    if gid not in existing:
                        return gid

        raise RuntimeError(
            f"No available group IDs left for base group '{base_id}' "
            f"after exhausting all uppercase letters and digits."
        )




    def make_unique_section_id(self, old_section):
        all_sections = self.sections.keys()
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

    def write_model(self, input_filename, mode=None, verbose=True):
        input_path = Path(input_filename)

        # If mode is provided, use controlled naming
        if mode is not None:
            stem = re.sub(r"sac", f"{mode}_SAC", input_path.stem, flags=re.IGNORECASE)
            output_name = stem + ".inp"
            output_path = input_path.with_name(output_name)

        else:
            # If mode is omitted, use the exact filename provided
            output_path = input_path

        raw = input_path.read_text(encoding="utf-8", errors="ignore")
        lines = raw.splitlines(keepends=True)

        # -------------------------
        # Build replacement JOINT block
        # -------------------------
        joint_block = ["JOINT\n"]
        for jnt in self.joints.values():
            for cmd in jnt.SACS_commands:
                joint_block.append(cmd if cmd.endswith("\n") else cmd + "\n")

        # -------------------------
        # Build replacement GRUP block
        # -------------------------
        grup_block = ["GRUP\n"]
        for grp in self.groups.values():
            for cmd in grp.SACS_commands:
                grup_block.append(cmd if cmd.endswith("\n") else cmd + "\n")

        # -------------------------
        # Build replacement MEMBER block
        # -------------------------
        member_block = ["MEMBER\n"]
        for mem in self.members.values():
            for cmd in mem.SACS_commands:
                member_block.append(cmd if cmd.endswith("\n") else cmd + "\n")

        out_lines = []
        joints_inserted = False
        groups_inserted = False
        members_inserted = False

        # -------------------------
        # SECT: patch only CON lines (do NOT replace whole SECT block)
        # -------------------------
        in_sect = False
        sect_replaced = 0
        written_sections = set()   # track which section IDs were written

        block_starts = ("JOINT", "GRUP", "MEMBER", "LOAD", "PILE", "SUPP", "END", "DYN", "MASS", "WAVE")

        for line in lines:

            # Detect entry to SECT block
            if not in_sect and line.startswith("SECT"):
                in_sect = True
                out_lines.append(line)
                continue

            if in_sect:

                # Detect exit from SECT block
                if line.startswith(block_starts):
                    in_sect = False

                    # 🔥 Append NEW CON sections not present in original file
                    for sec_id, sec_obj in self.sections.items():
                        if sec_obj.stype == "CON" and sec_id not in written_sections:
                            out_lines.extend(sec_obj.SACS_commands())
                            if verbose:
                                print(f"[SECT] Added NEW CON section: {sec_id}")

                    # Now process this line normally
                    # (do NOT continue — fall through)
                else:
                    # Inside SECT block
                    if line.startswith("SECT") and len(line) >= 18:
                        sec_id = line[5:12].strip()
                        sec_type = line[15:18].strip()

                        if sec_type == "CON":
                            sec_obj = self.sections.get(sec_id)
                            if sec_obj is not None and sec_obj.stype == "CON":
                                out_lines.extend(sec_obj.SACS_commands())
                                written_sections.add(sec_id)
                                sect_replaced += 1
                                if verbose:
                                    print(f"[SECT] Replaced CON section line: {sec_id}")
                                continue

                    # Default: keep original SECT line
                    out_lines.append(line)
                    continue


            # -------------------------
            # Normal replacements (JOINT / GRUP / MEMBER)
            # -------------------------

            # Replace entire JOINT section
            if line.startswith("JOINT"):
                if not joints_inserted:
                    out_lines.extend(joint_block)
                    joints_inserted = True
                continue

            # Replace entire GRUP section
            if line.startswith("GRUP"):
                if not groups_inserted:
                    out_lines.extend(grup_block)
                    groups_inserted = True
                continue

            # Replace entire MEMBER section (includes MEMBER OFFSETS lines)
            if line.startswith("MEMBER"):
                if not members_inserted:
                    out_lines.extend(member_block)
                    members_inserted = True
                continue

            out_lines.append(line)

        # If file had no JOINT / GRUP / MEMBER blocks, append them at the end
        if not joints_inserted:
            out_lines.extend(joint_block)
        if not groups_inserted:
            out_lines.extend(grup_block)
        if not members_inserted:
            out_lines.extend(member_block)

        output_path.write_text("".join(out_lines), encoding="utf-8")

        if verbose:
            print(f"[WRITE] Output written: {output_path}")
            print(f"[SECT] Replaced {sect_replaced} CON line(s) (left all other section types untouched).")


    


class Section:
    def __init__(self, section_id, stype, OD=None, THK=None, OD_L=None, OD_S=None, THK1=None, THK2=None, OD_I=None, THK_I=None):
        self.Id = section_id
        self.stype = stype
        self.THK = aux.safe_float(THK)  if THK else None
        if stype == "TUB":
            self.OD = float(OD) if OD else None
            self.THK = float(THK) if THK else None
            self.OD_I = float(OD_I) if OD_I else None
            self.THK_I = float(THK_I) if THK_I else None
        elif stype == "CON":
            self.OD_L = aux.safe_float(OD_L) if OD_L else None
            self.OD_S = aux.safe_float(OD_S) if OD_S else None
            self.THK1 = aux.safe_float(THK1) if THK1 else None
            self.THK2 = aux.safe_float(THK2) if THK2 else None
        # else:
        # raise ValueError(f"Unsupported section type '{stype}'. Only 'TUB' and 'CON' are supported.")
    
    def clone_section(self, model_out, new_id: str, *, add_to_model=True):
        """
        Create a fresh, fullFy independent copy of this Section.
        - New Section object
        - Same stype and geometry/thickness values
        - Optionally added to model_out
        """
        if self.stype == "TUB":
            new_sec = Section(section_id=new_id, stype="TUB",  OD=self.OD, THK=self.THK   )
        elif self.stype == "CON":
            new_sec = Section(section_id=new_id, stype="CON",OD_L=self.OD_L, OD_S=self.OD_S, THK1=self.THK1,THK2=self.THK2)
        else:
            raise ValueError(f"Unsupported section type '{self.stype}'")
        # Optionally register in the model
        if add_to_model:
            model_out.add_section(new_sec)
        return new_sec

    def SACS_commands(self):
        """
        Returns a list with one SECT card line (80 columns) + newline.
        Uses fixed columns per SECT table (1-based, inclusive).
        """
        stype = (self.stype or "").upper().strip()
        # 80-col buffer
        buf = list(" " * 80)
        # --- fixed SECT columns (per screenshot) ---
        _put(buf,     1,  4, "SECT")
        _put_raw(buf, 6, 12, self.Id)
        _put_raw(buf, 16, 18, stype)
        if stype == "TUB":
            _put(buf, 50, 55, self.OD)
            _put(buf, 56, 60, self.THK)
        elif stype == "CON":
            _put(buf, 50, 55, self.OD_L)
            _put(buf, 56, 60, self.THK)
            _put(buf, 61, 66, self.OD_S)
            _put(buf, 67, 71, self.THK1)
            _put(buf, 72, 76, self.THK2)
        else:
            raise ValueError(f"Unsupported section type '{stype}'. Only 'TUB' and 'CON' are supported.")
        return ["".join(buf).rstrip() + "\n"]
    
class Joint:
    def __init__(self, joint_id, model, X, Y, Z, fixity_field):
        self.Id = joint_id
        self.model = model
        self.X = float(X)
        self.Y = float(Y)
        self.Z = float(Z)
        self.fixity = fixity_field
        self.AttachedMembers = []

        self.elasti = ""
        self.perset = ""

    @property
    def coord(self):
        return [self.X, self.Y, self.Z]
    @property
    def is_joint(self):
        return self.AttachedMembers > 2

    @property
    def is_cantilever(self):
        return self.AttachedMembers == 1

    @property
    def SACS_commands(self):
        cmds = []

        # Build a fixed-width JOINT line buffer.
        # We need at least up to column 60.
        buf = list(" " * 80)  # 80 gives some slack; only cols up to 60 are important

        # Card name
        _put(buf, 1, 5, "JOINT")     # cols 1-5
        _put(buf, 6, 6, " ")         # col 6 is a space (optional, but keeps it tidy)

        # Joint id at cols 7-10
        _put_raw(buf, 7, 10, self.Id)

        # Split coords into integer and 2-dec "cent" parts to match your parser logic
        def split_int_cent(v):
            iv = math.floor(v) if v >= 0 else math.ceil(v)
            cent = (v - iv) * 100
            return iv, cent

        xi, xd = split_int_cent(self.X)
        yi, yd = split_int_cent(self.Y)
        zi, zd = split_int_cent(self.Z)

        # Integers (cols 12-32)
        _put(buf, 12, 18, xi)
        _put(buf, 19, 25, yi)
        _put(buf, 26, 32, zi)

        # Decimals (cols 33-53) stored as "cent" integers because your reader does /100
        _put(buf, 33, 39, xd)
        _put(buf, 40, 46, yd)
        _put(buf, 47, 53, zd)

        # Fixity flags (cols 55-60), ensure 6 chars max
        _put_raw(buf, 55, 60, (self.fixity or "")[:6])

        # Finalize JOINT line
        joint_line = "".join(buf).rstrip() + "\n"
        cmds.append(joint_line)

        # Append optional extra lines exactly as stored
        if self.elasti:
            cmds.append(self.elasti if self.elasti.endswith("\n") else self.elasti + "\n")
        if self.perset:
            cmds.append(self.perset if self.perset.endswith("\n") else self.perset + "\n")

        return cmds
    
class Member:
    def __init__(self, member_id, model, joint1_id, joint2_id, group_id, offset_type, offset1, offset2, fixity1, fixity2, chord_angle, ref_joint_id, length_option, Ly, Lz, flooded, memb2):
        self.Id = member_id
        self.model = model
        self.group_id = group_id
        self.joint1_id = joint1_id  
        self.joint2_id = joint2_id  
        self.offset_type = offset_type  
        self.offset1 = offset1  
        self.offset2 = offset2  
        self.fixity1 = fixity1  
        self.fixity2 = fixity2  
        self.chord_angle = chord_angle
        self.ref_joint_id = ref_joint_id
        self.length_option = length_option
        self.Ly = Ly
        self.Lz = Lz
        self.flooded = str(flooded).strip().upper() == "F"
        self.memb2 = memb2


    @property
    def joint1(self):
        return self.model.joints.get(self.joint1_id)
    @property
    def joint2(self):
        return self.model.joints.get(self.joint2_id)
    
    @property
    def group(self):
        return self.model.groups.get(self.group_id)

    @property
    def n_joints1(self):
        return len(self.joint1.AttachedMembers)

    @property
    def n_joints2(self):
        return len(self.joint2.AttachedMembers)
    
    @property
    def joint_vector(self):
        # Member axis unit vector from joint1 to joint2
        vec = np.array(self.joint2.coord) - np.array(self.joint1.coord)
        return vec / np.linalg.norm(vec)

    @property
    def rotational_matrix(self):
        x_local = self.joint_vector

        if self.ref_joint_id != "":
            # Define z_local from ref_joint
            ref_joint = self.model.joints.get(self.ref_joint_id)
            ref_vec_ = np.array(ref_joint.coord) - np.array(self.joint1.coord)
            z_proj = ref_vec_ - np.dot(ref_vec_, x_local) * x_local  # remove x_local component
            z_local = z_proj / np.linalg.norm(z_proj)

            y_local = np.cross(z_local, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)
            z_local /= np.linalg.norm(z_local)
        # Check if member is vertical (aligned with global Z)
        elif np.allclose(np.abs(x_local), [0, 0, 1], atol=1e-2):
            y_local = np.array([1, 0, 0]) # Local y follows Global X
            z_local = np.cross(x_local, y_local)
        else: 
            global_z = np.array([0, 0, 1])

            # Project global Z onto the plane perpendicular to x_local
            z_proj = global_z - np.dot(global_z, x_local) * x_local
            z_local = z_proj / np.linalg.norm(z_proj)

            # Ensure right-handed system
            y_local = np.cross(z_local, x_local)
            y_local /= np.linalg.norm(y_local)
            z_local = np.cross(x_local, y_local)  # Rebuild for orthogonality
            z_local /= np.linalg.norm(z_local)


        if self.chord_angle != "":
            # Now apply rotation about x_local by chord_angle (in degrees)
            theta_rad = np.deg2rad(float(self.chord_angle))
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)

            # Rotation around x_local (Rodrigues' formula, simplified for axis-aligned)
            # Rotating y_local and z_local around x_local
            y_rotated = cos_t * y_local + sin_t * z_local
            z_rotated = -sin_t * y_local + cos_t * z_local
            y_local = y_rotated
            z_local = z_rotated

        return np.array([x_local, y_local, z_local]) # 3x matrix

    @property
    def coord1(self):
        if self.offset1 is not None:
            if self.offset_type == "global":  # global
                offset = self.offset1
            elif self.offset_type == "local":  # local
                offset = self.rotational_matrix @ self.offset1
            else:
                offset = [0.0, 0.0, 0.0]
        else:
            offset = [0.0, 0.0, 0.0]

        return np.array(self.joint1.coord) + np.array(offset)
    

    @property
    def coord2(self):
        if self.offset2 is not None:
            if self.offset_type == "global":
                offset = self.offset2
            elif self.offset_type == "local":
                offset = self.rotational_matrix @ self.offset2
            else:
                offset = [0.0, 0.0, 0.0]
        else:
            offset = [0.0, 0.0, 0.0]

        return np.array(self.joint2.coord) + np.array(offset)

    @property
    def ref_vec(self):
        # Member axis unit vector from joint1 to joint2
        vec = np.array(self.coord2 - self.coord1)
        return vec / np.linalg.norm(vec)    

    def dir_vector(self, joint_id):
        if self.joint2_id == joint_id: # Member axis unit vector from joint1 to joint2
            vec = np.array(self.coord2 - self.coord1)
        elif self.joint1_id == joint_id:
             vec = np.array(self.coord1 - self.coord2)   
        else:
            vec = None        
        return vec / np.linalg.norm(vec)    
    
    def outer_vector(self, joint_id):
        if self.joint1_id == joint_id: # Member axis unit vector from joint1 to joint2
            vec = np.array(self.coord2 - self.coord1)
        elif self.joint2_id == joint_id:
            vec = np.array(self.coord1 - self.coord2)   
        else:
            vec = None        
        return vec / np.linalg.norm(vec)    
    
    def outer_coord(self, central_joint_id):
        if self.joint2_id == central_joint_id: # Member axis unit vector from joint1 to joint2
            outer_coords = self.coord1
        elif self.joint1_id == central_joint_id:
             outer_coords = self.coord2  
        else:
            outer_coords = None        
        return outer_coords

    def central_coord(self, central_joint_id):
        if self.joint2_id == central_joint_id: # Member axis unit vector from joint1 to joint2
            central_coords = self.coord2
        elif self.joint1_id == central_joint_id:
             central_coords = self.coord1  
        else:
            central_coords = None        
        return central_coords
    
    @property
    def length(self):
        return np.linalg.norm(self.coord2 - self.coord1)

    @property
    def is_leg(self):
        return self.group_id.startswith("L")

    @property
    def effective_Ly(self):
        if self.length_option == "L" and self.Ly != "":
            return float(self.Ly)
        elif self.length_option == "" and self.Ly != "":
            return self.length * float(self.Ly) # Ly interpreted as Ky in the absence of L marker in SACS input file
        elif self.length_option == "K":
            return self.length * float(self.group.Ky)
        else:
            return self.length

    @property
    def effective_Lz(self):
        if self.length_option == "L" and self.Lz != "":
            return float(self.Lz)
        elif self.length_option == "" and self.Lz != "":
            return self.length * float(self.Lz) 
        elif self.length_option == "K":
            return self.length * float(self.group.Kz)
        else:
            return self.length
         
    @property
    def is_tube(self):
        grp = self.group
        sect = grp.section
        # Group-level tube definition
        if grp.OD not in (None, ""):
            return True
        # Section-level tube definition
        if sect is not None and sect.stype == "TUB":
            return sect.OD_I in (None, "")
        return False

        
    @property
    def is_cone(self):
        if self.group.section != None:
            return self.group.section.stype == "CON"
        else:
            return False
    @property    
    def FY(self):
            return float(self.group.FY)        
    @property    
    def E(self):
            return float(self.group.E)
        
    @property    
    def OD(self):
        if self.is_tube:
            return float(self.group.OD)
    @property
    def OD_L(self):
        if self.is_cone:
            return float(self.group.section.OD_L)
    @property
    def OD_S(self):
        if self.is_cone:
            return float(self.group.section.OD_S)

    @property
    def THK1(self):
        if self.is_cone:
            return float(self.group.section.THK1)
    @property
    def THK2(self):
        if self.is_cone:
            return float(self.group.section.THK2)
    @property
    def THK(self):
        if self.is_tube:
            return float(self.group.THK)
        if self.is_cone:
            return float(self.group.section.THK)

    @property
    def section_id(self):
        if self.group is None:
            raise RuntimeError(f"Member '{self.Id}' has no group assigned")
        if self.group.section != None:
            return self.group.section.Id
        else:
            return ""
        
    def invert(self):
        model = self.model
        # Remove old key from dictionary
        old_key = self.Id
        if old_key in model.members:
            del model.members[old_key]
        # Swap joint IDs
        self.joint1_id, self.joint2_id = self.joint2_id, self.joint1_id
        # Swap end-dependent attributes
        self.offset1, self.offset2 = self.offset2, self.offset1
        self.fixity1, self.fixity2 = self.fixity2, self.fixity1
        # Update this member's ID
        self.Id = f"{self.joint1_id}-{self.joint2_id}"
        # Insert under new key
        model.members[self.Id] = self

    def split_member_at_ratio(self, ratio, group1_id=None, group2_id=None):
        r = float(ratio)
        if not (0.0 < r < 1.0):
            raise ValueError(f"split ratio must be between 0 and 1 (exclusive). Got {r}")
        original_id = self.Id
        g1 = group1_id if group1_id is not None else self.group_id
        g2 = group2_id if group2_id is not None else self.group_id

        # 1) Create split joint
        new_joint_id = self.model.make_unique_joint_id_4(self.Id[0])
        p = self.coord1 + r * (self.coord2 - self.coord1)
        new_joint = Joint(new_joint_id, self.model, p[0], p[1], p[2], "")
        self.model.joints[new_joint_id] = new_joint

        # 2) New member IDs = concatenation of joint IDs
        new_member1_id = f"{self.joint1_id}-{new_joint_id}"
        new_member2_id = f"{new_joint_id}-{self.joint2_id}"

        if new_member1_id in self.model.members or new_member2_id in self.model.members:
            raise RuntimeError(
                f"Member ID collision: '{new_member1_id}' or '{new_member2_id}' already exists."
            )
        
        m1 = Member(new_member1_id, self.model, self.joint1_id, new_joint_id, g1, self.offset_type, self.offset1, None, self.fixity1, "", self.chord_angle, self.ref_joint_id, self.length_option, self.Ly, self.Lz, self.flooded, self.memb2)
        m2 = Member(new_member2_id, self.model, new_joint_id, self.joint2_id, g2, self.offset_type, None, self.offset2, "", self.fixity2, self.chord_angle, self.ref_joint_id, self.length_option, self.Ly, self.Lz, self.flooded, self.memb2)

        self.model.members[new_member1_id] = m1
        self.model.members[new_member2_id] = m2

        j1 = self.joint1
        j2 = self.joint2
        # Add new members to joints
        j1.AttachedMembers.append(m1)
        new_joint.AttachedMembers.append(m1)
        new_joint.AttachedMembers.append(m2)
        j2.AttachedMembers.append(m2)
        
        print(self.Id)
        # 3) Remove original member
        self.delete_from_model()

        return new_joint, m1, m2

    def delete_from_model(self):
        """
        Safely delete this member from the model:
        - remove from both joints' AttachedMembers (which store Member objects)
        - remove from model.members
        """
        # 1. Remove this member object from joint1 and joint2
        for joint in (self.joint1, self.joint2):
            if joint is not None and hasattr(joint, "AttachedMembers"):
                if self in joint.AttachedMembers:
                    joint.AttachedMembers.remove(self)

        # 2. Remove from model.members
        member_id = self.Id
        if member_id in self.model.members:
            del self.model.members[member_id]


    @property
    def SACS_commands(self):
        """
        Returns:
        - MEMBER card (80 chars)
        - MEMB2 card if memb2_line != ""
        - MEMBER OFFSETS card (80 chars) if offsets are active

        Notes:
        - Most attributes are already stored as strings (deck-ready).
        - memb2_line is stored as the raw line from the input file ("" if absent) and is re-emitted as-is.
        - Offsets are stored in meters; SACS OFFSETS expects cm -> multiply by 100.
        - offset_type stored as 'global'/'local' but SACS wants '1'/'2' in col 7.
        """
        lines = []

        # -------------------
        # 1) MEMBER main card
        # -------------------
        buf = list(" " * 80)
        _put_raw(buf, 1, 6, "MEMBER")

        # Col 7: offset option in SACS language
        if self.offset_type == "global":
            offset_option = "1"
        elif self.offset_type == "local":
            offset_option = "2"
        else:
            offset_option = ""
        _put(buf, 7, 7, offset_option)

        # Connecting joints
        _put_raw(buf, 8, 11, self.joint1_id)
        _put_raw(buf, 12, 15, self.joint2_id)

        # MEMB2 flag: "A" if memb2 exists, otherwise blank
        memb2_flag = "A" if self.memb2 != "" else ""
        _put_raw(buf, 16, 16, memb2_flag)

        # Group label (unchanged)
        _put_raw(buf, 17, 19, self.group_id)

        # Fixities
        _put_raw(buf, 23, 28, self.fixity1)
        _put_raw(buf, 29, 34, self.fixity2)

        # Chord angle, reference joint, flood, length option
        _put(buf, 36, 41, self.chord_angle)
        _put(buf, 42, 45, self.ref_joint_id)
        flood_flag = "F" if self.flooded else ""
        _put(buf, 46, 46, flood_flag)
        _put(buf, 47, 47, self.length_option)

        # Ky/Ly and Kz/Lz
        _put(buf, 52, 55, self.Ly)
        _put(buf, 56, 59, self.Lz)

        lines.append("".join(buf))

        # -------------------
        # 1b) MEMB2 (optional)
        # -------------------
        if self.memb2 != "":
            lines.append(self.memb2.rstrip("\n"))

        # -------------------
        # 2) MEMBER OFFSETS
        # -------------------
        if self.offset_type in ("global", "local"):
            buf = list(" " * 80)
            _put(buf, 1, 6, "MEMBER")
            _put(buf, 8, 14, "OFFSETS")

            ox1, oy1, oz1 = (self.offset1 if self.offset1 else [0, 0, 0])
            ox2, oy2, oz2 = (self.offset2 if self.offset2 else [0, 0, 0])

            # SACS expects cm (metric) -> *100, each field width 6
            _put(buf, 36, 41, _f(ox1 * 100, 6))
            _put(buf, 42, 47, _f(oy1 * 100, 6))
            _put(buf, 48, 53, _f(oz1 * 100, 6))
            _put(buf, 54, 59, _f(ox2 * 100, 6))
            _put(buf, 60, 65, _f(oy2 * 100, 6))
            _put(buf, 66, 71, _f(oz2 * 100, 6))

            lines.append("".join(buf))

        return lines





class GroupSegment:
    def __init__(self, index, section_id,  OD, _THK, E, G, FY, Ky, Kz, flooded, density, segment_ratio, model ):
        self.index = index
        self.section_id = section_id
        self.OD = OD
        self._THK = _THK
        self.E = E
        self.G = G
        self.FY = FY
        self.Ky = Ky
        self.Kz = Kz
        flooded_flag = str(flooded).strip().upper() == "F"
        self.flooded = flooded_flag
        if flooded_flag == True:
            a = 1
        self.density = density
        self.segment_ratio = segment_ratio
        self.model = model
    @property
    def THK(self):
        # If THK is empty, pull from section definition
        if self.section_id != "":
            return self.model.sections[self.section_id].THK
        return self._THK
    # @property
    # def model(self):
    #     return self.group.model
    
    # @property
    # def section(self):
    #     return self.model.sections.get(self.section_id)
    
    @property
    def segment_signature(self):
        return ( self.section_id, self.OD, self.THK, self.E, self.G, self.FY, self.Ky, self.Kz, self.flooded, self.density )
     
class Group:
    def __init__(self, group_id: str, model):
        self.Id = group_id
        self.model = model
        self.segments = []   # list[GroupSegment]

    def add_segment(self, seg):
        self.segments.append(seg)

    @property
    def has_segments(self) -> bool:
        return len(self.segments) > 0

    @property
    def section_ids(self):
        return [s.section_id for s in self.segments]

    @property
    def section(self):
        if not self.segments:
            return None
        sec0 = self.segments[0].section_id
        if all(s.section_id == sec0 for s in self.segments):
            return self.model.sections.get(sec0)
        return None
    
    @property
    def members(self):
        """
        Returns a list of Member objects belonging to this group.
        """
        return [ m for m in self.model.members.values() if m.group_id == self.Id ]
    # -----------------------
    # Segment-aware accessor
    # -----------------------
    
    def get_section(self, segment_index: int):
        """
        Returns the Section object for the given segment index (0-based).
        """
        if not self.segments:
            return None
        try:
            seg = self.segments[segment_index]
        except IndexError:
            return None

        return self.model.sections.get(seg.section_id)
    # -----------------------
    # Backward compatibility
    # -----------------------
    @property
    def E(self):
        return self.segments[0].E if self.segments else None
    
    @property
    def OD(self):
        return self.segments[0].OD if self.segments else None
    
    @OD.setter
    def OD(self, value: float):
        v = float(value)
        for seg in self.segments:
            seg.OD = v

    @property
    def THK(self):
        return self.segments[0].THK if self.segments else None
    
    @THK.setter
    def THK(self, value: float):
        v = float(value)
        for seg in self.segments:
            seg._THK = v
    
    @property
    def FY(self):
        return self.segments[0].FY if self.segments else None
    # -----------------------
    # Segment-aware accessors
    # -----------------------
   
    @property
    def is_cone(self):
        if self.section is None:
            return False
        return self.section.stype == "CON"

    @property
    def is_tube(self):
        return self.OD != "" or (self.section is not None and self.section.stype == "TUB")
    
    def get_OD(self, segment_index: int):
        """
        segment_index: 0-based index
        """
        if not self.segments:
            return None
        try:
            return self.segments[segment_index].OD
        except IndexError:
            return None

    def get_THK(self, segment_index: int):
        """
        segment_index: 0-based index
        """
        if not self.segments:
            return None
        try:
            return self.segments[segment_index].THK
        except IndexError:
            return None
        
    def get_FY(self, segment_index: int):
        """
        segment_index: 0-based index
        """
        if not self.segments:
            return None
        try:
            return self.segments[segment_index].FY
        except IndexError:
            return None

    def delete(self):
        """
        Deletes this group from the model and clears any member assignments
        that reference this group.
        """
        # 1. Remove the group from the model dictionary
        if self.Id in self.model.groups:
            del self.model.groups[self.Id]


    def clone_group(self, model_out, new_id: str, *, add_to_model=True):
        """
        Create a fresh, fully independent copy of this Group.
        - New Group object
        - New list of GroupSegment objects
        - New segment instances (no shared references)
        - Same section/geometry/material values
        - Optionally added to model_out
        """

        # Create the new group
        new_group = Group(new_id, model_out)

        # Deep-copy each segment into a brand‑new GroupSegment
        for i, s in enumerate(self.segments, start=1):
            seg = GroupSegment( index=i,section_id=s.section_id, OD=s.OD, _THK=s.THK, E=s.E, G=s.G,FY=s.FY, Ky=s.Ky, Kz=s.Kz, flooded=s.flooded, density=s.density, segment_ratio=s.segment_ratio, model= model_out)
            new_group.add_segment(seg)

        # Optionally register in the output model
        if add_to_model:
            model_out.add_group(new_group)

        return new_group

    @property
    def SACS_commands(self):
        lines = []

        for seg in self.segments:
            buf = list(" " * 80)

            # GRUP card fixed columns (1-based, inclusive)
            _put(buf, 1, 4, "GRUP")
            _put_raw(buf, 6, 8, self.Id)

            _put_raw(buf, 10, 16, seg.section_id)
            _put(buf, 18, 23, seg.OD)
            _put(buf, 24, 29, seg.THK)

            _put(buf, 31, 35, seg.E)
            _put(buf, 36, 40, seg.G)
            _put(buf, 41, 45, seg.FY)
            _put(buf, 52, 55, seg.Ky)
            _put(buf, 56, 59, seg.Kz)
            flood_flag = "F" if seg.flooded else ""
            _put(buf, 70, 71, flood_flag)
            _put(buf, 71, 76, seg.density)        # density exact location
            _put(buf, 77, 80, seg.segment_ratio)  # segment ratio exact location

            lines.append("".join(buf))

        return lines 

    
def print_all_attributes(obj):
    attributes = [
        attr for attr in dir(obj)
        if not attr.startswith('_') and not callable(getattr(obj, attr))
    ]
    for attr in attributes:
        try:
            value = getattr(obj, attr)
            print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: <error accessing attribute> ({e})")

def _f(val, width):
    """Left align. If numeric (or numeric string), fill width with decimals."""
    if val is None:
        return " " * width

    try:
        num = float(val)
        # Fill width with decimals (left aligned)
        fmt = f"{{:<{width}.{max(width - 2, 0)}f}}"
        return fmt.format(num)[:width]
    except (ValueError, TypeError):
        return f"{str(val):<{width}}"[:width]


def _put(buf, col1, col2, val):
    """
    Insert val into buf at 1-based inclusive columns [col1..col2].
    """
    width = col2 - col1 + 1
    s = _f(val, width)
    i0 = col1 - 1
    i1 = col2
    buf[i0:i1] = list(s)

def _put_raw(buf, col1, col2, text):
    """
    Insert text into buf at 1-based inclusive columns [col1..col2],
    WITHOUT numeric conversion (preserves leading zeros etc).
    """
    width = col2 - col1 + 1
    s = (" " * width) if text is None else f"{str(text):<{width}}"[:width]
    i0 = col1 - 1
    i1 = col2
    buf[i0:i1] = list(s)

def next_non_comment_line(f):
    """
    Returns the next line in file `f` that does not start with '*'.
    Returns None if EOF is reached.
    """
    for line in f:
        if not line.startswith("*"):
            return line
    return None

def write_seastate_overrides(model, corroded_group_map, out_path="seastate_overides.txt"):
    """
    Write GRPOV overrides for each corroded duplicate group, using ORIGINAL group's
    hydro properties:

      - Cross section area  = original tube steel area
      - Displaced area      = original displaced area (π/4 * OD^2)
      - Dim for forces Y/Z  = original OD

    corroded_group_map: dict with keys (original_gid, corrosion_mode) and values new_gid
      e.g. {("A01","OUTER-ONLY"):"Z001", ("J03","INNER+OUTER"):"Z002", ...}
    """
    out_path = Path(out_path)

    lines_out = []
    lines_out.append("* Seastate overrides (GRPOV) generated from corroded_group_map\n")

    for (original_gid, _mode), new_gid in corroded_group_map.items():

        # --- get ORIGINAL group geometry ---
        grp = model.groups[original_gid]
        od = float(grp.OD)
        thk = float(grp.THK)

        # --- compute ORIGINAL properties ---
        id_ = od - 2.0 * thk
        a_cs = (math.pi / 4.0) * (od * od - id_ * id_)   # steel area
        a_disp = (math.pi / 4.0) * (od * od)             # displaced area

        # --- build GRPOV line ---
        buf = list(" " * 80)

        _put_raw(buf, 1, 5, "GRPOV")
        _put_raw(buf, 6, 7, "AL")
        _put_raw(buf, 16, 18, new_gid)

        _put(buf, 27, 33, a_cs)
        _put(buf, 34, 40, a_disp)
        _put(buf, 41, 46, od)
        _put(buf, 47, 52, od)

        lines_out.append("".join(buf).rstrip() + "\n")

    out_path.write_text("".join(lines_out), encoding="utf-8")
    return str(out_path)


