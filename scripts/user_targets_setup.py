import os
import re
import pandas as pd

solved_meta_file = "" #input file containing tcrs
output_dir = "user_targets/" # change if needed

organism = 'human'
mhc_class = 1

required_cols = {
    'Peptide', 'HLA', 'Va', 'Ja', 'CDR3a',
    'Vb', 'Jb', 'CDR3b'
}

os.makedirs(output_dir, exist_ok=True)

def normalize_hla(raw):
    """
    Convert VDJdb HLA strings into TCRdock format:
      HLA-A0201     → A*02:01
      HLA-B*0801    → B*08:01
      B0702         → B*07:02
      HLA-A*02:01   → A*02:01
    """
    raw = str(raw).strip()

    # Remove HLA- or HLA_ prefixes
    raw = re.sub(r'^(HLA[-_])', '', raw)

    # Already correct (A*02:01)
    if re.match(r'^[ABCE]\*\d{2}:\d{2}$', raw):
        return raw

    # Case: B0702 → B*07:02
    m = re.match(r'^([ABCE])(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

    # Case: A*0201 → A*02:01
    m = re.match(r'^([ABCE])\*(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

    # Case: A0201 → A*02:01
    m = re.match(r'^([ABCE])(\d{2})(\d{2})$', raw)
    if m:
        return f"{m.group(1)}*{m.group(2)}:{m.group(3)}"

    raise ValueError(f"Unrecognized HLA format: {raw}")


df = pd.read_csv(solved_meta_file)

for idx, row in df.iterrows():

    if any(pd.isna(row[col]) or str(row[col]).strip()=="" for col in required_cols):
        print(f"Skipping row {idx} — missing fields")
        continue

    try:
        normalized_hla = normalize_hla(row['HLA'])
    except Exception as e:
        print(f"Skipping row {idx}: Bad HLA → {row['HLA']} ({e})")
        continue

    target = pd.DataFrame([{
        "organism": organism,
        "mhc_class": mhc_class,
        "mhc": normalized_hla,
        "peptide": row["Peptide"],
    
        "va": row["Va"],
        "ja": row["Ja"],
        "cdr3a": row["CDR3a"],
    
        "vb": row["Vb"],
        "jb": row["Jb"],
        "cdr3b": row["CDR3b"],
    }])

    outpath = os.path.join(output_dir, f"{idx}.tsv")
    target.to_csv(outpath, sep="\t", index=False)
    print(f"Wrote: {outpath} → HLA={normalized_hla}")
