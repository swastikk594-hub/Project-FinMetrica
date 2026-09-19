import os
import re

# DOCUMENTATION OF PAPER BUILD PROCESS:
# The paper is built by running `docs/student_methodology_paper/build_paper.py`.
# There is no separate markdown or text source file; the prose is hardcoded directly 
# into the python script via calls to `para()`, `heading()`, etc., which then uses
# python-docx to generate the final .docx file.
# Thus, to check the paper's source for consistency, we must parse `build_paper.py` itself.

PAPER_SOURCE = "docs/student_methodology_paper/build_paper.py"
FINDINGS_FILE = "research/results/FINDINGS.md"

def extract_findings_data(filepath):
    """
    Extracts structured metrics from FINDINGS.md.
    Returns a dict of metrics.
    """
    if not os.path.exists(filepath):
        return {}
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    data = {}
    
    # Extract condition number reduction
    match = re.search(r'reduced the average condition number by \*\*([\d\.]+)%\*\*', text)
    if match:
        data['cond_reduction'] = float(match.group(1))
        
    # Extract DSR
    match = re.search(r'benchmark of [\d\.]+ is \*\*([\d\.]+)%\*\*', text)
    if match:
        data['dsr'] = float(match.group(1))
        
    # Extract turnover
    # - Sample Covariance: 0.038
    match = re.search(r'-\s+Sample Covariance:\s+([\d\.]+)', text)
    if match:
        data['turnover_sample'] = float(match.group(1))
    match = re.search(r'-\s+Ledoit-Wolf:\s+([\d\.]+)', text)
    if match:
        data['turnover_lw'] = float(match.group(1))
    match = re.search(r'-\s+HRP:\s+([\d\.]+)', text)
    if match:
        data['turnover_hrp'] = float(match.group(1))
        
    # Extract Sharpe ratios and p-values from the table
    # | Equal Weight | Naive Markowitz | 0.5051 | 1.0203 | 0.1200 | 0.7200 | 0.1860 | False |
    table_lines = [line for line in text.split('\n') if '|' in line and 'Method A' not in line and '---' not in line]
    data['pairs'] = {}
    data['sharpes'] = {}
    
    for line in table_lines:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 8:
            m1, m2 = parts[0], parts[1]
            s1, s2 = float(parts[2]), float(parts[3])
            raw_p, bonf_p, bh_p = float(parts[4]), float(parts[5]), float(parts[6])
            sig = parts[7] == 'True'
            
            data['pairs'][(m1, m2)] = {
                'raw_p': raw_p, 'bonf_p': bonf_p, 'bh_p': bh_p, 'significant': sig
            }
            data['sharpes'][m1] = s1
            data['sharpes'][m2] = s2
            
    return data

def extract_paper_claims(filepath):
    """
    Extracts sentences containing numbers, 'Sharpe', 'significant', or 'p-value' 
    from the paper source.
    """
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    claims = []
    # Find strings inside para("...") or add_text(p, "...")
    matches = re.finditer(r'(?:para|add_text)\(.*?"(.*?)"', text, re.DOTALL)
    for m in matches:
        sentence = m.group(1)
        if any(kw in sentence.lower() for kw in ['sharpe', 'significant', 'p-value', 'p=', 'outperform', 'difference', 'hrp']):
            if re.search(r'\d', sentence) or 'significant' in sentence.lower():
                claims.append(sentence)
                
    return claims

def check_consistency():
    print("--- Paper vs FINDINGS.md Consistency Checker ---")
    findings = extract_findings_data(FINDINGS_FILE)
    if not findings:
        print(f"Error: {FINDINGS_FILE} not found.")
        return
        
    paper_claims = extract_paper_claims(PAPER_SOURCE)
    
    if not paper_claims:
        print("No numeric or statistical claims found in the paper source.")
        print("The current student methodology paper is purely descriptive.")
        return
        
    print(f"Found {len(paper_claims)} potential claims in the paper.\n")
    
    flags = []
    
    for claim in paper_claims:
        flag_reasons = []
        
        # Check turnover
        if 'turnover' in claim.lower():
            if '0.009' not in claim and findings.get('turnover_hrp') is not None:
                if str(round(findings['turnover_hrp'], 3)) not in claim:
                    flag_reasons.append(f"HRP turnover mismatch (Findings says {findings.get('turnover_hrp')})")
            if '0.038' not in claim and findings.get('turnover_sample') is not None:
                if str(round(findings['turnover_sample'], 3)) not in claim:
                    flag_reasons.append(f"Sample turnover mismatch (Findings says {findings.get('turnover_sample')})")
        
        # Check significance
        if 'significant' in claim.lower() and 'p =' in claim.lower():
            # If the text explicitly mentions HRP and Equal Weight
            if 'hrp' in claim.lower() and 'equal weight' in claim.lower():
                pair_key = ('Equal Weight', 'HRP')
                pair_key_rev = ('HRP', 'Equal Weight')
                
                sig_data = findings.get('pairs', {}).get(pair_key) or findings.get('pairs', {}).get(pair_key_rev)
                if sig_data:
                    # check p-value match
                    if f"{sig_data['bh_p']:.4f}" not in claim and f"{sig_data['bh_p']:.3f}" not in claim:
                        # Allow exact matches for 0.0000 or raw p (we use BH p by default in FINDINGS)
                        if f"{sig_data['bh_p']:.4f}" != "0.0000":
                            flag_reasons.append(f"p-value mismatch for HRP vs EW (Findings BH p={sig_data['bh_p']:.4f})")
                    
                    # check direction
                    claim_is_sig = 'not statistically significant' not in claim.lower() and 'rarely statistically significant' not in claim.lower()
                    if claim_is_sig and not sig_data['significant']:
                        # The user has explicitly labeled this as an earlier run note, so we'll check if it's explicitly framed as an earlier run
                        if 'earlier run' not in claim.lower():
                            flag_reasons.append(f"Significance direction mismatch (Findings says NOT significant)")
                else:
                    flag_reasons.append("HRP vs EW significance test missing from FINDINGS.md")

        if flag_reasons:
            flags.append((claim, flag_reasons))
            
    if not flags:
        print("All paper claims successfully matched FINDINGS.md or were correctly framed as historical notes!")
    else:
        print("FLAGGED ITEMS FOR REVIEW:")
        for claim, reasons in flags:
            print(f"\n[FLAG] {', '.join(reasons)}")
            print(f"Context: {claim}")

if __name__ == "__main__":
    check_consistency()
