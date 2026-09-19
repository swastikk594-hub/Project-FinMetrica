import pytest
from src.stats.performance_tests import apply_multiple_comparison_correction

def test_multiple_comparison_correction():
    p_values = [0.0000, 0.0100, 0.0400, 0.0600, 0.1500, 0.8000]
    
    # Bonferroni
    bonf = apply_multiple_comparison_correction(p_values, method='bonferroni')
    
    assert bonf[0][0] == 0.0  # 0 * 6 = 0
    assert bonf[0][1] is True
    
    assert bonf[1][0] == 0.06 # 0.01 * 6 = 0.06
    assert bonf[1][1] is False
    
    assert bonf[2][0] == 0.24 # 0.04 * 6 = 0.24
    assert bonf[2][1] is False
    
    assert bonf[5][0] == 1.0  # 0.8 * 6 capped at 1.0
    
    # BH
    bh = apply_multiple_comparison_correction(p_values, method='bh')
    
    assert bh[0][0] == 0.0
    assert bh[0][1] is True
    
    # p=0.01, rank 2: adj = min(0.01 * 6 / 2, next) = 0.03
    assert abs(bh[1][0] - 0.03) < 1e-6
    assert bh[1][1] is True
    
    # p=0.04, rank 3: adj = min(0.04 * 6 / 3, next) = 0.08
    assert abs(bh[2][0] - 0.08) < 1e-6
    assert bh[2][1] is False
    
    # Compare conservativeness
    for i in range(len(p_values)):
        assert bonf[i][0] >= bh[i][0] - 1e-6, "Bonferroni should be at least as conservative as BH"
        
    # verify a borderline p-value near 0.05 raw does not survive Bonferroni
    assert bonf[2][1] is False  # raw=0.04 doesn't survive Bonferroni across 6
