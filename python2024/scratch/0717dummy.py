n20 = 50
benzol = 100

def klassi(p_n20, p_benzol):
    if p_n20 > n20:
        if p_benzol > benzol:
            return "Both too high"
        else:
            return "N20 too high"
    else:
        if p_benzol > benzol:
            return "Benzol too high"
        else:
            return "Both fine"
        
        
    