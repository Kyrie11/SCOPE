def deterministic_split(items, ratios=(0.8,0.1,0.1)):
    n=len(items); a=int(n*ratios[0]); b=int(n*(ratios[0]+ratios[1])); return items[:a], items[a:b], items[b:]
