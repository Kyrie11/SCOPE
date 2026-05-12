def save_placeholder_plot(path, title='SCOPE plot'):
    from pathlib import Path
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(); plt.title(title); plt.plot([0,1],[0,1]); plt.savefig(path); plt.close()
