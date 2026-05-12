from pathlib import Path
import pandas as pd

def export_table(rows, path_base):
    Path(path_base).parent.mkdir(parents=True, exist_ok=True); df=pd.DataFrame(rows)
    df.to_csv(str(path_base)+'.csv', index=False); df.to_markdown(str(path_base)+'.md', index=False); df.to_latex(str(path_base)+'.tex', index=False); return df
