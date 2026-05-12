def summarize_selection(rows):
    import pandas as pd
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()
