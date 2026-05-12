import re, pathlib

def test_readme_scripts_exist():
    text=pathlib.Path('README.md').read_text(); scripts=set(re.findall(r'python (scripts/[\w_]+\.py)', text))
    assert scripts
    for s in scripts: assert pathlib.Path(s).exists(), s

def test_required_configs_exist():
    for p in ['configs/data/womd_waymax.yaml','configs/model/scope_full.yaml','configs/train/train_surface.yaml','configs/eval/eval_selection.yaml']:
        assert pathlib.Path(p).exists()
