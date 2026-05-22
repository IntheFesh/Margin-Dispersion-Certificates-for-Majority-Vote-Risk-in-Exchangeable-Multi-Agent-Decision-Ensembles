from pathlib import Path
def write_markdown(text,path):
 Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(text)
