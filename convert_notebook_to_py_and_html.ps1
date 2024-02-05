Set-Location "C:\Users\llag\source\repos\annat\ml-implementations"
jupyter nbconvert note.ipynb --to="python" --output-dir="\results" --output="note-convert.py"
jupyter nbconvert note.ipynb --to="html" --output-dir="\results" --output="note-convert.html"