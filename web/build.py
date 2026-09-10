"""inline data.js into index.html and strip the wrapper, for publishing."""
import re, sys
from pathlib import Path
WEB = Path(__file__).resolve().parent
html = (WEB / "index.html").read_text()
data = (WEB / "data.js").read_text()
html = html.replace('<script src="data.js"></script>', f"<script>\n{data}\n</script>")
body = html[html.index("<title>"):html.index("</body>")].replace("</head>\n<body>\n", "")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else WEB / "artifact.html"
out.write_text(body)
print(f"wrote {out} ({len(body)//1024} kb)")
