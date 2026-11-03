from sprout.rag.ingest import build_index, load_manifest

PAGE = """<html><head><title>T</title><style>p{color:red}</style><script>var secret=1;</script></head>
<body><nav>Home Menu Login</nav><main><h1>Cherry powdery mildew</h1>
<p>Early infections are small, round, paler spots on leaves that can curl slightly.</p>
<script>evil()</script></main><footer>Copyright</footer></body></html>"""


def test_html_page_becomes_text_without_navigation_or_scripts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "p.html").write_text(PAGE, encoding="utf-8")
    m = tmp_path / "m.csv"
    m.write_text("file,crop,diseases,title,publisher,source_url,date\np.html,cherry,powdery mildew,T,P,https://x.edu/p,\n")
    build_index(load_manifest(m), raw, tmp_path / "index")
    text = (tmp_path / "index" / "chunks.jsonl").read_text(encoding="utf-8")
    assert "paler spots on leaves" in text
    assert "Login" not in text and "evil" not in text and "secret" not in text and "Copyright" not in text
