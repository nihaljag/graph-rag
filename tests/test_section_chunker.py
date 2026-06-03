from specgraph.chunk.section_chunker import chunk_markdown

DOC = """# 5 Commands

## 5.21 Abort command

Abort aborts a command.

### 5.21.1 Completion

If the command being aborted is an Identify command then return status.
"""


def _chunk(md, **kw):
    opts = dict(doc_title="Spec", target_tokens=50, max_tokens=80, prepend_breadcrumb=True)
    opts.update(kw)
    return chunk_markdown(md, **opts)


def test_one_chunk_per_leaf_section_with_breadcrumb():
    chunks = _chunk(DOC)
    titles = [c.title for c in chunks]
    assert "Abort command" in titles
    assert "Completion" in titles
    abort = next(c for c in chunks if c.title == "Abort command")
    assert abort.breadcrumb == "Spec > Commands > Abort command"
    assert "> Context: Spec > Commands > Abort command" in abort.text
    assert abort.section_no == "5.21"


def test_deterministic_ids_and_ordering():
    a = _chunk(DOC)
    b = _chunk(DOC)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.order for c in a] == sorted(c.order for c in a)
    # id encodes zero-padded order
    assert a[0].chunk_id.split("__")[0] == "0001"


def test_large_table_split_repeats_header_and_stays_table():
    rows = "\n".join(f"| {i} | value-{i} long token padding here |" for i in range(60))
    md = "# 7 Status Codes\n\n## 7.1 Codes\n\n| Code | Meaning |\n| --- | --- |\n" + rows + "\n"
    chunks = _chunk(md, target_tokens=60, max_tokens=80)
    codes = [c for c in chunks if c.title == "Codes"]
    assert len(codes) > 1, "oversized table section should split into parts"
    for c in codes:
        # Every part keeps the header row so the table remains self-describing.
        assert "| Code | Meaning |" in c.text
        assert c.n_parts == len(codes)


def test_small_section_not_split():
    chunks = _chunk(DOC)
    for c in chunks:
        assert c.n_parts == 1
