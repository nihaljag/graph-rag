from specgraph.chunk.section_chunker import chunk_markdown
from specgraph.chunk.xref import process_chunks, build_section_index

DOC = """# 5 Commands

## 5.21 Abort command

The Abort command behaves per Section 5.21.1. See Status Code Definitions for results.
Refer to Table 45 for field layout. See Figure 12 too.

### 5.21.1 Identify handling

If the command being aborted is an Identify command, return a status code.

Table 45: Field layout

| Field | Description |
| --- | --- |
| A | B |

Figure 12: Abort flow

## 8 Status Code Definitions

Status codes are defined here.
"""


def _prep(**kw):
    chunks = chunk_markdown(DOC, doc_title="Spec", target_tokens=200, max_tokens=400,
                            prepend_breadcrumb=False)
    opts = dict(named_references=["Status Code Definitions"],
                inline_resolved_titles=True, mentioned_entities=True)
    opts.update(kw)
    edges = process_chunks(chunks, **opts)
    return chunks, edges


def test_section_reference_resolves_to_target_chunk():
    chunks, edges = _prep()
    by_id = {c.chunk_id: c for c in chunks}
    abort = next(c for c in chunks if c.title == "Abort command")
    target = next(c for c in chunks if c.section_no == "5.21.1")
    sec_refs = [r for r in abort.refs_out if r["ref_type"] == "references_section"]
    assert any(r["target_chunk_id"] == target.chunk_id and r["resolved"] for r in sec_refs)
    # An explicit traversable edge exists src->dst.
    assert any(e["src_chunk_id"] == abort.chunk_id and e["dst_chunk_id"] == target.chunk_id
               and e["edge_type"] == "references_section" for e in edges)


def test_named_reference_resolves():
    chunks, edges = _prep()
    abort = next(c for c in chunks if c.title == "Abort command")
    status = next(c for c in chunks if c.title == "Status Code Definitions")
    named = [r for r in abort.refs_out if r["ref_type"] == "references_named"]
    assert any(r["target_chunk_id"] == status.chunk_id for r in named)


def test_figure_and_table_resolve_to_caption_owner():
    chunks, edges = _prep()
    abort = next(c for c in chunks if c.title == "Abort command")
    owner = next(c for c in chunks if c.section_no == "5.21.1")  # holds the captions
    fig = [r for r in abort.refs_out if r["ref_type"] == "references_figure"]
    tbl = [r for r in abort.refs_out if r["ref_type"] == "references_table"]
    assert any(r["target_chunk_id"] == owner.chunk_id for r in fig)
    assert any(r["target_chunk_id"] == owner.chunk_id for r in tbl)


def test_refs_in_backlinks_populated():
    chunks, _ = _prep()
    target = next(c for c in chunks if c.section_no == "5.21.1")
    abort = next(c for c in chunks if c.title == "Abort command")
    assert any(b["src_chunk_id"] == abort.chunk_id for b in target.refs_in)


def test_inline_enrichment_adds_title():
    chunks, _ = _prep(inline_resolved_titles=True)
    abort = next(c for c in chunks if c.title == "Abort command")
    assert "Section 5.21.1 (Identify handling)" in abort.text


def test_unresolved_reference_recorded_not_invented():
    chunks = chunk_markdown("# 1 A\n\nSee Section 9.9.9 which does not exist.\n",
                            doc_title="Spec", target_tokens=200, max_tokens=400,
                            prepend_breadcrumb=False)
    edges = process_chunks(chunks, named_references=[], inline_resolved_titles=True,
                           mentioned_entities=True)
    ref = chunks[0].refs_out[0]
    assert ref["resolved"] is False
    assert ref["target_chunk_id"] is None


def test_mentioned_entities_generic_from_titles():
    chunks, _ = _prep()
    abort = next(c for c in chunks if c.title == "Abort command")
    names = {m["name"].lower() for m in abort.mentioned_entities}
    # "Status Code Definitions" is another section title mentioned in this chunk.
    assert any("status code" in n for n in names)
