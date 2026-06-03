from specgraph.query.xref_expand import XrefGraph, expand

ROWS = [
    {"src_chunk_id": "A", "dst_chunk_id": "B", "edge_type": "references_section",
     "ref_text": "Section 2", "resolved": True, "weight": 1.0},
    {"src_chunk_id": "B", "dst_chunk_id": "C", "edge_type": "references_section",
     "ref_text": "Section 3", "resolved": True, "weight": 1.0},
    {"src_chunk_id": "X", "dst_chunk_id": "Y", "edge_type": "references_named",
     "ref_text": "Foo", "resolved": False, "weight": 0.8},  # unresolved: ignored
]


def _tok(_cid):
    return 10


def test_depth_one_reaches_only_direct():
    g = XrefGraph.from_rows(ROWS)
    hits = expand(g, {"A": 1.0}, depth=1, hop_decay=0.5, token_of=_tok, token_budget=10_000)
    ids = {h.chunk_id for h in hits}
    assert ids == {"B"}


def test_depth_two_follows_chain_and_decays():
    g = XrefGraph.from_rows(ROWS)
    hits = expand(g, {"A": 1.0}, depth=2, hop_decay=0.5, token_of=_tok, token_budget=10_000)
    by_id = {h.chunk_id: h for h in hits}
    assert set(by_id) == {"B", "C"}
    assert by_id["B"].hop == 1 and by_id["C"].hop == 2
    # Direct reference outranks indirect.
    assert by_id["B"].score > by_id["C"].score


def test_in_edges_are_traversed():
    g = XrefGraph.from_rows(ROWS)
    hits = expand(g, {"C": 1.0}, depth=1, hop_decay=0.5, token_of=_tok, token_budget=10_000)
    assert {h.chunk_id for h in hits} == {"B"}
    assert hits[0].direction == "xref_in"


def test_unresolved_edges_excluded():
    g = XrefGraph.from_rows(ROWS)
    hits = expand(g, {"X": 1.0}, depth=2, hop_decay=0.5, token_of=_tok, token_budget=10_000)
    assert hits == []


def test_token_budget_limits_expansion():
    g = XrefGraph.from_rows(ROWS)
    # Budget below one neighbor cost but we always keep at least the first.
    hits = expand(g, {"A": 1.0}, depth=2, hop_decay=0.5, token_of=lambda c: 1000,
                  token_budget=1)
    assert len(hits) == 1 and hits[0].chunk_id == "B"
