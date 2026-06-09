from specgraph.llm import call_log


def setup_function(_fn):
    call_log.reset_stats()


def test_label_maps_task_from_system_prompt():
    msgs = [{"role": "system", "content": "You are extracting entities and relationships."},
            {"role": "user", "content": "TEXT"}]
    assert call_log.label(msgs) == "extract_graph"

    claim = [{"role": "system", "content": "Extract claims about the entity."}]
    assert call_log.label(claim) == "extract_claims"

    report = [{"role": "system", "content": "Write a community report summarizing."}]
    assert call_log.label(report) == "community_report"


def test_label_falls_back_to_snippet_for_string():
    text = "Some unusual instruction that matches no known task category here"
    assert call_log.label(text).startswith("Some unusual instruction")


def test_begin_end_track_counts():
    n, lbl = call_log.begin([{"role": "system", "content": "extract entities"}])
    stats = call_log.call_stats()
    assert stats["started"] == 1 and stats["in_flight"] == 1
    call_log.end(n, lbl, 0.1, failed=False)
    stats = call_log.call_stats()
    assert stats["in_flight"] == 0 and stats["done"] == 1 and stats["failed"] == 0


def test_failed_call_counted():
    n, lbl = call_log.begin("hi")
    call_log.end(n, lbl, 0.1, failed=True)
    assert call_log.call_stats()["failed"] == 1
