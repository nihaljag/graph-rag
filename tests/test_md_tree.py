from specgraph.chunk.md_tree import build_tree, split_section_number, Block

SAMPLE = """# 5 NVM Command Set

Intro text for chapter 5.

## 5.21 Abort command

The Abort command is used to abort commands.

| Field | Description |
| --- | --- |
| CDW10 | Command identifier |
| CDW11 | Reserved |

### 5.21.1 Command Completion

If the command being aborted is an Identify command, then foo.
"""


def test_section_number_parsing():
    assert split_section_number("5.21.1 Command Completion") == ("5.21.1", "Command Completion")
    assert split_section_number("Abort command") == (None, "Abort command")
    assert split_section_number("5 NVM Command Set") == ("5", "NVM Command Set")


def test_tree_structure_and_breadcrumb():
    root = build_tree(SAMPLE, doc_title="Spec")
    sections = list(root.iter_sections())
    titles = [s.title for s in sections]
    assert "NVM Command Set" in titles
    assert "Abort command" in titles
    assert "Command Completion" in titles

    completion = next(s for s in sections if s.title == "Command Completion")
    assert completion.number == "5.21.1"
    assert completion.breadcrumb == ["Spec", "NVM Command Set", "Abort command", "Command Completion"]


def test_table_is_atomic_block():
    root = build_tree(SAMPLE, doc_title="Spec")
    abort = next(s for s in root.iter_sections() if s.title == "Abort command")
    table_blocks = [b for b in abort.blocks if b.kind == "table"]
    assert len(table_blocks) == 1
    # All four pipe rows (header + sep + 2 data) survive in one block.
    assert table_blocks[0].text.count("\n") == 3
    assert "CDW10" in table_blocks[0].text and "CDW11" in table_blocks[0].text
