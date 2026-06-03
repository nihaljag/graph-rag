"""Generate a tiny synthetic specification PDF for testing the pipeline.

The PDF deliberately contains:
- a heading hierarchy with section numbers,
- a multi-page table (status codes) and a nested-header table,
- explicit cross-references (Section refs, a Figure/Table ref, and a named
  reference) plus a behavioral dependency (Abort -> Identify),

so you can verify table reconstruction and cross-reference resolution end to end.

Usage:
    python scripts/make_sample_pdf.py sample_spec.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path


def build(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    styles = getSampleStyleSheet()
    h1, h2, h3, body = styles["Heading1"], styles["Heading2"], styles["Heading3"], styles["BodyText"]
    story = []

    story += [Paragraph("Example Device Specification", styles["Title"]), Spacer(1, 12)]

    story += [Paragraph("3 Status Code Definitions", h1),
              Paragraph("This clause defines status codes returned by commands. "
                        "See Section 5.2 for the Abort command.", body), Spacer(1, 6)]
    rows = [["Code", "Name", "Description"]]
    for i in range(40):  # long enough to span pages
        rows.append([f"0x{i:02X}", f"Status {i}",
                     f"Condition number {i} describing a normative behavior that shall be reported."])
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story += [Paragraph("Table 7: Status Code Definitions", h3), t, PageBreak()]

    story += [Paragraph("5 Command Set", h1)]
    story += [Paragraph("5.1 Identify command", h2),
              Paragraph("The Identify command returns a data structure. Refer to "
                        "Table 7 for status codes. See Figure 2.", body), Spacer(1, 6)]
    nested = [
        ["Dword", "Bits 31:16", "Bits 15:0"],
        ["CDW10", "Reserved", "Controller ID"],
        ["CDW11", "Namespace ID", "CNS"],
    ]
    nt = Table(nested)
    nt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("SPAN", (1, 0), (2, 0)),  # nested header span
    ]))
    story += [Paragraph("Table 12: Identify Command Dwords", h3), nt, Spacer(1, 12)]

    story += [Paragraph("5.2 Abort command", h2),
              Paragraph("The Abort command requests abort of a previously submitted command. "
                        "If the command being aborted is an Identify command, then the "
                        "controller shall complete it per Section 5.1. Status is returned per "
                        "Status Code Definitions. The controller shall support this command.",
                        body)]

    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_spec.pdf")
    build(out)
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
