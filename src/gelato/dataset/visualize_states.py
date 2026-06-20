# Standard library imports
import sys
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from PySide6.QtGui import QFont, QColor, QPixmap, QShortcut, QKeySequence, QBrush
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QSplitter,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QApplication,
    QPlainTextEdit,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QGraphicsView,
    QGraphicsScene,
)

# Local imports
from gelato.model.tokenizer import build_abc_tokenizer
from gelato.model.static import ABCGrammarCompiler, ABCLogitsProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScoreView(QGraphicsView):
    """Zoomable / pannable score preview."""

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet("background-color: #2b2b2b; border: none;")
        self._zoom = 0
        self._has_image = False

    def wheelEvent(self, event):
        if not self._has_image:
            return
        if event.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
            self._zoom += 1
        else:
            self.scale(1 / 1.15, 1 / 1.15)
            self._zoom -= 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._has_image and self._zoom == 0:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def show_image(self, path: Path | None):
        self._scene.clear()
        self._zoom = 0
        if path is None or not path.exists():
            self._has_image = False
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._has_image = False
            return
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._scene.addPixmap(pixmap)
        self._has_image = True
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


def find_score_image(img_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# Names mirror the comments in src/gelato/model/static.py
STATE_NAMES = {
    0: "idle",
    1: "grace_inside",
    2: "after_grace",
    3: "chord_open",
    4: "chord_text",
    5: "dec_bound_open",
    6: "dec_name",
    7: "after_dec",
    8: "after_dec_short",
    9: "after_accid",
    10: "after_pitch",
    11: "after_octave",
    12: "after_duration",
    13: "after_tie_slur",
    14: "chord_close",
    15: "header_value",
    16: "bracket_open",
    17: "UNCATEGORIZED",
}

# Distinct hue per state — picked so adjacent states don't clash
STATE_COLORS = {
    0: "#3a3a3a",
    1: "#7e57c2",
    2: "#9575cd",
    3: "#ef6c00",
    4: "#ffa726",
    5: "#c2185b",
    6: "#e91e63",
    7: "#ad1457",
    8: "#880e4f",
    9: "#6a1b9a",
    10: "#2e7d32",
    11: "#388e3c",
    12: "#1b5e20",
    13: "#00695c",
    14: "#ff5722",
    15: "#ff8f00",
    16: "#1565c0",
    17: "#b71c1c",
}


def compute_state(processor: ABCLogitsProcessor, input_ids: torch.Tensor):
    """Return (primary_state, effective_state, adjustments) for a single-batch
    prefix.

    This is a thin wrapper over the processor's OWN state logic
    (ABCLogitsProcessor.effective_states) rather than a re-implementation, so
    the visualizer can never drift from static.py. `adjustments` is derived
    generically from the primary→effective difference — any future state rule
    added to the processor shows up here automatically.
    """
    if input_ids.shape[1] == 0:
        last_token = processor.bos_token_id
    else:
        last_token = int(input_ids[0, -1].item())

    primary_state = int(processor.token_to_state[last_token].item())
    effective_state = int(processor.effective_states(input_ids)[0].item())

    adjustments: list[str] = []
    if effective_state != primary_state:
        adjustments.append(
            f"state adjusted: {primary_state} ({STATE_NAMES.get(primary_state, '?')}) "
            f"→ {effective_state} ({STATE_NAMES.get(effective_state, '?')})"
        )

    return primary_state, effective_state, adjustments


class StateTraceVisualizer(QMainWindow):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.setWindowTitle("Gelato - ABC State-Trace Visualizer")
        self.resize(1600, 950)

        self.dataset_dir = Path(dataset_dir)
        self.abc_dir = self.dataset_dir / "abcs-strip"
        self.img_dir = self.dataset_dir / "imgs"

        # Tokenizer + processor — built once
        self.tokenizer = build_abc_tokenizer(save_dir=None)
        compiler = ABCGrammarCompiler(self.tokenizer)
        self.token_to_state, self.state_to_allowed, _ = compiler.build_state_tensors(
            device="cpu"
        )
        self.processor = ABCLogitsProcessor(
            self.token_to_state, self.state_to_allowed, self.tokenizer
        )
        self.vocab_size = len(self.tokenizer)

        # id → token string (for displaying allowed-next sets)
        self.id_to_token = {i: t for t, i in self.tokenizer.get_vocab().items()}

        self.abc_paths: list[Path] = []
        self.current_index = 0
        # Per-step records cached for the current file
        self.trace: list[dict] = []

        self.setup_ui()
        self.load_dataset(self.abc_dir)

    # ────────────────────────────────────────────────────────────────────
    # UI
    # ────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # --- Score preview pane ---
        score_pane = QWidget()
        score_layout = QVBoxLayout(score_pane)
        score_layout.addWidget(self._section_label("Score"))
        self.score_view = ScoreView()
        score_layout.addWidget(self.score_view)
        splitter.addWidget(score_pane)

        # --- Raw ABC pane ---
        raw_pane = QWidget()
        raw_layout = QVBoxLayout(raw_pane)
        raw_layout.addWidget(self._section_label("Raw ABC"))
        self.raw_view = QPlainTextEdit()
        self.raw_view.setReadOnly(True)
        self.raw_view.setFont(QFont("Monospace", 11))
        raw_layout.addWidget(self.raw_view)
        splitter.addWidget(raw_pane)

        # --- State-trace table ---
        trace_pane = QWidget()
        trace_layout = QVBoxLayout(trace_pane)
        trace_layout.addWidget(
            self._section_label("State trace (state BEFORE each token is emitted)")
        )
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["#", "token", "id", "primary", "effective", "state name", "allowed?"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setFont(QFont("Monospace", 10))
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setStretchLastSection(False)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        trace_layout.addWidget(self.table)
        splitter.addWidget(trace_pane)

        # --- Detail pane (allowed-next + adjustments) ---
        detail_pane = QWidget()
        detail_layout = QVBoxLayout(detail_pane)
        detail_layout.addWidget(self._section_label("Details for selected step"))
        self.detail_view = QPlainTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setFont(QFont("Monospace", 10))
        detail_layout.addWidget(self.detail_view)
        splitter.addWidget(detail_pane)

        splitter.setSizes([450, 280, 750, 400])

        # --- Bottom nav ---
        nav = QHBoxLayout()
        self.lbl_info = QLabel("0 / 0 : No files")
        self.lbl_info.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_stats = QLabel("")
        self.lbl_stats.setFont(QFont("Arial", 10))

        btn_open = QPushButton("Open dataset…")
        btn_prev = QPushButton("< Prev")
        btn_next = QPushButton("Next >")
        btn_open.clicked.connect(self.choose_dataset)
        btn_prev.clicked.connect(self.prev_file)
        btn_next.clicked.connect(self.next_file)

        nav.addWidget(self.lbl_info)
        nav.addSpacing(20)
        nav.addWidget(self.lbl_stats)
        nav.addStretch()
        nav.addWidget(btn_open)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        root.addLayout(nav)

        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_file)
        QShortcut(QKeySequence(Qt.Key_D), self, self.next_file)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_file)
        QShortcut(QKeySequence(Qt.Key_A), self, self.prev_file)
        QShortcut(QKeySequence("Ctrl+O"), self, self.choose_dataset)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 12, QFont.Bold))
        return lbl

    # ────────────────────────────────────────────────────────────────────
    # Dataset handling
    # ────────────────────────────────────────────────────────────────────
    def choose_dataset(self):
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select dataset directory (must contain an abcs-strip subfolder)",
            str(self.dataset_dir),
        )
        if not chosen:
            return
        chosen_path = Path(chosen)
        abc_dir = (
            chosen_path
            if chosen_path.name == "abcs-strip"
            else chosen_path / "abcs-strip"
        )
        self.load_dataset(abc_dir)

    def load_dataset(self, abc_dir: Path):
        if not abc_dir.exists():
            logger.warning(f"Folder not found: {abc_dir}")
            self.abc_paths = []
            self.current_index = 0
            self.lbl_info.setText(f"0 / 0 : {abc_dir} not found")
            self.raw_view.clear()
            self.table.setRowCount(0)
            self.detail_view.clear()
            return

        self.abc_dir = abc_dir
        self.dataset_dir = abc_dir.parent
        self.img_dir = self.dataset_dir / "imgs"
        self.abc_paths = sorted(abc_dir.glob("*.abc"))
        self.current_index = 0
        if self.abc_paths:
            self.load_file()
        else:
            self.lbl_info.setText(f"0 / 0 : No .abc in {abc_dir}")

    # ────────────────────────────────────────────────────────────────────
    # File handling
    # ────────────────────────────────────────────────────────────────────
    def next_file(self):
        if not self.abc_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.abc_paths)
        self.load_file()

    def prev_file(self):
        if not self.abc_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.abc_paths)
        self.load_file()

    def load_file(self):
        path = self.abc_paths[self.current_index]
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return

        self.lbl_info.setText(
            f"{self.current_index + 1} / {len(self.abc_paths)} : {path.name}"
        )
        self.score_view.show_image(find_score_image(self.img_dir, path.stem))
        self.raw_view.setPlainText(text)

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        self.trace = self.build_trace(tokens, ids)
        self.populate_table(self.trace)

        n_blocked = sum(1 for r in self.trace if not r["allowed"])
        n_adjusted = sum(1 for r in self.trace if r["adjustments"])
        self.lbl_stats.setText(
            f"steps: {len(self.trace)}   blocked: {n_blocked}   adjusted: {n_adjusted}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Trace construction
    # ────────────────────────────────────────────────────────────────────
    def build_trace(self, tokens: list[str], ids: list[int]) -> list[dict]:
        """For each token, capture the state the processor is in immediately
        BEFORE that token is emitted, plus whether the token would be allowed
        and the full set of allowed-next IDs at that step."""
        trace: list[dict] = []
        for step in range(len(ids)):
            prefix = torch.tensor([ids[:step]], dtype=torch.long)
            primary, effective, adj = compute_state(self.processor, prefix)

            # Authoritative allowed-token mask straight from the processor —
            # reuse the effective state we already computed. Any mask-level
            # rule (e.g. grace stickiness) is applied inside allowed_mask().
            cur = torch.tensor([effective], dtype=torch.long)
            allowed_mask = self.processor.allowed_mask(prefix, current_states=cur)[0]

            # Surface grace stickiness in the adjustments column when it opened
            # up the closing '}' that the raw effective state would have blocked.
            gc = self.processor._grace_close_id
            if gc is not None and bool(allowed_mask[gc].item()) and not bool(
                self.state_to_allowed[effective][gc].item()
            ):
                adj.append("grace-sticky: '}' allowed (inside { … })")

            allowed_ids = torch.nonzero(allowed_mask, as_tuple=False).flatten().tolist()
            tid = ids[step]
            is_allowed = bool(allowed_mask[tid].item()) if tid < self.vocab_size else False

            trace.append(
                {
                    "step": step,
                    "token": tokens[step],
                    "id": tid,
                    "primary": primary,
                    "effective": effective,
                    "adjustments": adj,
                    "allowed": is_allowed,
                    "allowed_ids": allowed_ids,
                }
            )
        return trace

    def populate_table(self, trace: list[dict]):
        self.table.setRowCount(len(trace))
        for row, r in enumerate(trace):
            self._set_item(row, 0, str(r["step"]))
            self._set_item(row, 1, repr(r["token"]))
            self._set_item(row, 2, str(r["id"]))

            primary_item = self._set_item(row, 3, str(r["primary"]))
            self._color_cell(primary_item, r["primary"])

            effective_item = self._set_item(row, 4, str(r["effective"]))
            self._color_cell(effective_item, r["effective"])

            name_item = self._set_item(row, 5, STATE_NAMES.get(r["effective"], "?"))
            self._color_cell(name_item, r["effective"])

            allowed_text = "✓" if r["allowed"] else "✗ BLOCKED"
            allowed_item = self._set_item(row, 6, allowed_text)
            if not r["allowed"]:
                allowed_item.setForeground(QBrush(QColor("#ff5555")))
                allowed_item.setFont(QFont("Monospace", 10, QFont.Bold))

        self.table.resizeColumnsToContents()
        if trace:
            self.table.selectRow(0)

    def _set_item(self, row: int, col: int, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, col, item)
        return item

    def _color_cell(self, item: QTableWidgetItem, state_id: int):
        color = QColor(STATE_COLORS.get(state_id, "#444444"))
        item.setBackground(QBrush(color))
        # Pick readable foreground
        luminance = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
        item.setForeground(QBrush(QColor("#000000" if luminance > 140 else "#ffffff")))

    # ────────────────────────────────────────────────────────────────────
    # Selection → detail pane
    # ────────────────────────────────────────────────────────────────────
    def on_row_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self.trace:
            self.detail_view.clear()
            return
        row = rows[0].row()
        if row >= len(self.trace):
            return
        r = self.trace[row]

        lines = []
        lines.append(f"step       : {r['step']}")
        lines.append(f"token      : {r['token']!r}  (id={r['id']})")
        lines.append(
            f"primary    : {r['primary']:>2}  ({STATE_NAMES.get(r['primary'], '?')})"
        )
        lines.append(
            f"effective  : {r['effective']:>2}  ({STATE_NAMES.get(r['effective'], '?')})"
        )
        lines.append(
            "allowed?   : " + ("yes" if r["allowed"] else "NO — token blocked by grammar")
        )
        lines.append("")
        lines.append("adjustments applied:")
        if r["adjustments"]:
            for a in r["adjustments"]:
                lines.append(f"  • {a}")
        else:
            lines.append("  (none — primary state used as-is)")
        lines.append("")

        allowed_tokens = [
            self.id_to_token.get(i, f"<id:{i}>") for i in r["allowed_ids"]
        ]
        lines.append(f"allowed next tokens ({len(allowed_tokens)}):")
        # Sort for stable presentation
        for tok in sorted(allowed_tokens, key=lambda s: (s.startswith("<"), s)):
            lines.append(f"  {tok!r}")

        self.detail_view.setPlainText("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Walk each ABC file through ABCLogitsProcessor and visualize the "
            "state at every step (primary, effective, adjustments, allowed-next)."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset-stage-3",
        help="Path to a dataset directory containing an 'abcs-strip' subfolder.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = StateTraceVisualizer(args.dataset)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
