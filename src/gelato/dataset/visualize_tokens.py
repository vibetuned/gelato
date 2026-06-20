# Standard library imports
import sys
import logging
import argparse
import colorsys
from pathlib import Path

# Third party imports
from PySide6.QtGui import (
    QFont,
    QColor,
    QPixmap,
    QShortcut,
    QKeySequence,
    QTextCharFormat,
    QTextCursor,
    QBrush,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QSplitter,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QApplication,
    QPlainTextEdit,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
)

# Local imports
from gelato.model.tokenizer import build_abc_tokenizer

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
    """Look for a sibling image matching the abc stem in the imgs folder."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def color_for_token(token: str) -> QColor:
    """Stable category-based color so similar tokens share a hue."""
    # Categorize tokens to make patterns visible at a glance
    categories = [
        ("ctrl", lambda t: t in {"<pad>", "</s>", "<s>", "<unk>"}),
        ("hdr", lambda t: t.endswith(":") and len(t) <= 2),
        ("ws", lambda t: t in {"\n", " "}),
        ("bar", lambda t: t in {"|", "||", "|]", "|:", ":|", "::", "[1", "[2"}),
        ("deco", lambda t: t.startswith("!") and t.endswith("!")),
        ("digit", lambda t: t.isdigit() or "/" in t),
        ("acc", lambda t: t in {"^", "^^", "_", "__", "="}),
        ("oct", lambda t: t in {",", ",,", "'", "''"}),
        ("note_up", lambda t: len(t) == 1 and "A" <= t <= "G"),
        ("note_lo", lambda t: len(t) == 1 and "a" <= t <= "g"),
        ("rest", lambda t: t in {"z", "Z", "X", "x", "y"}),
        ("group", lambda t: t in {"(", ")", "[", "]", "{", "}", '"', "-"}),
    ]
    palette = {
        "ctrl": "#888888",
        "hdr": "#ff9900",
        "ws": "#3a3a3a",
        "bar": "#ffcc00",
        "deco": "#ff66b2",
        "digit": "#33cccc",
        "acc": "#9933cc",
        "oct": "#6699ff",
        "note_up": "#33cc33",
        "note_lo": "#66ff66",
        "rest": "#cccccc",
        "group": "#ff3333",
    }
    for name, pred in categories:
        if pred(token):
            return QColor(palette[name])

    # Fallback: hash-based pastel color
    h = (sum(ord(c) for c in token) * 2654435761) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.95)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


class TokenVisualizer(QMainWindow):
    def __init__(self, dataset_dir: str, tokenizer_dir: str | None = None):
        super().__init__()
        self.setWindowTitle("Gelato - ABC Token Visualizer")
        self.resize(1500, 900)

        self.dataset_dir = Path(dataset_dir)
        self.abc_dir = self.dataset_dir / "abcs-strip"
        self.img_dir = self.dataset_dir / "imgs"

        # Build tokenizer in-memory (no save) unless a saved dir was supplied
        save_dir = tokenizer_dir if tokenizer_dir else None
        self.tokenizer = build_abc_tokenizer(save_dir=save_dir)
        self.unk_id = self.tokenizer.unk_token_id

        self.abc_paths: list[Path] = []
        self.current_index = 0

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

        # --- Tokenized pane ---
        tok_pane = QWidget()
        tok_layout = QVBoxLayout(tok_pane)
        tok_layout.addWidget(self._section_label("Tokenized (colored)"))
        self.tok_view = QTextEdit()
        self.tok_view.setReadOnly(True)
        self.tok_view.setFont(QFont("Monospace", 11))
        tok_layout.addWidget(self.tok_view)
        splitter.addWidget(tok_pane)

        # --- Token list pane (token → id) ---
        list_pane = QWidget()
        list_layout = QVBoxLayout(list_pane)
        list_layout.addWidget(self._section_label("Token / ID"))
        self.list_view = QPlainTextEdit()
        self.list_view.setReadOnly(True)
        self.list_view.setFont(QFont("Monospace", 10))
        list_layout.addWidget(self.list_view)
        splitter.addWidget(list_pane)

        splitter.setSizes([500, 350, 600, 300])

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

        # Shortcuts
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
            self.tok_view.clear()
            self.list_view.clear()
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
    # File loading & rendering
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

        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.render_tokens(tokens, ids)
        self.render_token_list(tokens, ids)

        unk_count = sum(1 for i in ids if i == self.unk_id)
        self.lbl_stats.setText(
            f"chars: {len(text)}   tokens: {len(tokens)}   <unk>: {unk_count}"
        )

    def render_tokens(self, tokens: list[str], ids: list[int]):
        """Render each token as a colored span with a thin separator."""
        self.tok_view.clear()
        cursor = self.tok_view.textCursor()

        base_bg = QColor("#1e1e1e")
        for tok, tid in zip(tokens, ids):
            display = tok
            if tok == "\n":
                # Visualise a newline token then emit a real line break
                self._insert_colored(cursor, "⏎", QColor("#444444"), base_bg)
                cursor.insertText("\n")
                continue
            if tok == " ":
                display = "·"
                color = QColor("#444444")
            else:
                color = color_for_token(tok)
                if tid == self.unk_id:
                    color = QColor("#ff0000")

            self._insert_colored(cursor, display, color, base_bg)
            # Thin separator so consecutive tokens are visually distinct
            cursor.insertText(" ")

    def _insert_colored(
        self, cursor: QTextCursor, text: str, fg: QColor, bg: QColor
    ):
        fmt = QTextCharFormat()
        fmt.setForeground(QBrush(fg))
        fmt.setBackground(QBrush(bg))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        cursor.setCharFormat(QTextCharFormat())

    def render_token_list(self, tokens: list[str], ids: list[int]):
        lines = []
        width = max((len(repr(t)) for t in tokens), default=4)
        for i, (tok, tid) in enumerate(zip(tokens, ids)):
            marker = "  <unk>" if tid == self.unk_id else ""
            lines.append(f"{i:>4}  {tid:>5}  {repr(tok):<{width}}{marker}")
        self.list_view.setPlainText("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tokenized ABC files from a dataset's abcs-strip folder."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset-stage-3",
        help="Path to a dataset directory containing an 'abcs-strip' subfolder.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Optional save dir for the tokenizer. If omitted, it is built in memory.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = TokenVisualizer(args.dataset, args.tokenizer_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
