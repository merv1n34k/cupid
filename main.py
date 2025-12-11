#!/usr/bin/env python3
"""Minimal technological scheme DXF generator from YAML."""

import yaml
import ezdxf
from ezdxf.enums import TextEntityAlignment
from dataclasses import dataclass, field
import re
import argparse
import math


# Page sizes in mm
PAGE_SIZES = {
    "a4": (210, 297),
    "a3": (297, 420),
    "a2": (420, 594),
    "a1": (594, 841),
    "a0": (841, 1189),
}

# Layout constants
HEADER_RATIO = 0.2  # 20% header, 80% content
CONDITION_HEIGHT_RATIO = 0.5  # conditions are 50% of main block height
BLOCK_HEIGHT = 30  # mm, fixed height for all blocks
BLOCK_WIDTH = 150  # mm, base width
VERTICAL_GAP = 15  # mm, gap between blocks
HORIZONTAL_INDENT = BLOCK_WIDTH * HEADER_RATIO  # indent for nested blocks
TEXT_MARGIN = 2  # mm, margin inside cells
IO_ARROW_LENGTH = 60  # mm, length of input/output arrows
IO_ARROW_LONG_SCALE = 1.5  # scale for even arrows in chess pattern
IO_TEXT_PADDING = 5  # mm, padding for iotext from block edge
COLUMN_GAP = 40  # mm, gap between columns
CONNECTOR_RADIUS = 10  # mm, radius of connector circles
PAGE_MARGIN = 50  # mm, margin from page edges
ARROW_HEAD_SIZE = 3  # mm, size of arrow head
CONTENT_PADDING = 12  # mm, padding inside content block for text


def wrap_text(text: str, max_width: float, char_width: float) -> list[str]:
    """Wrap text to fit within max_width, splitting by words."""
    if not text:
        return [""]

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        if len(test_line) * char_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def normalize_newlines(text: str) -> str:
    """Convert escaped \\n to actual newlines."""
    if text is None:
        return ""
    return text.replace("\\n", "\n")


@dataclass
class Block:
    """Represents a single process block."""

    id: str
    type: str
    name: str
    number: str  # e.g., "1", "1.1", "2.3.1"
    controls: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    nested: list["Block"] = field(default_factory=list)
    depth: int = 0  # nesting level


def parse_step(
    step_data: dict, parent_number: str, index: int, depth: int, parent_type: str = None
) -> Block:
    """Parse a single step from YAML into Block."""
    s = step_data.get("step", step_data)

    # Generate number
    if parent_number:
        number = f"{parent_number}.{index}"
    else:
        number = str(index)

    # Type is required for root steps, inherited from parent for nested
    block_type = s.get("type")
    if block_type is None:
        if parent_type is not None:
            block_type = parent_type
        else:
            raise ValueError(
                f"Step '{s.get('name', number)}' is missing required 'type' field"
            )

    block = Block(
        id=s.get("id", ""),
        type=block_type,
        name=normalize_newlines(s.get("name", "")),
        number=number,
        controls=s.get("controls", []),
        conditions=s.get("conditions", []),
        inputs=[normalize_newlines(i) for i in s.get("in", [])],
        outputs=[normalize_newlines(o) for o in s.get("out", [])],
        depth=depth,
    )

    # Parse nested steps
    nested_steps = s.get("steps", [])
    if isinstance(nested_steps, dict):
        nested_steps = [nested_steps]

    for i, ns in enumerate(nested_steps, start=1):
        block.nested.append(parse_step(ns, number, i, depth + 1, block_type))

    return block


def flatten_blocks(blocks: list[Block]) -> list[Block]:
    """Flatten nested blocks into a linear list for drawing."""
    result = []
    for block in blocks:
        result.append(block)
        if block.nested:
            result.extend(flatten_blocks(block.nested))
    return result


def get_block_height(block: Block) -> float:
    """Calculate total height of a block including conditions."""
    height = BLOCK_HEIGHT
    if block.conditions:
        height += BLOCK_HEIGHT * CONDITION_HEIGHT_RATIO
    return height


def draw_connector(msp, x: float, y: float, label: str, config: dict):
    """Draw a connector circle with label."""
    font_size = config["font_size"]
    font = config["font_family"]

    # Draw circle
    msp.add_circle((x, y), CONNECTOR_RADIUS, dxfattribs={"layer": "arrows"})

    # Draw label inside circle
    msp.add_text(
        label, height=font_size * 0.3, dxfattribs={"layer": "text", "style": font}
    ).set_placement((x, y), align=TextEntityAlignment.MIDDLE_CENTER)


def build_id_map(blocks: list[Block]) -> dict[str, str]:
    """Build a map from block id to 'TYPE NUMBER'. Mark duplicates with '?'."""
    id_counts: dict[str, list[str]] = {}

    def collect_ids(block_list: list[Block]):
        for block in block_list:
            if block.id:
                if block.id not in id_counts:
                    id_counts[block.id] = []
                id_counts[block.id].append(f"{block.type} {block.number}")
            if block.nested:
                collect_ids(block.nested)

    collect_ids(blocks)

    # Build final map: if duplicate, use '?'
    id_map = {}
    for bid, refs in id_counts.items():
        if len(refs) == 1:
            id_map[bid] = refs[0]
        else:
            id_map[bid] = "?"

    return id_map


def resolve_references(text: str, id_map: dict[str, str]) -> str:
    """Replace {id} references with actual type+number or '?'."""

    def replacer(match):
        ref_id = match.group(1)
        return id_map.get(ref_id, "?")

    return re.sub(r"\{([\w.]+)\}", replacer, text)


def parse_yaml(yaml_path: str) -> tuple[dict, list[Block]]:
    """Parse YAML file and return config and blocks."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = {
        "format": data.get("format", "a4").lower(),
        "font_family": data.get("font", {}).get("family", "ISOCPEUR"),
        "font_size": data.get("font", {}).get("size", 12),
    }

    blocks = []
    for i, step in enumerate(data.get("steps", []), start=1):
        blocks.append(parse_step(step, "", i, 0))

    return config, blocks


def draw_block(msp, block: Block, x: float, y: float, config: dict) -> float:
    """Draw a single block and return total height used."""
    header_width = BLOCK_WIDTH * HEADER_RATIO
    content_width = BLOCK_WIDTH * (1 - HEADER_RATIO)
    font_size = config["font_size"]
    io_font_size = max(font_size - 4, 2)  # same as iotext
    font = config["font_family"]

    # Apply indent based on depth
    x_offset = x + block.depth * HORIZONTAL_INDENT

    # Header box
    msp.add_line(
        (x_offset, y), (x_offset + header_width, y), dxfattribs={"layer": "steps"}
    )
    msp.add_line(
        (x_offset, y - BLOCK_HEIGHT),
        (x_offset + header_width, y - BLOCK_HEIGHT),
        dxfattribs={"layer": "steps"},
    )
    msp.add_line(
        (x_offset, y), (x_offset, y - BLOCK_HEIGHT), dxfattribs={"layer": "steps"}
    )
    msp.add_line(
        (x_offset + header_width, y),
        (x_offset + header_width, y - BLOCK_HEIGHT),
        dxfattribs={"layer": "steps"},
    )

    # Content box
    msp.add_line(
        (x_offset + header_width, y),
        (x_offset + BLOCK_WIDTH, y),
        dxfattribs={"layer": "steps"},
    )
    msp.add_line(
        (x_offset + header_width, y - BLOCK_HEIGHT),
        (x_offset + BLOCK_WIDTH, y - BLOCK_HEIGHT),
        dxfattribs={"layer": "steps"},
    )
    msp.add_line(
        (x_offset + BLOCK_WIDTH, y),
        (x_offset + BLOCK_WIDTH, y - BLOCK_HEIGHT),
        dxfattribs={"layer": "steps"},
    )

    # Header text: type + number (e.g., "ДР 1.1")
    header_text = f"{block.type} {block.number}"
    header_center_x = x_offset + header_width / 2
    header_center_y = y - BLOCK_HEIGHT / 3
    msp.add_text(
        header_text,
        height=font_size * 0.35,
        dxfattribs={"layer": "text", "style": font},
    ).set_placement(
        (header_center_x, header_center_y), align=TextEntityAlignment.MIDDLE_CENTER
    )

    # Controls text (below type/number)
    if block.controls:
        controls_text = ", ".join(block.controls)
        msp.add_text(
            controls_text,
            height=font_size * 0.3,
            dxfattribs={"layer": "text", "style": font},
        ).set_placement(
            (header_center_x, y - BLOCK_HEIGHT * 2 / 3),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

    # Content text (name) - with word wrapping
    content_center_x = x_offset + header_width + content_width / 2
    text_height = font_size * 0.35
    # Estimate char width (roughly 0.6 of height for typical fonts)
    char_width = text_height * 0.6
    # Available width with padding on both sides
    available_width = content_width - CONTENT_PADDING * 2

    # Wrap the name text
    wrapped_lines = wrap_text(block.name, available_width, char_width)
    num_lines = len(wrapped_lines)

    # Calculate vertical spacing for wrapped lines
    line_spacing = text_height * 1.2
    total_text_height = num_lines * line_spacing
    start_y = y - BLOCK_HEIGHT / 2 + total_text_height / 2 - text_height / 2

    for i, line in enumerate(wrapped_lines):
        line_y = start_y - i * line_spacing
        msp.add_text(
            line, height=text_height, dxfattribs={"layer": "text", "style": font}
        ).set_placement(
            (content_center_x, line_y), align=TextEntityAlignment.MIDDLE_CENTER
        )

    total_height = BLOCK_HEIGHT

    # Conditions section (if any)
    if block.conditions:
        cond_height = BLOCK_HEIGHT * CONDITION_HEIGHT_RATIO
        cond_y = y - BLOCK_HEIGHT

        # Conditions box (full width)
        msp.add_line(
            (x_offset, cond_y - cond_height),
            (x_offset + BLOCK_WIDTH, cond_y - cond_height),
            dxfattribs={"layer": "steps"},
        )
        msp.add_line(
            (x_offset, cond_y),
            (x_offset, cond_y - cond_height),
            dxfattribs={"layer": "steps"},
        )
        msp.add_line(
            (x_offset + BLOCK_WIDTH, cond_y),
            (x_offset + BLOCK_WIDTH, cond_y - cond_height),
            dxfattribs={"layer": "steps"},
        )

        # Conditions text - same size as iotext
        cond_text = "; ".join(block.conditions)
        msp.add_text(
            cond_text,
            height=io_font_size * 0.35,
            dxfattribs={"layer": "text", "style": font},
        ).set_placement(
            (x_offset + BLOCK_WIDTH / 2, cond_y - cond_height / 2),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

        total_height += cond_height

    return total_height


def draw_arrow_head(msp, x: float, y: float, direction: str):
    """Draw a filled arrow head (small triangle) pointing in direction (left/right)."""
    size = ARROW_HEAD_SIZE
    if direction == "right":
        # Arrow pointing right: tip at (x, y)
        points = [
            (x, y),
            (x - size, y + size / 2),
            (x - size, y - size / 2),
        ]
    else:  # left
        # Arrow pointing left: tip at (x, y)
        points = [
            (x, y),
            (x + size, y + size / 2),
            (x + size, y - size / 2),
        ]

    # Create filled hatch
    hatch = msp.add_hatch(color=7, dxfattribs={"layer": "arrows"})
    hatch.paths.add_polyline_path(points + [points[0]], is_closed=True)


def draw_input_output(
    msp, block: Block, x: float, y: float, config: dict, id_map: dict[str, str]
):
    """Draw input/output arrows and labels for a block."""
    font_size = config["font_size"]
    io_font_size = max(font_size - 4, 2)  # font_size - 4 for i/o text, min 2
    text_height = io_font_size * 0.35
    text_offset = text_height * 0.3  # small gap between text and line
    font = config["font_family"]

    x_offset = x + block.depth * HORIZONTAL_INDENT

    num_inputs = len(block.inputs)
    num_outputs = len(block.outputs)

    # Calculate spacing - account for two-line texts needing more space
    def calc_spacing(items):
        # Count how many have newline (need double space)
        multi_line = sum(1 for item in items if "\n" in item)
        single_line = len(items) - multi_line
        # Each multi-line needs ~2x vertical space
        effective_count = single_line + multi_line * 2
        if effective_count == 0:
            return BLOCK_HEIGHT, []

        spacing = BLOCK_HEIGHT / (effective_count + 1)
        positions = []
        pos = 1
        for item in items:
            if "\n" in item:
                positions.append(pos + 0.5)  # center of 2-unit space
                pos += 2
            else:
                positions.append(pos)
                pos += 1
        return spacing, positions

    # Chess pattern: for 3+ arrows, even indices (0-based) get 1.5x length
    # BUT only if at least one arrow has text below (contains \n)
    def get_arrow_length(idx: int, total: int, items: list[str]) -> tuple[float, float]:
        """Return (arrow_length, text_offset_from_start) for chess pattern."""
        # Check if any item has text below the arrow
        has_text_below = any("\n" in item for item in items)

        if total >= 3 and idx % 2 == 1 and has_text_below:
            # Even arrows (2nd, 4th...) are longer only if there's text below
            long_len = IO_ARROW_LENGTH * IO_ARROW_LONG_SCALE
            return long_len, long_len - IO_ARROW_LENGTH  # text offset by difference
        return IO_ARROW_LENGTH, 0

    # Inputs on left side
    if num_inputs > 0:
        spacing, positions = calc_spacing(block.inputs)
        for i, inp in enumerate(block.inputs):
            arrow_y = y - spacing * positions[i]

            # Get arrow length based on chess pattern
            arrow_len, text_extra_offset = get_arrow_length(i, num_inputs, block.inputs)

            arrow_start_x = x_offset - arrow_len
            arrow_end_x = x_offset

            # Draw horizontal arrow line
            msp.add_line(
                (arrow_start_x, arrow_y),
                (arrow_end_x, arrow_y),
                dxfattribs={"layer": "arrows"},
            )

            # Draw arrow head pointing right (into block)
            draw_arrow_head(msp, arrow_end_x, arrow_y, "right")

            # Resolve references
            inp_resolved = resolve_references(inp, id_map)

            # Text position - with padding before block (left side of arrow)
            # For longer arrows, text is in the outer portion
            text_center_x = arrow_start_x + (IO_ARROW_LENGTH - IO_TEXT_PADDING) / 2

            # Handle newline - two separate text objects
            if "\n" in inp_resolved:
                parts = inp_resolved.split("\n", 1)
                # Text above arrow
                msp.add_text(
                    parts[0],
                    height=text_height,
                    dxfattribs={"layer": "iotext", "style": font},
                ).set_placement(
                    (text_center_x, arrow_y + text_offset),
                    align=TextEntityAlignment.BOTTOM_CENTER,
                )
                # Text below arrow
                msp.add_text(
                    parts[1],
                    height=text_height,
                    dxfattribs={"layer": "iotext", "style": font},
                ).set_placement(
                    (text_center_x, arrow_y - text_offset),
                    align=TextEntityAlignment.TOP_CENTER,
                )
            else:
                # Single text above arrow
                msp.add_text(
                    inp_resolved,
                    height=text_height,
                    dxfattribs={"layer": "iotext", "style": font},
                ).set_placement(
                    (text_center_x, arrow_y + text_offset),
                    align=TextEntityAlignment.BOTTOM_CENTER,
                )

    # Outputs on right side
    if num_outputs > 0:
        spacing, positions = calc_spacing(block.outputs)
        for i, out in enumerate(block.outputs):
            arrow_y = y - spacing * positions[i]

            # Get arrow length based on chess pattern
            arrow_len, text_extra_offset = get_arrow_length(
                i, num_outputs, block.outputs
            )

            arrow_start_x = x_offset + BLOCK_WIDTH
            arrow_end_x = arrow_start_x + arrow_len

            # Draw horizontal arrow line
            msp.add_line(
                (arrow_start_x, arrow_y),
                (arrow_end_x, arrow_y),
                dxfattribs={"layer": "arrows"},
            )

            # Draw arrow head pointing right (out of block)
            draw_arrow_head(msp, arrow_end_x, arrow_y, "right")

            # Resolve references
            out_resolved = resolve_references(out, id_map)

            # Text position - for longer arrows, text is in the outer portion
            text_center_x = (
                arrow_start_x
                + IO_TEXT_PADDING
                + text_extra_offset
                + (IO_ARROW_LENGTH - IO_TEXT_PADDING) / 2
            )

            # Handle newline - two separate text objects
            if "\n" in out_resolved:
                parts = out_resolved.split("\n", 1)
                # Text above arrow
                msp.add_text(
                    parts[0],
                    height=text_height,
                    dxfattribs={"layer": "iotext", "style": font},
                ).set_placement(
                    (text_center_x, arrow_y + text_offset),
                    align=TextEntityAlignment.BOTTOM_CENTER,
                )
                # Text below arrow
                msp.add_text(
                    parts[1],
                    height=text_height,
                    dxfattribs={"layer": "iotext", "style": font},
                ).set_placement(
                    (text_center_x, arrow_y - text_offset),
                    align=TextEntityAlignment.TOP_CENTER,
                )
            else:
                # Single text above arrow
                msp.add_text(
                    out_resolved,
                    height=text_height,
                    dxfattribs={"layer": "iotext", "style": font},
                ).set_placement(
                    (text_center_x, arrow_y + text_offset),
                    align=TextEntityAlignment.BOTTOM_CENTER,
                )


def generate_dxf(yaml_path: str, output_path: str, generate_pdf: bool = False):
    """Main function: parse YAML and generate DXF."""
    config, blocks = parse_yaml(yaml_path)

    # Build id map for references
    id_map = build_id_map(blocks)

    # Create DXF document
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Setup layers
    doc.layers.add("steps", color=7)
    doc.layers.add("text", color=7)
    doc.layers.add("arrows", color=7)
    doc.layers.add("iotext", color=7)

    # Setup text style
    doc.styles.add(config["font_family"], font=config["font_family"])

    # Get page size
    page_w, page_h = PAGE_SIZES.get(config["format"], PAGE_SIZES["a4"])

    # Available height for content
    available_height = page_h - 2 * PAGE_MARGIN

    # Flatten blocks for linear drawing
    flat_blocks = flatten_blocks(blocks)

    # Calculate column assignments
    columns = []  # list of (start_idx, end_idx) for each column
    current_column_start = 0
    current_height = 0

    for i, block in enumerate(flat_blocks):
        block_h = get_block_height(block) + VERTICAL_GAP

        # Check if adding this block would exceed available height
        # Reserve space for connector at end of column
        connector_space = CONNECTOR_RADIUS * 2 + VERTICAL_GAP * 2

        if (
            current_height + block_h + connector_space > available_height
            and i > current_column_start
        ):
            # Start new column, previous block was last in column
            columns.append((current_column_start, i - 1))
            current_column_start = i
            current_height = block_h
        else:
            current_height += block_h

    # Add last column
    columns.append((current_column_start, len(flat_blocks) - 1))

    # Calculate column width (block + io arrows + gaps)
    # Use scaled arrow length to account for chess pattern when 3+ arrows
    max_arrow_len = IO_ARROW_LENGTH * IO_ARROW_LONG_SCALE
    column_width = max_arrow_len + BLOCK_WIDTH + max_arrow_len + COLUMN_GAP

    # Draw blocks column by column
    block_positions = {}  # store positions for connections

    # Extra left margin for first column to accommodate chess pattern arrows
    first_col_margin = max_arrow_len

    for col_idx, (col_start, col_end) in enumerate(columns):
        # Column X position - first column needs extra margin for input arrows
        if col_idx == 0:
            col_x = PAGE_MARGIN + first_col_margin
        else:
            col_x = PAGE_MARGIN + first_col_margin + col_idx * column_width
        current_y = page_h - PAGE_MARGIN

        # Draw incoming connector at top of column (if not first column)
        if col_idx > 0:
            prev_block = flat_blocks[col_start - 1]
            connector_label = f"{prev_block.type} {prev_block.number}"
            header_width = BLOCK_WIDTH * HEADER_RATIO
            connector_x = (
                col_x
                + flat_blocks[col_start].depth * HORIZONTAL_INDENT
                + header_width / 2
            )

            draw_connector(
                msp, connector_x, current_y - CONNECTOR_RADIUS, connector_label, config
            )

            # Arrow from connector to first block
            current_y -= CONNECTOR_RADIUS * 2 + VERTICAL_GAP
            msp.add_line(
                (connector_x, current_y + VERTICAL_GAP),
                (connector_x, current_y),
                dxfattribs={"layer": "arrows"},
            )

        # Draw blocks in this column
        for i in range(col_start, col_end + 1):
            block = flat_blocks[i]
            block_height = draw_block(msp, block, col_x, current_y, config)
            draw_input_output(msp, block, col_x, current_y, config, id_map)

            block_positions[block.number] = (
                current_y,
                current_y - block_height,
                block.depth,
                col_x,
            )
            current_y -= block_height + VERTICAL_GAP

        # Draw outgoing connector at bottom of column (if not last column)
        if col_idx < len(columns) - 1:
            next_block = flat_blocks[col_end + 1]
            connector_label = f"{next_block.type} {next_block.number}"
            last_block = flat_blocks[col_end]
            header_width = BLOCK_WIDTH * HEADER_RATIO

            # Position connector at last block's header center
            connector_x = (
                col_x + last_block.depth * HORIZONTAL_INDENT + header_width / 2
            )

            # Adjust for depth transition to next block
            next_depth = next_block.depth
            if next_depth > last_block.depth:
                connector_x += (next_depth - last_block.depth) * HORIZONTAL_INDENT

            # Arrow from last block to connector
            last_y_bottom = block_positions[last_block.number][1]
            msp.add_line(
                (connector_x, last_y_bottom),
                (connector_x, current_y + CONNECTOR_RADIUS),
                dxfattribs={"layer": "arrows"},
            )

            draw_connector(msp, connector_x, current_y, connector_label, config)

    # Draw flow connections within columns
    for col_idx, (col_start, col_end) in enumerate(columns):
        for i in range(col_start, col_end):
            current_block = flat_blocks[i]
            next_block = flat_blocks[i + 1]

            current_y_top, current_y_bottom, current_depth, current_col_x = (
                block_positions[current_block.number]
            )
            next_y_top, next_y_bottom, next_depth, _ = block_positions[
                next_block.number
            ]

            header_width = BLOCK_WIDTH * HEADER_RATIO

            # Base X position from current block's header center
            arrow_x = (
                current_col_x + current_depth * HORIZONTAL_INDENT + header_width / 2
            )

            # Adjust for depth transitions
            if next_depth > current_depth:
                arrow_x += (next_depth - current_depth) * HORIZONTAL_INDENT

            # Simple straight line down to next block's top
            msp.add_line(
                (arrow_x, current_y_bottom),
                (arrow_x, next_y_top),
                dxfattribs={"layer": "arrows"},
            )

    # Save DXF
    doc.saveas(output_path)

    # Generate PDF if requested
    if generate_pdf:
        pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
        try:
            from ezdxf.addons.drawing import Frontend, RenderContext
            from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
            from ezdxf.addons.drawing.config import (
                Configuration,
                ColorPolicy,
                BackgroundPolicy,
            )
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])

            # Configure rendering: black lines on white background
            config = Configuration(
                background_policy=BackgroundPolicy.WHITE,
                color_policy=ColorPolicy.BLACK,
            )
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out, config=config).draw_layout(msp, finalize=True)

            # Set white background
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            # Set figure size based on page format (mm to inches)
            page_w_in = page_w / 25.4
            page_h_in = page_h / 25.4
            fig.set_size_inches(page_w_in, page_h_in)

            fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
            plt.close(fig)
        except ImportError as e:
            import sys

            print(f"PDF generation requires matplotlib: {e}", file=sys.stderr)
        except Exception as e:
            import sys

            print(f"PDF generation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DXF technological scheme from YAML"
    )
    parser.add_argument("input", help="Input YAML file")
    parser.add_argument(
        "output", nargs="?", help="Output DXF file (default: input name with .dxf)"
    )
    parser.add_argument("--pdf", action="store_true", help="Also generate PDF output")

    args = parser.parse_args()

    yaml_file = args.input
    dxf_file = args.output if args.output else yaml_file.rsplit(".", 1)[0] + ".dxf"

    generate_dxf(yaml_file, dxf_file, generate_pdf=args.pdf)

