# CuPID

CuPID is a recursive acronym for *"Cupid's Piping and Instrumentation Diagram"*. Its purpose is to automate drawing technical and apparatus schemes with a definitive `yaml` configuration. It produces `dxf` format, which is open source and can be used by most CAD software.

## Idea

The idea behind CuPID is so you:

- *Don't waste your time* with routine schemes
- *Make beautiful, clean schemes* your classmates will be jealous about
- Program will automatically adjust pipelines, scales, text and apparatus positions
- Define custom apparatus library (or use our default) to make your schemes understandable
- Forget manual tweaks and let the program do its job

## WARNING: Alpha State

As of now, the program draws flow diagrams for processing steps. It is solely a proof of concept and upon author's sufficient time will be transformed into a proper P&ID drawing engine.

## Requirements

This project uses and highly recommends `uv` to manage `python` and its packages.

Dependencies:
- `ezdxf` — DXF file generation
- `pyyaml` — YAML parsing
- `matplotlib` — PDF export (optional)

## How to Use

1. Clone this repository
2. Run `uv sync` to install all packages
3. Run `uv run main.py config.yaml [output.dxf] [--pdf]` to generate the scheme

### Command Line Options

```
usage: main.py input.yaml [output.dxf] [--pdf]

positional arguments:
  input         Input YAML configuration file
  output        Output DXF file (default: input name with .dxf extension)

options:
  --pdf         Also generate PDF output alongside DXF
```

## YAML Configuration

The configuration file defines the document settings and process steps hierarchy.

### Document Settings

```yaml
format: "a1"          # Paper size: a0, a1, a2, a3, a4
font:
  family: "ISOCPEUR"  # Font family (system font name)
  size: 14            # Base font size in points
```

### Steps Structure

Steps are defined as a hierarchical list. Each step can contain nested steps.

```yaml
steps:
  - step:
      id: "unique_id"           # Optional: unique identifier for references
      type: "SP"                # Step type (e.g., SP, TP, etc.)
      name: "Step Name"         # Step name displayed in content block
      controls:                 # Optional: list of control indicators
        - "Ct"
        - "Cc"
      conditions:               # Optional: process conditions (shown below block)
        - "T=30min"
        - "t=90°C"
      in:                       # Optional: input streams
        - "Input material"
        - "Water\nt=25°C"       # Use \n for text above/below arrow
      out:                      # Optional: output streams
        - "to {next_step}"      # Use {id} to reference other steps
        - "Waste to WW"
      steps:                    # Optional: nested sub-steps
        - step:
            name: "Sub-step 1"
            # ... nested step properties
```

### Step Properties

| Property | Required | Description |
|----------|----------|-------------|
| `id` | No | Unique identifier for cross-references |
| `type` | Yes | required for root steps, inherited from parent for nested|
| `name` | Yes | Step name displayed in content block |
| `controls` | No | List of control point indicators (e.g., Ct, Cc, Cb) |
| `conditions` | No | Process conditions shown below the main block |
| `in` | No | List of input streams (arrows from left) |
| `out` | No | List of output streams (arrows to right) |
| `steps` | No | List of nested sub-steps |

### Cross-References

Use `{id}` syntax in `in` or `out` fields to reference other steps:

```yaml
steps:
  - step:
      id: "mixing"
      name: "Mixing"
      out:
        - "Product to {sterilization}"

  - step:
      id: "sterilization"
      name: "Sterilization"
      in:
        - "From {mixing}"
```

This will automatically resolve to the step's type and number (e.g., "From DR 1").

If an ID is not found or defined multiple times, it will be replaced with `?`.

### Multi-line Arrow Text

Use `\n` in input/output text to split text above and below the arrow line:

```yaml
in:
  - "Steam\nP=0.2 MPa"    # "Steam" above arrow, "P=0.2 MPa" below
  - "Water t=25°C"         # Single line above arrow
```

### Automatic Features

- **Numbering**: Steps are automatically numbered (1, 1.1, 1.1.1, etc.)
- **Type inheritance**: Nested steps inherit parent's type if not specified
- **Column wrapping**: If scheme exceeds page height, it wraps to multiple columns with connector circles
- **Chess pattern arrows**: When 3+ arrows exist and any has multi-line text, alternating arrows are extended to prevent overlap

## Example

See `sample.yaml` for a complete English example of a biotechnology process flow diagram.

## Output

The generator creates:

- **DXF file**: Compatible with AutoCAD, LibreCAD, QCAD, NanoCAD, and other CAD software
- **PDF file** (optional): For quick preview and printing

### Layers

The DXF file uses the following layers:
- `steps` — Block outlines (rectangles)
- `text` — Block text (headers, names, conditions)
- `arrows` — Flow lines and arrow heads
- `iotext` — Input/output stream labels

## License

Distributed under MIT license, see `LICENSE` for more.
