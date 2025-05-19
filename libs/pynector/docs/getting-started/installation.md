# Installation

Pynector is available on PyPI and can be installed using pip or other Python
package managers.

## Basic Installation

```bash
pip install pynector
```

## Using uv

```bash
uv pip install pynector
```

## With Optional Dependencies

Pynector offers several optional dependency groups that you can install based on
your needs:

### Observability Support

```bash
pip install "pynector[observability]"
```

### Tracing with Zipkin

```bash
pip install "pynector[zipkin]"
```

### Tracing with OTLP

```bash
pip install "pynector[otlp]"
```

### SDK Support (OpenAI, Anthropic)

```bash
pip install "pynector[sdk]"
```

### Documentation Tools

```bash
pip install "pynector[docs]"
```

### All Optional Dependencies

To install Pynector with all optional dependencies:

```bash
pip install "pynector[all]"
```

## Development Installation

For development, you might want to install in editable mode:

```bash
git clone https://github.com/ohdearquant/pynector.git
cd pynector
uv pip install -e ".[dev]"
```

## Requirements

Pynector requires Python 3.9 or later.
