# Scripts

This directory contains utility scripts for the lionagi project.

## Available Scripts

### 1. Code Concatenation and Compression (`concat.py`)

To run codebase concatenation and compression:

1. Ensure you have `uv` installed in your environment. You can install it using pip:
   ```bash
   pip install uv
   ```

2. The script uses `OpenRouter` to conduct compression, you need to have
   `OPENROUTER_API_KEY` set in your `.env`. You can get it from
   [OpenRouter](https://openrouter.ai/).

3. Install the required dependencies and run:
   ```bash
   uv run scripts/concat.py
   ```

### 2. OpenAI Models Update (`update_openai_models.py`)

To update the OpenAI models from the latest OpenAPI specification:

1. From the project root directory, run:
   ```bash
   python scripts/update_openai_models.py
   ```

This script will:
- Install the required `datamodel-code-generator` tool
- Generate fresh OpenAI models from the latest schema
- Apply necessary post-processing (imports, type aliases, warning suppressions)
- Verify the generated models can be imported
- Provide file size and line count statistics

**Note**: The generated `openai_models.py` file is automatically excluded from git tracking and will be regenerated during CI/CD builds.
