# tw-code-qa

Traditional Chinese Code-QA Dataset Conversion System using Multi-Agent Architecture

## Description

This project is a system for converting and processing Traditional Chinese Code-QA datasets using a multi-agent architecture built with LangChain and LangGraph.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ai-twinkle/tw-code-qa.git
   cd tw-code-qa
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

   For development dependencies:
   ```bash
   uv sync --extra dev
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` file with your actual API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   **Important**: API keys are required in production mode (default). For development mode, you can run without API keys using `--environment development`.

## Usage

First, download the dataset:

```bash
uv run python scripts/download_dataset.py
```

Then, run the main script with the dataset path and **always specify output directory** to avoid mixing different datasets:

### Processing Different Datasets

**Important**: Always specify `--output-dir` for each dataset to prevent mixing results from different datasets.

- **Educational Instruct Dataset**:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_educational_instruct --dataset-type opencoder --output-dir output/educational_instruct
  ```

- **Evol Instruct Dataset**:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_evol_instruct --dataset-type opencoder --output-dir output/evol_instruct
  ```

- **McEval Instruct Dataset**:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_mceval_instruct --dataset-type opencoder --output-dir output/mceval_instruct
  ```

- **Package Instruct Dataset**:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_package_instruct --dataset-type opencoder --output-dir output/package_instruct
  ```

### Other Usage Examples

- Test mode (process only first 10 records):
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_educational_instruct --output-dir output/test_run --max-records 10 --environment development
  ```

- Production mode with full processing:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_package_instruct --output-dir output/package_instruct --environment production
  ```

## Features

- Dataset conversion for Traditional Chinese Code-QA
- Multi-agent architecture using LangChain and LangGraph
- Support for various LLM providers (OpenAI, Anthropic, Google)
- Real-time processing with immediate save after each record
- Automatic failure recovery and checkpoint system
- Environment-specific configurations (development/production)

## Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black src/
uv run isort src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
