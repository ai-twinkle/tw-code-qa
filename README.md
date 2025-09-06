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

## Usage

Run the main script with the dataset path:

```bash
uv run python main.py --dataset-path data/opencoder_dataset_educational_instruct --dataset-type opencoder
```

Other examples:

- Process OpenCoder educational instruct dataset:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_educational_instruct --dataset-type opencoder
  ```

- Specify output directory and batch size:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_evol_instruct --output-dir output/evol --batch-size 50
  ```

- Test mode (process only first 10 records):
  ```bash
  uv run python main.py --dataset-path data/sample --max-records 10 --environment development
  ```

- Production mode:
  ```bash
  uv run python main.py --dataset-path data/opencoder_dataset_package_instruct --environment production --batch-size 200
  ```

## Features

- Dataset conversion for Traditional Chinese Code-QA
- Multi-agent architecture using LangChain and LangGraph
- Support for various LLM providers (OpenAI, Anthropic, Google)
- Batch processing capabilities
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
