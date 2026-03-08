# OpenModel Registry

The official public model registry for [OpenModelStudio](https://openmodel.studio).

Browse, discover, and install community-contributed models into any OpenModelStudio instance.

## How It Works

Each model lives in its own folder under `models/`. A GitHub Actions workflow aggregates all model metadata into `registry/index.json` on every push, providing a single endpoint for fast lookups.

## Installing Models

```bash
# Install a model from the registry
openmodelstudio install titanic-rf

# Search for models
openmodelstudio search "image classification"

# List installed models
openmodelstudio list
```

Or from a notebook / Python script:

```python
import openmodelstudio as oms

# Install from registry
oms.registry_install("titanic-rf")

# Search the registry
results = oms.registry_search("classification")

# Load an installed model and train it
model = oms.use_model("titanic-rf")
handle = oms.register_model("my-titanic", model=model)
job = oms.start_training(handle.model_id, wait=True)
```

## Submitting a Model

1. Fork this repository
2. Create a folder under `models/` with your model name (e.g., `models/my-model/`)
3. Add a `model.json` with metadata (see schema below)
4. Add your model code as `model.py` with `train(ctx)` and `infer(ctx)` functions
5. Open a Pull Request

### model.json Schema

```json
{
  "name": "my-model",
  "version": "1.0.0",
  "description": "Short description of what the model does",
  "author": "your-github-username",
  "framework": "pytorch",
  "category": "classification",
  "tags": ["tabular", "binary-classification"],
  "license": "MIT",
  "files": ["model.py"],
  "dependencies": ["scikit-learn>=1.3"],
  "homepage": "https://github.com/you/my-model"
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `name` | Unique model identifier (lowercase, hyphens) |
| `version` | Semantic version string |
| `description` | What the model does |
| `author` | GitHub username or org |
| `framework` | One of: `pytorch`, `sklearn`, `tensorflow`, `jax`, `python`, `rust` |
| `category` | One of: `classification`, `regression`, `nlp`, `computer-vision`, `generative`, `time-series`, `clustering`, `anomaly-detection`, `reinforcement-learning`, `multimodal` |
| `tags` | Array of searchable tags |
| `license` | SPDX license identifier |
| `files` | Array of files to include (relative to model folder) |

## Custom Registries

You can point your OpenModelStudio instance at a different registry:

```python
import openmodelstudio as oms
oms.set_registry("https://raw.githubusercontent.com/your-org/your-registry/main/registry/index.json")
```

Or set the environment variable:

```bash
export OPENMODELSTUDIO_REGISTRY_URL="https://raw.githubusercontent.com/your-org/your-registry/main/registry/index.json"
```

## License

This registry and all contributed models are subject to their individual licenses. The registry infrastructure itself is licensed under GPL-3.0.
