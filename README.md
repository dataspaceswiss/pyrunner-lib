# Pyrunner for DataSpace


### Python Usage

```python
from pyrunner_lib import transform

# Run a transformation
transform("my_transform_id", "/path/to/base")
```

### CLI Usage

```bash
python __main__.py <transform_id> <base_path>
```

## DataSpace Pipelines Skill

This repository includes the [`dataspace-pipelines`](https://github.com/dataspaceswiss/pyrunner-lib/tree/main/skills/dataspace-pipelines) skill for Codex and Claude Code.

### Codex

```text
$skill-installer Install dataspace-pipelines from
https://github.com/dataspaceswiss/pyrunner-lib/tree/main/skills/dataspace-pipelines
```

Invoke it with `$dataspace-pipelines`.

### Claude Code

```bash
npx skills add https://github.com/dataspaceswiss/pyrunner-lib/tree/main/skills/dataspace-pipelines \
  --agent claude-code --global
```

Invoke it with `/dataspace-pipelines`.
