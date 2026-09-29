def prompt_sft(desc: str, var_name="model") -> str:
    return f"""Generate CadQuery Python code based on the following description: {desc}

You must define the final model as the variable `{var_name}`, and return the Python code only in the following format:
```python
import cq_monkey_patch
import cadquery as cq

# Your code here, final result assigned to `{var_name}`
```""".strip()
