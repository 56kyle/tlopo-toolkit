# tlopo-toolkit

tlopo_toolkit

---

**[📚 View Documentation](https://tlopo-toolkit.readthedocs.io/)** | **[🐛 Report a Bug](https://github.com/56kyle/tlopo-toolkit/issues)** | **[✨ Request a Feature](https://github.com/56kyle/tlopo-toolkit/issues)**

---

# WIP

This is a very, very casual project and by no means should anything of value depend on it for the time being.

Stuff will change, break, and rebreak routinely, although if it does become stable I will indicate it as such.
As a side note, please don't use this to abuse TLOPO. I love the game and have used this as a way to reexperience parts
of it. If you have found yourself wanting to limit test things, I invite you to do so, but please do so with discretion
so that you don't affect others while doing so.

## Installation

You can install `tlopo_toolkit` via [pip](pip-documentation) from PyPI:

```bash
pip install tlopo_toolkit
```

### Installation for Development

To set up `tlopo_toolkit` for local development:

1.  Clone the repository:
    ```bash
    git clone https://github.com/56kyle/tlopo-toolkit.git
    cd tlopo-toolkit
    ```
2.  Install dependencies using [:term:`uv`](uv-documentation):
    ```bash
    uv sync
    ```
3.  Install pre-commit hooks:
    ```bash
    uvx nox -s pre-commit -- install
    ```

This sets up a virtual environment and installs core, development, and quality check dependencies.

## Usage

(This section should explain how to use the generated application. Replace the content below with instructions specific to your project's functionality. If your project is a library, show import examples. If it's a CLI application, show command examples. Link to the full documentation for details.)

### As a Library

```python
# Example usage of your package as a library
# from tlopo_toolkit import some_function
# result = some_function()
# print(result)
```

### As a Command-Line Application

If your project defines command-line entry points in `pyproject.toml`:

```bash
# Example usage of your CLI application
# tlopo-toolkit --help
# tlopo-toolkit do-something --input file.txt
```

For detailed API documentation and CLI command references, see the **[Documentation][documentation]**.

## Development Workflow

This project uses a robust set of tools for development, testing, and quality assurance. All significant automated tasks are run via [:term:`Nox`](nox-documentation), orchestrated by the central `noxfile.py`.

- **Run all checks (lint, typecheck, security):** `uvx nox -s check`
- **Run test suite with coverage:** `uvx nox -s test`
- **Build documentation:** `uvx nox -s docs`
- **Build package:** `uvx nox -s build`
- **See all available tasks:** `uvx nox -l`

Explore the `noxfile.py` and the project documentation for detailed information on the automated workflow.

## Contributing

(This section should guide contributions _to this specific generated project_, not the template. It should refer to the project's `CODE_OF_CONDUCT.md` and link to a `CONTRIBUTING.md` specific to the project, if you choose to generate one.)

Report bugs or suggest features via the [issue tracker](https://github.com/56kyle/tlopo-toolkit/issues).

See [CONTRIBUTING.md](#) for contribution guidelines.

## License

Distributed under the terms of the **MIT** license. See [LICENSE](LICENSE) for details.

---

**This project was generated from the [cookiecutter-robust-python template][cookiecutter-robust-python].**

<!-- Reference Links -->

[cookiecutter-robust-python]: https://github.com/56kyle/cookiecutter-robust-python
[documentation]: https://tlopo-toolkit.readthedocs.io/
