# The documentation site

Built with [Zensical](https://zensical.org) from `zensical.toml` (repository
root) and this directory, with the same setup as
[mlx-graphs](https://github.com/mlx-graphs/mlx-graphs/tree/main/docs).
Published at <https://mlx-graphs.github.io/mlx_cluster/>.

## Preview locally

The API page reads the docstrings of the C++ extension from the installed
package, so build mlx_cluster first (a CPU-only build is enough):

```shell
pip install -e .
pip install -e '.[docs]'
zensical serve                     # http://localhost:8000/mlx_cluster/, live reload
```

CI runs `zensical build --strict --clean` on pull requests that touch the docs
or the API, and deploys the site on each release
(`.github/workflows/publish_docs.yml`).

## Where things are

| Path | What |
|---|---|
| `content/*.md` | the pages: home, installation, quickstart, the SpMM guide and the API reference |
| `content/api.md` | the API page: one `::: mlx_cluster.<name>` block per public function or class, rendered from its docstring (Google style) |
| `content/*.html` | redirects from the URLs of the old Sphinx site |
| `content/assets/` | CSS, JavaScript and icons (shared with mlx-graphs) |
| `overrides/` | theme overrides: extra icons |
| `scripts/griffe_rst.py` | griffe extension: the reST of the docstrings to Markdown, and the signatures of the nanobind functions |

A new public function or class goes in a `:::` block of `content/api.md`.
