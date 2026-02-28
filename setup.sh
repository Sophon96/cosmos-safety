# Run the following command to initialize and update submodules recursively
git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync

git config --global alias.a add
git config --global alias.cm commit -m
git config --global alias.s status
 