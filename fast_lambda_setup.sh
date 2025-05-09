sudo apt-get update -y
sudo apt-get install tmux -y
sudo apt-get install python3-dev -y

sudo apt-get install tmux libaio-dev libopenmpi-dev python3-mpi4py -y

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv .env --python 3.10

source .env/bin/activate

uv pip install wheel packaging

uv pip install -r train_requirements.txt
uv pip install flash-attn==2.5.9.post1 --no-build-isolation
