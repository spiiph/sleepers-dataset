# Sleeper dataset utilities

**Table of Contents**
- [Installation](#installation)
- [License](#license)

## Installation
To install the package in development mode, there are two options.

### With `hatch`
Installation with `hatch` is the preferred method. First install `hatch` with
`pipx`:

```
python -m pip install --user pipx
pipx ensurepath
pipx install hatch
```

Then clone the repository, create an environment, and enter it
```
git clone https://github.com/spiiph/sleeper-dataset
cd sleeper-dataset
hatch env create
hatch shell
```

### With `pip`
Installation with `pip` in a virtual environment is also possible. First
create the virtual environment:

```
git clone https://github.com/spiiph/sleeper-dataset
cd sleeper-dataset
python -m venv create .env
source .env/bin/activate
```

Then upgrade `pip` and install requirements:
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[tests]"
```

## Console scripts
Some console scripts are provided with the `sleepers-dataset` package.

### Hello World
This script prints "Hello World!" and exits.
```
hello-world
```

### Create patches
This script splits an image and its corresponding binary mask into eight
equally sized patches.
```
create-patches <image-path> <mask-path> <output-path>
```

## License

`sleeper-dataset` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
