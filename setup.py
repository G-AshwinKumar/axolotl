"""setup.py for axolotl"""

import ast
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements(extras_require_map):
    _install_requires = []
    _dependency_links = []
    with open("./requirements.txt", encoding="utf-8") as requirements_file:
        lines = [r.strip() for r in requirements_file.readlines()]
        for line in lines:
            is_extras = "deepspeed" in line or "mamba-ssm" in line
            if line.startswith("--extra-index-url"):
                # Handle custom index URLs
                _, url = line.split()
                _dependency_links.append(url)
            elif not is_extras and line and line[0] != "#":
                # Handle standard packages
                _install_requires.append(line)
    try:
        xformers_version = [req for req in _install_requires if "xformers" in req][0]
        autoawq_version = [req for req in _install_requires if "autoawq" in req][0]
        if "Darwin" in platform.system():
            # skip packages not compatible with OSX
            skip_packages = [
                "bitsandbytes",
                "triton",
                "mamba-ssm",
                "xformers",
                "autoawq",
                "liger-kernel",
            ]
            _install_requires = [
                req
                for req in _install_requires
                if re.split(r"[>=<]", req)[0].strip() not in skip_packages
            ]
            print(
                _install_requires, [req in skip_packages for req in _install_requires]
            )
        else: # not Darwin
            # detect the version of torch already installed
            # and set it so dependencies don't clobber the torch version
            try:
                torch_version = version("torch")
                print(f"INFO: Detected torch version: {torch_version} for axolotl setup logic.")
            except PackageNotFoundError:
                # If torch is not found by version(), we still need a torch_version
                # for the subsequent xformers/vllm logic.
                # However, we will NOT add torch to _install_requires.
                # The environment (and its constraints.txt) MUST provide torch.
                # For the logic below, we'll use the version from your constraint file if possible,
                # or fall back to a sensible default if we can't infer it.
                # Since your constraint is 2.7.0a0+..., let's use that as a hint.
                print("WARNING: torch.version() could not find an installed PyTorch in the build env. "
                      "Axolotl will NOT add torch to its dependencies. "
                      "Ensure PyTorch is in your base environment. "
                      "Using a placeholder for dependent package logic (xformers, etc.).")
                # For the xformers/etc. logic, it needs a major/minor.
                # Your actual torch is 2.7.0a0...
                # So, for that logic, let's assume 2.7.
                torch_version = "2.7.0" # This helps the xformers/vllm logic below pick correctly.
                                        # If this default is wrong for some edge case, it may need adjustment.
                                        # The original default was "2.6.0". If "2.7.0" causes issues
                                        # with xformers selection, revert this specific line to "2.6.0"
                                        # but KEEP the _install_requires.append line commented out.

            # _install_requires.append(f"torch=={torch_version}") # <<< IMPORTANT: Keep this commented/deleted

            # The rest of the logic for xformers, autoawq, vllm will use the torch_version determined above.
            # This is fine, as it's about selecting *other* packages based on an *assumed* torch environment.
            version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)
            # ... (rest of the xformers/vllm/etc. logic remains unchanged)
            if version_match:
                major, minor, patch = version_match.groups()
                major, minor = int(major), int(minor)
                patch = (
                    int(patch) if patch is not None else 0
                )  # Default patch to 0 if not present
            else:
                raise ValueError("Invalid version format")

            if (major, minor) >= (2, 7):
                _install_requires.pop(_install_requires.index(xformers_version))
                # _install_requires.append("xformers==0.0.29.post3")  # xformers seems to be hard pinned to 2.6.0
                extras_require_map["vllm"] = ["vllm==0.8.5.post1"]
            elif (major, minor) >= (2, 6):
                _install_requires.pop(_install_requires.index(xformers_version))
                _install_requires.append(
                    "xformers==0.0.29.post2"
                )  # vllm needs post2 w torch 2.6
                extras_require_map["vllm"] = ["vllm==0.8.5.post1"]
            elif (major, minor) >= (2, 5):
                _install_requires.pop(_install_requires.index(xformers_version))
                if patch == 0:
                    _install_requires.append("xformers==0.0.28.post2")
                else:
                    _install_requires.append("xformers>=0.0.28.post3")
                _install_requires.pop(_install_requires.index(autoawq_version))
            elif (major, minor) >= (2, 4):
                if patch == 0:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers>=0.0.27")
                else:
                    _install_requires.pop(_install_requires.index(xformers_version))
                    _install_requires.append("xformers==0.0.28.post1")
            else:
                raise ValueError("axolotl requires torch>=2.4")

    except PackageNotFoundError:
        pass
    return _install_requires, _dependency_links, extras_require_map


def get_package_version():
    with open(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "src"
        / "axolotl"
        / "__init__.py",
        "r",
        encoding="utf-8",
    ) as fin:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", fin.read(), re.MULTILINE)
    version_ = ast.literal_eval(version_match.group(1))
    return version_


extras_require = {
    "flash-attn": ["flash-attn==2.7.4.post1"],
    "ring-flash-attn": [
        "flash-attn==2.7.4.post1",
        "ring-flash-attn>=0.1.4",
        "yunchang==0.6.0",
    ],
    "deepspeed": [
        "deepspeed==0.15.4",
        "deepspeed-kernels",
    ],
    "mamba-ssm": [
        "mamba-ssm==1.2.0.post1",
        "causal_conv1d",
    ],
    "auto-gptq": [
        "auto-gptq==0.5.1",
    ],
    "mlflow": [
        "mlflow",
    ],
    "galore": [
        "galore_torch",
    ],
    "apollo": [
        "apollo-torch",
    ],
    "optimizers": [
        "galore_torch",
        "apollo-torch",
        "lomo-optim==0.1.1",
        "torch-optimi==0.2.1",
        "came_pytorch==0.1.3",
    ],
    "ray": [
        "ray[train]",
    ],
    "vllm": [
        "vllm==0.7.2",
    ],
    "llmcompressor": [
        "llmcompressor==0.5.1",
    ],
}

install_requires, dependency_links, extras_require_build = parse_requirements(
    extras_require
)

setup(
    version=get_package_version(),
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=install_requires,
    dependency_links=dependency_links,
    entry_points={
        "console_scripts": [
            "axolotl=axolotl.cli.main:main",
        ],
    },
    extras_require=extras_require_build,
)
