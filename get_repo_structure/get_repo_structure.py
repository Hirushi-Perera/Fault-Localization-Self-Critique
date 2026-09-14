import argparse
import ast
import json
import os
import shutil
import subprocess
import uuid

import pandas as pd
from tqdm import tqdm

repo_to_top_folder = {
    "django/django": "django",
    "sphinx-doc/sphinx": "sphinx",
    "scikit-learn/scikit-learn": "scikit-learn",
    "sympy/sympy": "sympy",
    "pytest-dev/pytest": "pytest",
    "matplotlib/matplotlib": "matplotlib",
    "astropy/astropy": "astropy",
    "pydata/xarray": "xarray",
    "mwaskom/seaborn": "seaborn",
    "psf/requests": "requests",
    "pylint-dev/pylint": "pylint",
    "pallets/flask": "flask",
}


def checkout_commit(repo_path, commit_id):
    """Checkout the specified commit in the given local git repository.
    :param repo_path: Path to the local git repository
    :param commit_id: Commit ID to checkout
    :return: None
    """
    try:
        # Change directory to the provided repository path and checkout the specified commit
        print(f"Checking out commit {commit_id} in repository at {repo_path}...")
        subprocess.run(["git", "-C", repo_path, "checkout", commit_id], check=True)
        print("Commit checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def clone_repo(repo_name, repo_playground, commit_id=None):
    repo_path = f"{repo_playground}/{repo_to_top_folder[repo_name]}"
    url = f"https://github.com/{repo_name}.git"

    if commit_id is not None:
        # Shallow-fetch just the target commit instead of the full history.
        # Much smaller download (a few MB vs the repo's entire history), and
        # GitHub supports fetching an arbitrary commit SHA directly.
        try:
            print(
                f"Shallow-fetching commit {commit_id} of {repo_name} to {repo_path}..."
            )
            os.makedirs(repo_path, exist_ok=True)
            subprocess.run(["git", "-C", repo_path, "init", "-q"], check=True)
            subprocess.run(
                ["git", "-C", repo_path, "remote", "add", "origin", url], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    repo_path,
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    commit_id,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", repo_path, "checkout", "FETCH_HEAD"], check=True
            )
            print("Repository shallow-fetched successfully.")
            return
        except subprocess.CalledProcessError as e:
            print(f"Shallow fetch failed ({e}), falling back to full clone...")
            subprocess.run(["rm", "-rf", repo_path], check=True)

    try:
        print(f"Cloning repository from {url} to {repo_path}...")
        subprocess.run(["git", "clone", url, repo_path], check=True)
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running git command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_project_structure_from_scratch(
    repo_name, commit_id, instance_id, repo_playground
):

    # Generate a temperary folder and add uuid to avoid collision
    repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))

    # assert playground doesn't exist
    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    # create playground
    os.makedirs(repo_playground)

    clone_repo(repo_name, repo_playground, commit_id=commit_id)
    # clone_repo already checks out commit_id (shallow fetch -> FETCH_HEAD, or
    # falls back to a full clone here needing an explicit checkout)
    repo_path = f"{repo_playground}/{repo_to_top_folder[repo_name]}"
    current_head = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_head != commit_id:
        checkout_commit(repo_path, commit_id)
    structure = create_structure(f"{repo_playground}/{repo_to_top_folder[repo_name]}")
    # clean up (shutil.rmtree instead of `rm -rf` subprocess - `rm` isn't on PATH
    # in a plain Windows/PowerShell environment, only when Git Bash's usr/bin is)
    shutil.rmtree(
        f"{repo_playground}/{repo_to_top_folder[repo_name]}", ignore_errors=True
    )
    d = {
        "repo": repo_name,
        "base_commit": commit_id,
        "structure": structure,
        "instance_id": instance_id,
    }
    return d


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            # explicit utf-8 (with a lossy fallback) - Django's source has plenty
            # of non-ASCII bytes, and relying on the platform-default encoding
            # breaks on Windows (cp1252) even though it happens to work under
            # Git Bash / Linux / Mac (utf-8 default there)
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[
                                n.lineno - 1 : n.end_lineno
                            ],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[
                        node.lineno - 1 : node.end_lineno
                    ],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[
                            node.lineno - 1 : node.end_lineno
                        ],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure
