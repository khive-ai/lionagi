prefix = "full"
postfix = "source_codes"
crates = ["lionagi"]

compress_prefix = ""
compress = False
compress_cumulative = False
compression_iterations = 0

config = {
    "dir": "/Users/lion/lionagi/",
    "output_dir": "/Users/lion/lionagi/dev/data/lionagi",
    "prefix": prefix,
    "postfix": postfix,
    "crates": crates,
    "exclude_patterns": [
        ".venv",
        "dist",
        "node_modules",
        "target",
        "package-lock",
        "lock",
        "__pycache__",
        ".git",
        ".idea",
        ".vscode",
        ".DS_Store",
        ".github",
        ".gitignore",
    ],
    "file_types": [
        ".tsx",
        ".ts",
        ".py",
        ".md",
        ".js",
        ".jsx",
        ".rs",
        ".toml",
        ".yaml",
        ".json",
    ],
}
