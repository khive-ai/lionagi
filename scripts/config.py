prefix = "full"
postfix = "validation_codes_tests"
crates = ["validation"]

compress_prefix = ""
compress = False
compress_cumulative = False
compression_iterations = 0

config = {
    "dir": "/Users/lion/google_drive/khivecode/libs/lionagi/tests/unit",
    "output_dir": "/Users/lion/lionagi/dev/data/khivecode",
    "prefix": prefix,
    "postfix": postfix,
    "crates": crates,
    "exclude_patterns": [".venv", "__pycache__"],
    "file_types": [".py"],
}
