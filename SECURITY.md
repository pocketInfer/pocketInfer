# Security

PocketInfer reads local JSON and writes generated JSON. It does not execute
model repository code, download checkpoints, or evaluate configuration values.

Do not add `trust_remote_code` or implicit network fetches to the core compiler.
Report security issues privately to the maintainers once a public repository
defines its security contact.
