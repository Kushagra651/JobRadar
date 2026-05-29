from pathlib import Path

structure = {
    "config": ["settings.py"],

    "agent": [
        "graph.py",
        "state.py",
    ],

    "agent/nodes": [
        "fetch_node.py",
        "extract_node.py",
        "retrieve_node.py",
        "analyze_node.py",
    ],

    "ml": [
        "train.py",
        "evaluate.py",
        "explain.py",
        "predict.py",
    ],

    "rag": [
        "embedder.py",
        "indexer.py",
        "retriever.py",
    ],

    "pipeline": [
        "ingest.py",
        "build_features.py",
    ],

    "mcp": [
        "server.py",
    ],

    "mcp/tools": [
        "fetch_tool.py",
        "analyze_tool.py",
        "explain_tool.py",
    ],

    "api": [
        "main.py",
        "schemas.py",
    ],

    "api/routers": [
        "analyze.py",
        "health.py",
    ],

    "dashboard": [
        "app.py",
    ],

    "tests": [
        "test_nodes.py",
        "test_api.py",
    ],

    "notebooks": [
        "01_eda.ipynb",
        "02_ml_baseline.ipynb",
    ],
}

root_files = [
    "README.md",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
    "requirements.txt",
]

# Create folders and files
for folder, files in structure.items():
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    # create __init__.py
    init_file = folder_path / "__init__.py"
    init_file.touch(exist_ok=True)

    for file in files:
        file_path = folder_path / file
        file_path.touch(exist_ok=True)

# Create root files
for file in root_files:
    Path(file).touch(exist_ok=True)

print("✅ JobRadar project structure created successfully!")