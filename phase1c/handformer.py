"""Thin, hash-guarded loader for official HF_Pose with compatibility-only patches.

The checkout remains outside Git; no upstream architecture source is vendored.
All classes and forward methods execute from the pinned upstream source AST.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import types

from e0.data import sha256
from .handformer_data import HandFormerDataset


REVISION = "640a10a6bd03f85842a2662e67cf30444ab1d673"
NAMESPACE = "_jpa_pinned_handformer"
PROVENANCE = Path(__file__).resolve().parents[1] / "reproducibility/upstreams/handformer.json"


class Compatibility(ast.NodeTransformer):
    """Exact edits for this revision; source hash guards reject unknown code."""
    def visit_ImportFrom(self, node):
        if node.module in ("thop", "fvcore.nn.flop_count", "tqdm", "einops"):
            return None
        if node.module == "utils" or (node.module or "").startswith("models."):
            node.module = NAMESPACE + "." + node.module
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get_device"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "batch_seq_ain_pe"
                and not node.args and not node.keywords):
            return ast.copy_location(ast.Attribute(value=node.func.value, attr="device", ctx=ast.Load()), node)
        if isinstance(node.func, ast.Name) and node.func.id == "repeat":
            if (len(node.args) != 2 or not isinstance(node.args[1], ast.Constant)
                    or node.args[1].value != "() n d -> b n d" or len(node.keywords) != 1
                    or node.keywords[0].arg != "b"
                    or ast.unparse(node.args[0]) != "self.action_token"):
                raise ValueError("Unknown repeat usage in pinned HandFormer")
            return ast.copy_location(ast.Call(func=ast.Attribute(value=node.args[0], attr="expand", ctx=ast.Load()),
                args=[node.keywords[0].value, ast.Constant(-1), ast.Constant(-1)], keywords=[]), node)
        return node


def compatible_tree(source, filename):
    tree = ast.parse(source, filename=filename)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.If) and ast.unparse(node.test) in ("__name__ == '__main__'", '"__main__" == __name__'))
        and not (isinstance(node, ast.Expr) and ast.unparse(node.value) == "sys.path.insert(0, '')")]
    return ast.fix_missing_locations(Compatibility().visit(tree))


def source_root(config):
    value = config.get("upstream_path") or os.environ.get("JPA_HANDFORMER_ROOT")
    if not value and os.environ.get("JPA_ROOT"):
        value = str(Path(os.environ["JPA_ROOT"]) / "upstreams/HandFormer")
    if not value:
        raise ValueError("Set JPA_HANDFORMER_ROOT to the pinned official checkout")
    return Path(value).resolve()


def verify_upstream(root):
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
        "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("HandFormer upstream revision differs from the pinned source")
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for path, expected in manifest["source_sha256"].items():
        if sha256(root / path) != expected:
            raise ValueError(f"HandFormer source hash mismatch: {path}")
    return manifest


def official_class(root):
    root = Path(root).resolve()
    manifest = verify_upstream(root)
    if NAMESPACE + ".models.hf_pose" in sys.modules:
        return sys.modules[NAMESPACE + ".models.hf_pose"].HF_Pose
    for name in (NAMESPACE, NAMESPACE + ".models"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    # The pinned provenance preserves dependency order; no package __init__ is
    # executed because upstream's initializer also imports the RGB-only model.
    try:
        for path in manifest["source_sha256"]:
            relative = Path(path).relative_to("HandFormer").with_suffix("")
            name = NAMESPACE + "." + ".".join(relative.parts)
            module = types.ModuleType(name)
            module.__file__ = str(root / path)
            module.__package__ = name.rpartition(".")[0]
            sys.modules[name] = module
            code = compatible_tree((root / path).read_text(encoding="utf-8"), str(root / path))
            exec(compile(code, str(root / path), "exec"), module.__dict__)
    except Exception:
        for name in list(sys.modules):
            if name == NAMESPACE or name.startswith(NAMESPACE + "."):
                del sys.modules[name]
        raise
    return sys.modules[NAMESPACE + ".models.hf_pose"].HF_Pose


def build_model(config, artifact):
    cls = official_class(source_root(config))
    arguments = dict(config["model"])
    if "num_classes" in arguments and arguments["num_classes"] != len(artifact["classes"]):
        raise ValueError("Configured HandFormer head differs from frozen vocabulary")
    arguments["num_classes"] = len(artifact["classes"])
    return cls(**arguments)


def build_dataset(dataset_path, artifact, split, config, compatibility=None):
    compatibility = compatibility or config.get("compatibility_path")
    if not compatibility:
        raise ValueError("HandFormer requires its validated compact source cache manifest")
    return HandFormerDataset(dataset_path, artifact, split, config, compatibility)
