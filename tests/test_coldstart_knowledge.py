import json
from types import SimpleNamespace

from engine.coldstart.knowledge import build_guidance_description


def test_dinov3_guidance_requires_local_code_and_weights(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    models_path = tmp_path / "models.json"
    torch_hub_dir = tmp_path / "torch"
    tasks_path.write_text(json.dumps({"vision-task": "General Image"}))
    models_path.write_text(json.dumps({
        "General Image": {
            "DINOv3": {
                "Description": "local DINO",
                "Code_template": "load {TORCH_HUB_DIR}/dinov3-main",
            },
            "Siglip2": {
                "Description": "downloadable fallback",
                "Code_template": "load siglip",
            },
        }
    }))
    cfg = SimpleNamespace(
        exp_id="vision-task",
        torch_hub_dir=str(torch_hub_dir),
        coldstart=SimpleNamespace(
            task_json_path=str(tasks_path),
            model_json_path=str(models_path),
        ),
    )

    guidance = build_guidance_description(cfg)
    assert "DINOv3" not in guidance
    assert "Siglip2" in guidance

    repo_file = torch_hub_dir / "dinov3-main" / "hubconf.py"
    weights_file = (
        torch_hub_dir
        / "checkpoints"
        / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    )
    repo_file.parent.mkdir(parents=True)
    weights_file.parent.mkdir(parents=True)
    repo_file.write_text("# admitted local repository\n")
    weights_file.write_bytes(b"admitted local weights")

    guidance = build_guidance_description(cfg)
    assert "DINOv3" in guidance
    assert str(torch_hub_dir / "dinov3-main") in guidance
    assert "Siglip2" in guidance
