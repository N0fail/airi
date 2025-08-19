import os
import yaml
import pathlib
from huggingface_hub import hf_hub_download

CONFIG_FILE = "models.yaml"
BASE_DIR = pathlib.Path(".")


def download_and_link(task: str, model_path: str):
    """
    Скачивает модель с Hugging Face и создаёт ссылку current -> файл
    """
    parts = model_path.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Неверный формат пути '{model_path}'. Ожидался вид user/repo/path/to/file"
        )

    repo_id = "/".join(parts[:2])  # user/repo
    filename = "/".join(parts[2:])  # path/to/file внутри репо

    target_dir = BASE_DIR / task
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{task}] Скачиваю {model_path} ...")
    downloaded_file = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=target_dir
    )

    # Создаем (или обновляем) ссылку current как относительную ссылку
    symlink_path = target_dir / "current"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    target_relpath = os.path.relpath(downloaded_file, start=target_dir)
    symlink_path.symlink_to(target_relpath)

    print(f"[{task}] Готово. Ссылка: {symlink_path} -> {target_relpath}")


def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"Файл {CONFIG_FILE} не найден.")
        return

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for task, model in config.items():
        download_and_link(task, model)


if __name__ == "__main__":
    main()
