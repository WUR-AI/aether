import json
import os

from src.data.base_caption_builder import DummyCaptionBuilder
from src.data.base_datamodule import BaseDataModule
from src.data.butterfly_caption_builder import ButterflyCaptionBuilder
from src.data.butterfly_dataset import ButterflyDataset


def test_datamodule_uses_collate_when_aux_data(request, sample_csv, tmp_path):
    use_mock = request.config.getoption("--use-mock")
    templates_path = tmp_path / "location_caption_templates" / "v1.json"
    os.makedirs(str(tmp_path / "location_caption_templates"), exist_ok=True)
    print(f"Mock captions written to {templates_path}")
    templates_path.write_text(json.dumps(["<name_loc> text"]))

    concepts_path = tmp_path / "concept_captions" / "v1.json"
    os.makedirs(str(tmp_path / "concept_captions"), exist_ok=True)
    print(f"Concept captions written to {concepts_path}")
    concepts_path.write_text(
        json.dumps(
            """[{
            "concept_caption": "Forested area",
            "is_max": true,
            "theta_k": 0.5,
            "col": "aux_corine_frac_311"
          }]"""
        )
    )

    caption_builder = DummyCaptionBuilder("v1.json", "v1.json", data_dir=str(tmp_path), seed=0)

    dataset = ButterflyDataset(
        data_dir=sample_csv,
        cache_dir=str(tmp_path),
        modalities={"coords": None},
        use_target_data=True,
        use_aux_data="all",
        seed=0,
        mock=use_mock,
    )

    dm = BaseDataModule(
        dataset,
        batch_size=2,
        train_val_test_split=(4, 1, 1),
        split_mode="random",
        caption_builder=caption_builder,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "text" in batch
    assert len(batch["text"]) == dm.batch_size_per_device, batch


def test_captionbuilder_generic_properties(tmp_path):
    """This test checks that all caption builders implement the basic properties and methods."""
    dict_caption_builders = {"butterfly": ButterflyCaptionBuilder, "dummy": DummyCaptionBuilder}

    templates_fname = "v1.json"
    concepts_fname = "v1.json"

    for name_cb, cb_class in dict_caption_builders.items():
        # There is no data on git anymore
        # if name_cb == "butterfly":
        #     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        #     templates_path = os.path.join(repo_root, "data", "s2bms")
        # else:
        templates_path = tmp_path / "location_caption_templates" / templates_fname
        os.makedirs(str(tmp_path / "location_caption_templates"), exist_ok=True)
        print(f"Mock captions written to {templates_path}")
        templates_path.write_text(json.dumps(["<name_loc> text"]))

        concepts_path = tmp_path / "concept_captions" / concepts_fname
        os.makedirs(str(tmp_path / "concept_captions"), exist_ok=True)
        print(f"Concept captions written to {concepts_path}")
        concepts_path.write_text(
            json.dumps(
                """[{
                "concept_caption": "Forested area",
                "is_max": true,
                "theta_k": 0.5,
                "col": "aux_corine_frac_311"
              }]"""
            )
        )

        caption_builder = cb_class(
            templates_fname=templates_fname,
            concepts_fname=concepts_fname,
            data_dir=tmp_path,
            seed=0,
        )

        assert hasattr(
            caption_builder, "templates"
        ), f"'templates' attribute missing in {cb_class.__name__}."
        assert hasattr(
            caption_builder, "concepts"
        ), f"'concepts' attribute missing in {cb_class.__name__}."
        assert hasattr(
            caption_builder, "data_dir"
        ), f"'data_dir' attribute missing in {cb_class.__name__}."
        assert hasattr(
            caption_builder, "seed"
        ), f"'seed' attribute missing in {cb_class.__name__}."
        assert hasattr(
            caption_builder, "column_to_metadata_map"
        ), f"'column_to_metadata_map' attribute missing in {cb_class.__name__}."
        assert hasattr(
            caption_builder, "sync_with_dataset"
        ), f"'sync_with_dataset' method missing in {cb_class.__name__}."
        assert callable(
            getattr(caption_builder, "sync_with_dataset")
        ), f"'sync_with_dataset' is not callable in {cb_class.__name__}."
