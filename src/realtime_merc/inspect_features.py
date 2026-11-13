import argparse

from .features import MMAlignFeatureStore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect MM-Align MELD feature shapes."
    )
    parser.add_argument(
        "--features-dir",
        required=True,
        help="Directory containing MM-Align .pkl feature files (or a .pkl path).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split name if the .pkl contains multiple splits (train/dev/val/test).",
    )
    parser.add_argument(
        "--pool",
        choices=["mean", "flatten"],
        default="mean",
        help="Pooling for MM-Align features when they are not 1D.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=3,
        help="Number of sample entries to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    store = MMAlignFeatureStore(
        args.features_dir, split=args.split, pool=args.pool
    )
    print("MM-Align pkl: {}".format(store.pkl_path))
    print("Split: {}".format(args.split or "auto"))
    print("Total entries: {}".format(len(store.feature_map)))
    print("Feature dim: {}".format(store.feature_dim))
    if store.text_embedding is not None:
        print("Text embedding: {} {}".format(
            store.text_embedding_path, tuple(store.text_embedding.shape)
        ))
    if store.example_modalities:
        for name, shape in store.example_modalities.items():
            print("  modality {} shape: {}".format(name, shape))
    else:
        print("  modality shapes: unavailable (fused or unknown format)")

    printed = 0
    for (dialogue_id, utterance_id), feat in store.feature_map.items():
        print(
            "  sample dia={} utt={} fused_shape={}".format(
                dialogue_id, utterance_id, tuple(feat.shape)
            )
        )
        printed += 1
        if printed >= args.max_items:
            break


if __name__ == "__main__":
    main()
