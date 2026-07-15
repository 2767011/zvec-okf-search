import gc
import tempfile
import unittest
from pathlib import Path

import zvec


class ZvecIntegrationTests(unittest.TestCase):
    def test_vector_order_and_native_metadata_filter(self):
        temporary = tempfile.TemporaryDirectory()
        collection_path = Path(temporary.name) / "collection"
        schema = zvec.CollectionSchema(
            name="integration",
            fields=[
                zvec.FieldSchema(
                    "category",
                    zvec.DataType.STRING,
                    nullable=False,
                    index_param=zvec.InvertIndexParam(),
                ),
                zvec.FieldSchema(
                    "tags",
                    zvec.DataType.ARRAY_STRING,
                    nullable=False,
                    index_param=zvec.InvertIndexParam(),
                ),
                zvec.FieldSchema(
                    "path",
                    zvec.DataType.STRING,
                    nullable=False,
                    index_param=zvec.InvertIndexParam(enable_extended_wildcard=True),
                ),
                zvec.FieldSchema(
                    "timestamp",
                    zvec.DataType.STRING,
                    nullable=False,
                    index_param=zvec.InvertIndexParam(enable_range_optimization=True),
                ),
            ],
            vectors=zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                2,
                index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE),
            ),
        )
        collection = zvec.create_and_open(str(collection_path), schema)
        collection.insert(
            [
                zvec.Doc(
                    "same",
                    vectors={"embedding": [1.0, 0.0]},
                    fields={"category": "a", "tags": ["red"], "path": "other.md", "timestamp": ""},
                ),
                zvec.Doc(
                    "side",
                    vectors={"embedding": [0.0, 1.0]},
                    fields={
                        "category": "b",
                        "tags": ["blue", "green"],
                        "path": "topics/side.md",
                        "timestamp": "2026-07-15T10:00:00+05:00",
                    },
                ),
                zvec.Doc(
                    "opposite",
                    vectors={"embedding": [-1.0, 0.0]},
                    fields={"category": "a", "tags": ["blue"], "path": "topics/old.md", "timestamp": "2025-01-01"},
                ),
            ]
        )
        collection.flush()

        query = zvec.Query("embedding", vector=[1.0, 0.0])
        results = collection.query(query, topk=3)
        self.assertEqual(results[0].id, "same")
        self.assertLess(results[0].score, results[1].score)

        filtered = collection.query(
            query,
            topk=3,
            filter=(
                'category = "b" AND tags CONTAIN_ALL ("blue") '
                'AND path LIKE "topics/%" AND timestamp >= "2026-07-01"'
            ),
        )
        self.assertEqual([item.id for item in filtered], ["side"])

        del filtered, results, collection
        gc.collect()
        temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
