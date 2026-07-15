import unittest
from unittest import mock

import okf_zvec


def result(identifier, score, **fields):
    return {
        "id": identifier,
        "score": score,
        "fields": {
            "type": "",
            "tags": [],
            "project": "",
            "path": "",
            "timestamp": "",
            "title": identifier,
            "heading": "",
            "text": "",
            **fields,
        },
    }


class SearchQualityTests(unittest.TestCase):
    def test_semantic_weight_changes_rrf_order(self):
        first = result("first", 0.1)
        second = result("second", 0.2)

        semantic_first = okf_zvec.weighted_rrf(
            [first, second],
            [second, first],
            okf_zvec.SearchOptions(semantic_weight=3, fts_weight=1),
        )
        fts_first = okf_zvec.weighted_rrf(
            [first, second],
            [second, first],
            okf_zvec.SearchOptions(semantic_weight=1, fts_weight=3),
        )

        self.assertEqual(okf_zvec.doc_id(semantic_first[0]["result"]), "first")
        self.assertEqual(okf_zvec.doc_id(fts_first[0]["result"]), "second")

    def test_metadata_filters_are_combined(self):
        fields = {
            "type": "software-project",
            "tags": ["zvec", "okf"],
            "project": "search",
            "path": "topics/zvec.md",
            "timestamp": "2026-07-09T10:00:00+05:00",
        }
        options = okf_zvec.SearchOptions(
            doc_type="software-project",
            tags=("zvec",),
            project="search",
            path_pattern="topics/*",
            date_from="2026-07-01",
            date_to="2026-07-31",
        )

        self.assertTrue(okf_zvec.result_matches_filters(fields, options))
        self.assertFalse(
            okf_zvec.result_matches_filters(
                fields,
                okf_zvec.SearchOptions(tags=("missing",)),
            )
        )

        expression = okf_zvec.build_zvec_filter(options)
        self.assertIn('filter_type = "software-project"', expression)
        self.assertIn('filter_tags CONTAIN_ALL ("zvec")', expression)
        self.assertIn('filter_path LIKE "topics/%"', expression)

        exact_path = okf_zvec.build_zvec_filter(
            okf_zvec.SearchOptions(path_pattern="topics/my_note.md")
        )
        self.assertEqual(exact_path, 'filter_path = "topics/my_note.md"')

    def test_hybrid_relevance_uses_only_available_signals(self):
        options = okf_zvec.SearchOptions(semantic_weight=0.7, fts_weight=0.3)
        item = {"semantic_score": None, "fts_score": 3.0}
        self.assertAlmostEqual(okf_zvec.hybrid_relevance(item, options), 0.75)

    def test_relevance_is_normalized(self):
        self.assertAlmostEqual(okf_zvec.semantic_relevance(0.2), 0.8)
        self.assertAlmostEqual(okf_zvec.fts_relevance(3.0), 0.75)
        self.assertEqual(okf_zvec.semantic_relevance(2.0), 0.0)

    def test_matching_terms_preserves_document_word_form(self):
        fields = {
            "title": "Поставщики",
            "heading": "",
            "path": "topics/suppliers.md",
            "text": "Работа с поставщиками",
        }
        terms = okf_zvec.matching_terms("поставщик", fields)
        self.assertIn("Поставщики", terms)
        self.assertIn("поставщиками", terms)

    def test_benchmark_rank_checks_result_content(self):
        results = [
            {"rank": 1, "title": "Другое", "heading": "", "path": "", "text": ""},
            {"rank": 2, "title": "POS Center", "heading": "", "path": "", "text": ""},
        ]
        self.assertEqual(okf_zvec.benchmark_rank(results, "POS Center"), 2)
        self.assertEqual(okf_zvec.benchmark_rank(results, "Отсутствует"), 0)

    def test_benchmark_metrics_support_multiple_relevant_results(self):
        recall, ndcg = okf_zvec.benchmark_ranking_metrics([1, 3, 0], topk=3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertGreater(ndcg, 0.6)
        self.assertLess(ndcg, 1.0)

    def test_relevance_threshold_removes_weak_results(self):
        class FakeCollection:
            def query(self, _query, topk, filter=None):
                self.filter = filter
                return [
                    result("strong", 0.1),
                    result("weak", 0.8),
                ][:topk]

        okf_zvec._QUERY_CACHE.clear()
        with (
            mock.patch.object(okf_zvec, "get_model"),
            mock.patch.object(okf_zvec, "embed", return_value=[0.0] * okf_zvec.DIMENSION),
        ):
            results = okf_zvec.search_collection(
                FakeCollection(),
                "проверка",
                topk=5,
                rerank_pool=5,
                search_mode="semantic",
                options=okf_zvec.SearchOptions(min_relevance=0.5),
            )

        self.assertEqual([item["id"] for item in results], ["strong"])
        self.assertAlmostEqual(results[0]["relevance"], 0.9)


if __name__ == "__main__":
    unittest.main()
