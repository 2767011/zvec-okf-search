import unittest

import okf_zvec


class TextProcessingTests(unittest.TestCase):
    def test_frontmatter_is_removed(self):
        metadata, body = okf_zvec.split_frontmatter(
            '---\ntype: service\ntitle: Проверка\n---\n\n# Содержимое\n'
        )
        self.assertEqual(metadata["type"], "service")
        self.assertEqual(metadata["title"], "Проверка")
        self.assertIn("# Содержимое", body)

    def test_markdown_is_chunked_by_list_item(self):
        chunks = okf_zvec.section_chunks("# Работа\n\n- Первый\n- Второй")
        self.assertEqual(len(chunks), 2)
        self.assertIn("Первый", chunks[0][1])
        self.assertIn("Второй", chunks[1][1])

    def test_russian_forms_share_a_lemma(self):
        self.assertEqual(okf_zvec.token_lemma("поставщиками"), "поставщик")
        self.assertEqual(okf_zvec.token_lemma("поставщиков"), "поставщик")

    def test_fts_text_is_whitespace_safe(self):
        normalized = okf_zvec.normalize_fts_text("Портал: поставщиками!")
        self.assertEqual(normalized, "портал поставщик")


if __name__ == "__main__":
    unittest.main()
