import unittest

import okf_zvec
from okf_zvec_search import cli


class PackagingTests(unittest.TestCase):
    def test_legacy_module_keeps_the_cli_entrypoint(self):
        self.assertIs(okf_zvec.main, cli.main)


if __name__ == "__main__":
    unittest.main()
