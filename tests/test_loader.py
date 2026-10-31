"""Uji kecil untuk pembaca data: python -m unittest discover tests"""
import unittest

from src.data_loader import parse_period, to_number


class TestLoader(unittest.TestCase):
    def test_periode(self):
        self.assertEqual(str(parse_period("Januari 2019").date()), "2019-01-01")
        self.assertEqual(str(parse_period("2019-03").date()), "2019-03-01")
        self.assertEqual(str(parse_period("12/2020").date()), "2020-12-01")
        self.assertIsNone(parse_period("Soekarno-Hatta"))

    def test_angka(self):
        self.assertEqual(to_number("1.234.567"), 1234567.0)
        self.assertEqual(to_number("1,234"), 1234.0)
        self.assertEqual(to_number(25.0), 25.0)


if __name__ == "__main__":
    unittest.main()
