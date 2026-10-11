import unittest

from tempolens.tags import possessing_side, resolve_side


class TagTests(unittest.TestCase):
    def test_misspelled_away_name_still_resolves(self):
        # The tagging writes "Atletic Club"; the tracking metadata may not.
        for home, away in (("Barcelona", "Athletic Club"), ("FC Barcelona", "Athletic Club"),
                           ("Barcelona", "Atletic Club")):
            self.assertEqual(resolve_side("Barcelona", home, away), 0)
            self.assertEqual(resolve_side("Atletic Club", home, away), 1)

    def test_na_and_unknown_teams_are_ignored(self):
        self.assertIsNone(resolve_side("N/A", "Barcelona", "Athletic Club"))
        self.assertIsNone(resolve_side("", "Barcelona", "Athletic Club"))
        self.assertIsNone(resolve_side("Real Sociedad", "Barcelona", "Athletic Club"))

    def test_defensive_codes_point_at_the_other_side(self):
        self.assertEqual(possessing_side("BUILD UP", 0), 0)
        self.assertEqual(possessing_side("DEFENDING IN MIDDLE THIRD", 0), 1)
        self.assertEqual(possessing_side("Defensive Transition", 1), 0)
        self.assertIsNone(possessing_side("DEFENSIVE TRANSITION", None))


if __name__ == "__main__":
    unittest.main()
