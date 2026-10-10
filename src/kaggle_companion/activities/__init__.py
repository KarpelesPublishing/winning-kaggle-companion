"""Chapter activities: one measured experiment per chapter.

Each module `chNN` runs a real method on seeded generated data (or a labelled
bundled dataset) and measures an effect the chapter teaches. The notebooks, the
interactive chapter demonstrations and the workbook pages all read these results.
See ACTIVITIES.md in the companion folder for the contract.
"""
import importlib

CHAPTERS = range(1, 66)


def load(number):
    """Import the activity module for chapter `number`."""
    return importlib.import_module(f"kaggle_companion.activities.ch{int(number):02d}")
