from Music_Generation.Analysis import MusicAnalysis
import unittest
import sys
from pathlib import Path
import pandas as pd

import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMusicAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_data = [60, 62, 64, 60, 62, 64, 65, 67, 69, 60, 62, 64]
        self.music_analysis = MusicAnalysis(self.test_data)

    def tearDown(self):
        self.test_data = None
        self.music_analysis = None

    def test_count_notes(self):
        merged_counts = self.music_analysis.count_notes()
        self.assertIsInstance(merged_counts, pd.DataFrame)
        self.assertIn('note', merged_counts.columns)
        self.assertIn('count', merged_counts.columns)
        self.assertEqual(merged_counts['count'].sum(), len(self.test_data))

        note_map = {'60': 3, '62': 3, '64': 3, '65': 1, '67': 1, '69': 1}

        for note_int, expected_count in note_map.items():
            name = self.music_analysis.char_notes.loc[self.music_analysis.char_notes['int'] == int(
                note_int), 'note'].iat[0]
            actual = merged_counts.loc[merged_counts['note']
                                       == name, 'count'].iat[0]
            self.assertEqual(actual, expected_count)

    def test_riffs(self):
        pattern_counts = self.music_analysis.riffs()
        expected_pattern = (60, 62, 64)
        self.assertIn(expected_pattern, pattern_counts)
        self.assertEqual(pattern_counts[expected_pattern], 3)

    def test_pitch(self):
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        self.music_analysis.pitch()
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("Average note value is", output)
        self.assertIn("which is between", output)
        self.assertIn("E4", output)

    def test_plot_music(self):
        """Test plot_music creates a figure with correct properties"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.close('all')
        self.music_analysis.plot_music()
        fig = plt.gcf()
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_size_inches().tolist(), [12, 4])
        ax = fig.gca()
        self.assertEqual(ax.get_xlabel(), "Note Position")
        self.assertEqual(ax.get_ylabel(), "Pitch")
        self.assertEqual(ax.get_title(), "Pitch Plot of Song")
        xticks = ax.get_xticks()
        self.assertIn(0, xticks)
        self.assertIn(len(self.test_data)-1, xticks)
        yticks = ax.get_yticks()
        self.assertIn(0, yticks)
        self.assertIn(127, yticks)
        plt.close('all')

    def test_counts_plot(self):
        """Test counts_plot creates a figure with correct properties"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.close('all')
        self.music_analysis.counts_plot()
        fig = plt.gcf()
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_size_inches().tolist(), [12, 6])
        ax = fig.gca()
        self.assertEqual(ax.get_xlabel(), "Note")
        self.assertEqual(ax.get_ylabel(), "Count")
        self.assertEqual(ax.get_title(), "Note Counts")
        bars = ax.patches
        self.assertGreater(len(bars), 0, "Should have bar patches")
        plt.close('all')

    def test_riffs_edge_case_short_data(self):
        """Test riffs with minimal data"""
        short_data = [60, 62, 64]
        analysis = MusicAnalysis(short_data)
        pattern_counts = analysis.riffs()
        self.assertEqual(len(pattern_counts), 1)
        self.assertIn((60, 62, 64), pattern_counts)

    def test_count_notes_single_note(self):
        """Test count_notes with repeated single note"""
        single_note_data = [60, 60, 60, 60]
        analysis = MusicAnalysis(single_note_data)
        result = analysis.count_notes()
        self.assertEqual(len(result), 1)
        self.assertEqual(result['count'].iat[0], 4)

    def test_pitch_with_integer_average(self):
        """Test pitch when average is exactly an integer"""
        integer_avg_data = [60, 60, 60]
        analysis = MusicAnalysis(integer_avg_data)

        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        analysis.pitch()
        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("60.0", output)
        self.assertIn("which is between", output)

    def test_init(self):
        """Test initialization stores data correctly"""
        test_data = [1, 2, 3, 4, 5]
        analysis = MusicAnalysis(test_data)
        self.assertEqual(analysis.data, test_data)

    def test_char_notes_structure(self):
        """Test that char_notes class variable is properly structured"""
        self.assertIsInstance(MusicAnalysis.char_notes, pd.DataFrame)
        self.assertIn('note', MusicAnalysis.char_notes.columns)
        self.assertIn('int', MusicAnalysis.char_notes.columns)
        self.assertGreater(len(MusicAnalysis.char_notes), 0)
