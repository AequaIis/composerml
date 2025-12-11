import unittest
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

from Music_Generation.midi_to_dataset import MidiDatasetLoader


class TestMidiToDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(__file__) 
        project_root = os.path.dirname(base_dir)
        cls.example_folder = os.path.join(project_root, "Music_Generation","example_music")

        # make sure the file exsists
        if not os.path.isdir(cls.example_folder):
            raise RuntimeError(f"Example MIDI folder not found: {cls.example_folder}")
        
        
        cls.loader = MidiDatasetLoader(cls.example_folder)
        
    @classmethod
    def tearDownClass(cls):
        cls.loader = None
    
    
    def setUp(self):
        """Run before each test."""
        
        self.loader = MidiDatasetLoader(self.example_folder)
    
    def tearDown(self):
        self.loader = None
        
    def test_get_midi_files_finds_example_files(self):
        midi_files = self.loader._get_midi_files()

        
        self.assertGreater(len(midi_files), 0)

        
        for path in midi_files:
            name = os.path.basename(path).lower()
            self.assertTrue(
                name.endswith(".mid") or name.endswith(".midi"),
                msg=f"Non-MIDI file returned: {path}",
            )
    def test_extract_notes_returns_int_pitches(self):
        midi_files = self.loader._get_midi_files()
        first_midi = midi_files[0]

        notes = self.loader._extract_notes(first_midi)
        self.assertIsInstance(notes, list)

        if notes:
            for n in notes:
                self.assertIsInstance(n, int)
                self.assertGreaterEqual(n, 0)
                self.assertLessEqual(n, 127)
    
    def test_init_populates_songs_from_midi_folder(self):
        songs = self.loader.songs

        self.assertIsInstance(songs, list)
        self.assertGreater(len(songs), 0)

        for song in songs:
            self.assertIsInstance(song, list)
            for n in song:
                self.assertIsInstance(n, int)
                self.assertGreaterEqual(n, 0)
                self.assertLessEqual(n, 127)


if __name__ == "__main__":
    unittest.main()