import unittest
import os
from unittest.mock import patch, MagicMock


from composerml.music_generation import MidiDatasetLoader


class TestMidiToDataset(unittest.TestCase):
    def setUp(self):
        # Creating some fake folder and data
        self.example_folder = "/fake/midi/folder"
        self.notes_1 = [60, 62, 64]
        self.notes_2 = [65, 67]


    def test_get_midi_files_finds_example_files(self):
        # Use __new__ so __init__ doesn't run
        loader = MidiDatasetLoader.__new__(MidiDatasetLoader)
        
        # Manually set folder_path
        loader.folder_path = self.example_folder

        # Pretend the folder contains some files
        with patch(
            "composerml.music_generation.midi_to_dataset.os.listdir",
            return_value=["track1.mid", "readme.txt", "track2.midi"],
        ):
            midi_files = loader._get_midi_files()

        self.assertGreater(len(midi_files), 0)

        for path in midi_files:
            name = os.path.basename(path).lower()
            self.assertTrue(
                name.endswith(".mid") or name.endswith(".midi"),
                msg=f"Non-MIDI file returned: {path}",
            )

   
    def test_extract_notes_returns_int_pitches(self):
        # Use __new__ so __init__ doesn't run
        loader = MidiDatasetLoader.__new__(MidiDatasetLoader)
        
        
        # Replaces the MidiFile() call inside _extract_notes
        with patch(
            "composerml.music_generation.midi_to_dataset.MidiFile"
        ) as mock_MidiFile:
            # Fake a MidiFile instance with tracks and messages
            # Make it so that it always returns this fake track
            mock_mid_instance = MagicMock()
            mock_MidiFile.return_value = mock_mid_instance

            msg1 = MagicMock(type="note_on", note=60, velocity=64)
            msg2 = MagicMock(type="note_on", note=62, velocity=70)
            msg3 = MagicMock(type="note_off", note=62, velocity=0)  # ignored

            track = [msg1, msg2, msg3]
            mock_mid_instance.tracks = [track]

            #read the fake track
            notes = loader._extract_notes("dummy.mid")

        self.assertIsInstance(notes, list)
        self.assertEqual(notes, [60, 62])

        

    def test_init_populates_songs_from_midi_folder(self):
        
        # patch so that os.path.isdir always returns True (a folder always exists)
        with patch(
            "composerml.music_generation.midi_to_dataset.os.path.isdir",
            return_value=True,
        ), patch.object(
            MidiDatasetLoader,
            "_get_midi_files",
            return_value=["track1.mid", "track2.mid"], # _get_midi_files returns two files
        ) as mock_get_files, patch.object(
            MidiDatasetLoader,
            "_extract_notes",
            side_effect=[self.notes_1, self.notes_2], # _extract_notes returns a set of fake notes per file
        ) as mock_extract:

            loader = MidiDatasetLoader(self.example_folder)

        # _get_midi_files should be called once in __init__
        mock_get_files.assert_called_once()
        # _extract_notes should be called once per file
        self.assertEqual(mock_extract.call_count, 2)

        songs = loader.songs

        self.assertIsInstance(songs, list)
        self.assertGreater(len(songs), 0)
        self.assertEqual(songs, [self.notes_1, self.notes_2])


if __name__ == "__main__":
    unittest.main()